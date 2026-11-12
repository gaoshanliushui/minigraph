"""
§5 Pregel 超步循环（通道触发版）

这一节是整个框架的**内核**。它不再是"查边列表决定下一批跑谁"，
而是"看通道版本决定谁该醒"。

每个超步只做三件事：

    ① 算触发   哪些节点的输入通道变了？
    ② 并发跑   同一超步的节点看到同一份状态快照
    ③ 收写入   一次性合并进通道，推进通道版本号

对照真实 LangGraph：langgraph/pregel/_algo.py 的
prepare_next_tasks / _triggers / apply_writes。

---

## 触发是怎么算出来的

每个节点订阅一组**触发通道**。节点该跑，当且仅当存在一个触发通道 `chan`
满足：

    channels[chan].is_available()                       # 通道有值
    and channel_versions[chan] > versions_seen[node][chan]   # 版本比水位新

`versions_seen[node]` 是**每节点的水位线**：这个节点上次跑的时候，
各个触发通道分别是第几版。节点跑完就把水位线推到当时的版本，
于是它不会重复触发 —— 除非通道又被写了一次（版本继续前进）。

这解释了循环为什么能工作：`tools -> agent` 这条回边每轮把
`branch:to:agent` 重写一遍，版本推过 agent 的水位线，agent 就再次被唤醒。

---

## 两条任务轨道

    PULL  静态可预测：节点订阅的通道有新版本 -> 触发
    PUSH  动态：Send 写进 __pregel_tasks -> 下一超步每个 Send 变一个任务

PUSH 任务**不走触发判定**（目标节点的 versions_seen 保持为空）。
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping

from .channels import (
    BaseChannel,
    ConcurrentWriteError,
    channel_for_field,
)
from .checkpoint import Checkpoint, InMemorySaver
from .constants import BRANCH_TO, END, MISSING, Reducer, START, TASKS
from .context import GraphInterrupt, _ctx
from .control import Command, Send
from .state import State

if TYPE_CHECKING:  # 仅作类型注解；与 builder 互相不依赖运行时
    from .builder import NodeSpec, StateGraph


@dataclass
class Task:
    """一个待执行的任务。"""

    name: str
    # PUSH 任务携带 Send 包；PULL 任务为 None
    send: Send | None = None
    # 同一超步内该节点的第几个任务（用于生成稳定的 resume 槽位）
    idx: int = 0

    @property
    def is_push(self) -> bool:
        return self.send is not None


class CompiledGraph:
    def __init__(self, graph: StateGraph, checkpointer: InMemorySaver | None = None):
        self.graph = graph
        self.reducers = graph.reducers
        self.checkpointer = checkpointer
        # 编译产物：通道表 + 节点规格表
        self.channels, self.nodes = graph._compile()
        # channel -> subscribers 反向索引。调度时只看「上一轮变了的通道」
        # 的订阅者，不必再扫描整张图（对齐 LangGraph 的 trigger_to_nodes）。
        self.trigger_to_nodes: dict[str, list[str]] = {}
        for name, spec in self.nodes.items():
            for chan in spec.triggers:
                self.trigger_to_nodes.setdefault(chan, []).append(name)
        self._validate()

    def _validate(self) -> None:
        used = set()
        for src, dst in self.graph.edges:
            used.update((src, dst))
        for starts, dst in self.graph.waiting_edges:
            used.update(starts)
            used.add(dst)
        used.update(b["source"] for b in self.graph.branches)
        # 条件边的目标也是合法入口；它可以是没有出边的终端节点，
        # 不应因为没再写 add_edge("worker", END) 就被判成悬空。
        for branch in self.graph.branches:
            used.update(branch["map"].values())
        dangling = [n for n in self.graph.nodes if n not in used]
        if dangling:
            raise ValueError(f"节点 {dangling} 没有连入任何边，图是悬空的")

    # -- 对外 API（与 LangGraph 同名）---------------------------------------
    def invoke(self, input: State | None = None, config: dict | None = None, **kw) -> State:
        for _ in self.stream(input, config, **kw):
            pass
        return dict(self._last_values)

    @property
    def interrupted(self) -> Any:
        """最近一次 invoke/stream 是否因 interrupt 挂起；返回挂起时传出的 value。"""
        return getattr(self, "_interrupted", None)

    def stream(self, input: State | None = None, config: dict | None = None, **kw):
        yield from self._run(input, {**(config or {}), **kw})

    def get_state(self, config: dict) -> Checkpoint:
        thread_id = (config or {}).get("thread_id")
        if not thread_id:
            raise ValueError("get_state 需要 config={'thread_id': ...}")
        if not self.checkpointer:
            raise ValueError("未配置 checkpointer")
        snap = self.checkpointer.get(thread_id, config.get("checkpoint_id"))
        if snap is None:
            raise ValueError(f"thread {thread_id} 还没有 checkpoint")
        return snap

    def get_state_history(self, config: dict) -> list[Checkpoint]:
        if not self.checkpointer:
            raise ValueError("未配置 checkpointer")
        return list(self.checkpointer.list(config["thread_id"]))

    def update_state(self, config: dict, values: dict, as_node: str | None = None) -> str:
        """
        人工改写状态并生成新 checkpoint（LangGraph 的 update_state）。

        在通道模型下，"改写状态"= 往状态通道里写（走各自的 reducer）；
        as_node 指定"假装是哪个节点写的"，决定接下来从哪继续 ——
        做法是写 branch:to:<as_node> 推进它的版本（保证它被触发），
        同时把其他节点的 versions_seen 补齐到当前版本（让它们不被触发）。
        """
        thread_id = config["thread_id"]
        if not self.checkpointer:
            raise ValueError("update_state 需要 checkpointer")
        snap = self.get_state(config).copy()

        channels, versions, seen = self._restore_channels(snap)
        extra_values = {k: v for k, v in snap.values.items() if k not in self.reducers}
        # ① 把 values 写进状态通道
        _write_state_channels(self.reducers, channels, values, node=as_node or "<update_state>")

        # ② 决定"接下来跑谁"
        if as_node:
            if as_node not in self.nodes:
                raise ValueError(f"未知节点 {as_node!r}")
            chan = BRANCH_TO.format(as_node)
            channels[chan].update([None])
            versions[chan] = _next_version(versions)
            # 其他节点的水位线补齐 -> 它们不会触发
            current = max(versions.values(), default=0)
            for name, spec in self.nodes.items():
                if name == as_node:
                    continue
                seen.setdefault(name, {}).update(
                    {c: versions.get(c, current) for c in spec.triggers}
                )

        snap.id = self.checkpointer.new_id()
        snap.step += 1
        snap.interrupt = None
        snap.values = _state_projection(channels, self.reducers)
        snap.channel_values = {c: ch.checkpoint() for c, ch in channels.items()}
        snap.channel_versions = dict(versions)
        snap.versions_seen = {k: dict(v) for k, v in seen.items()}
        snap.next = (as_node,) if as_node else snap.next
        snap.tasks = [as_node] if as_node else snap.tasks
        self.checkpointer.put(thread_id, snap)
        return snap.id

    # -- 通道状态的恢复 -----------------------------------------------------
    def _restore_channels(self, snap: Checkpoint):
        """
        从 checkpoint 重建运行期的通道状态。

        通道的"值"和"版本"是两回事：
          - 值   存在 channel_values 里
          - 版本 存在 channel_versions 里（单调递增，触发靠它）
          - 水位 存在 versions_seen 里（每个节点各存一份）
        """
        channels = {c: ch.copy() for c, ch in self.channels.items()}
        for chan, val in (snap.channel_values or {}).items():
            ch = channels.get(chan)
            if ch is None:
                continue
            channels[chan] = ch.from_checkpoint(val)

        versions = {c: 0 for c in channels}
        versions.update(snap.channel_versions or {})

        seen = {k: dict(v) for k, v in (snap.versions_seen or {}).items()}
        return channels, versions, seen

    # -- 主循环 -------------------------------------------------------------
    def _run(self, input: State | None, cfg: dict):
        graph, reducers, saver = self.graph, self.reducers, self.checkpointer
        thread_id = cfg.get("thread_id")
        debug = cfg.get("debug", False)
        limit = cfg.get("recursion_limit", 50)
        has_resume = "resume" in cfg
        resume_val = cfg.get("resume")

        self._interrupted = None
        snap = saver.get(thread_id, cfg.get("checkpoint_id")) if (saver and thread_id) else None

        # 运行期的通道状态（活实例）
        channels: dict[str, BaseChannel]
        versions: dict[str, int]
        seen: dict[str, dict[str, int]]
        # 未声明的输入键：允许被节点读到，但不允许被写（见 _write_state_channels）
        extra_values: dict[str, Any] = {}

        if input is None and snap is not None:
            # ---------- 从 checkpoint 恢复 ----------
            channels, versions, seen = self._restore_channels(snap)
            # checkpoint 目前不保存 updated_channels；恢复后的第一轮检查
            # 所有节点，之后仍走 changed-channel 反向索引。
            updated_channels: set[str] | None = None
            extra_values = {
                k: v for k, v in snap.values.items() if k not in reducers
            }
            step = snap.step
            if snap.interrupt:
                if not has_resume:
                    raise RuntimeError(
                        f"该 thread 停在 interrupt({snap.interrupt['value']!r})，"
                        f"需要 config={{'resume': ...}} 才能继续"
                    )
                _ctx()["resume"] = {snap.interrupt["slot"]: resume_val}
                if debug:
                    print(f"[resume] step={step} from interrupt at {snap.interrupt['node']}")
            else:
                _ctx()["resume"] = {}
                if debug:
                    print(f"[resume] step={step} versions={versions}")
        else:
            # ---------- 全新运行 ----------
            # 输入即初始状态。注意：不像节点那样强制校验——输入里带些
            # 图内只读的键（如 user_id）是允许的，只有"写"才要求声明。
            channels = {c: ch.copy() for c, ch in self.channels.items()}
            versions = {c: 0 for c in channels}
            seen = {}
            raw = dict(input or {})
            # 输入分两拨：已声明的写进状态通道（走 reducer），
            # 未声明的留作 extra_values —— 它们能被节点读到，但不允许被写。
            extra_values = {k: v for k, v in raw.items() if k not in reducers}
            _write_state_channels(
                reducers, channels,
                {k: v for k, v in raw.items() if k in reducers},
                node=START,
            )
            step = 0
            _ctx()["resume"] = {}
            if debug:
                print(f"[start] {raw}")

            # 入口：__start__ 写所有入口点的触发通道
            start_spec = self.nodes.get(START)
            updated_channels = set()
            if start_spec:
                start_version = _next_version(versions)
                for chan in start_spec.writers:
                    channels[chan].update([None])
                    versions[chan] = start_version
                    updated_channels.add(chan)
                seen[START] = {start_spec.triggers[0]: versions[start_spec.triggers[0]]}

        # 恢复时：被中断的任务必须落回它当初的槽位，才能取到 resume 值
        pending_slot = None
        if snap is not None and snap.interrupt:
            pending_slot = snap.interrupt["slot"]

        # ---------- 超步循环（Pregel / BSP）----------
        self._last_values = _state_projection(channels, reducers, extra_values)
        while True:
            # ① 算触发：上一轮哪些通道变了，它们的订阅者里谁该醒
            tasks = _prepare_next_tasks(
                channels, versions, seen, self.nodes,
                trigger_to_nodes=self.trigger_to_nodes,
                updated_channels=updated_channels,
            )
            if not tasks:
                break
            # 只有"还有任务要跑"才消耗步数预算。图恰好在第 limit 步结束
            # 是正常完成，不应在下一轮发现没有任务前被误判为死循环。
            if step >= limit:
                raise RuntimeError(f"超过 recursion_limit={limit}，疑似死循环")

            snapshot = _read_state(channels, reducers, extra_values)

            # ② 并发执行本批任务（同一超步共享同一快照）
            runs = []
            for task in tasks:
                is_pending = (
                    pending_slot
                    and task.name == snap.interrupt["node"]
                    and task.idx == snap.interrupt.get("idx", 0)
                    and task.is_push == snap.interrupt.get("push", False)
                )
                kind = "push" if task.is_push else "pull"
                slot = pending_slot if is_pending else f"{task.name}@{step + 1}#{kind}:{task.idx}"
                runs.append((task, slot))
            pending_slot = None

            results = _execute_batch(self.nodes, runs, snapshot)

            # ③ 收齐写入
            writes: list[tuple[str, str, Any]] = []   # (节点, 通道, 值)
            ran: list[str] = []
            hit = None
            hit_task = None
            hit_slot = None

            for (task, slot), out in zip(runs, results):
                name = task.name
                if isinstance(out, GraphInterrupt):
                    # 挂起 = 本超步作废：已提交的写入会被丢弃，恢复后整批重跑。
                    # 这正是 BSP 屏障（"一步要么全提交，要么全不提交"）的含义，
                    # 也是 checkpoint 存在**超步级**的根本原因 —— 所以节点要幂等。
                    hit, hit_task, hit_slot = out, task, slot
                    continue
                if isinstance(out, _NodeError):
                    raise RuntimeError(f"节点 {name!r} 执行失败：{out.exc}") from out.exc
                ran.append(name)
                node_writes = _normalize(name, out, snapshot, self.nodes, self.channels,
                                         reducers)
                writes.extend((name, c, v) for c, v in node_writes)

            # 有节点挂起 -> 整步作废：不应用任何写入、不推进版本、不动水位线。
            # 于是恢复时 prepare_next_tasks 会算出**同一批任务**，整批重跑。
            if hit is not None:
                # 挂起这一步的编号 = 它本该成为的超步号（step + 1）
                this_step = step + 1
                ckpt = Checkpoint(
                    id=saver.new_id() if saver else "ephemeral",
                    step=this_step, values=_state_projection(channels, reducers, extra_values),
                    next=tuple(t.name for t in tasks),
                    tasks=[t.send if t.is_push else t.name for t in tasks],
                    interrupt={"slot": hit_slot, "node": hit_task.name,
                               "idx": hit_task.idx, "push": hit_task.is_push,
                               "step": this_step, "value": hit.value},
                    channel_values={c: ch.checkpoint() for c, ch in channels.items()},
                    channel_versions=dict(versions),
                    versions_seen={k: dict(v) for k, v in seen.items()},
                )
                if saver and thread_id:
                    saver.put(thread_id, ckpt)
                self._last_values = _state_projection(channels, reducers, extra_values)
                self._interrupted = hit.value
                yield hit_task.name, this_step, {}
                return

            # ④ 更新水位线：本步跑过的节点把水位推到**它当时看到的版本**。
            #    注意顺序：必须用**写入前**的版本，而不是写入后的版本。
            #    否则节点自己写 branch:to:自己 就会把自己的触发信号抹掉，
            #    循环（回边）就再也转不起来。
            #    （对齐 LangGraph apply_writes：versions_seen 用的是
            #     checkpoint["channel_versions"][chan] 的写入前取值。）
            for task in tasks:
                # PUSH 的触发源是保留通道 __pregel_tasks，不是目标节点的
                # branch:to:X；因此它不能推进 X 的水位线（对齐真实 LangGraph）。
                if task.is_push:
                    seen.setdefault(task.name, {})
                    continue
                spec = self.nodes.get(task.name)
                if spec is None or not spec.triggers:
                    continue
                node_seen = seen.setdefault(task.name, {})
                for chan in spec.triggers:
                    if chan in versions:
                        node_seen[chan] = versions[chan]

            # 消费本步的触发通道。绝大多数通道无需处理；NamedBarrierValue
            # 在这里清空 seen、重新武装，下一轮汇聚才能再次等齐并触发。
            next_version = _next_version(versions)
            updated_channels = set()
            for task in tasks:
                if task.is_push:
                    continue
                spec = self.nodes.get(task.name)
                if spec is None:
                    continue
                for chan in spec.triggers:
                    ch = channels.get(chan)
                    if ch is not None and ch.consume():
                        versions[chan] = next_version
                        if ch.is_available():
                            updated_channels.add(chan)

            # ⑤ 应用写入（推进通道版本号）
            updated_channels.update(
                _apply_writes(channels, versions, writes, next_version=next_version)
            )

            # ⑥ 自清：本步没人写但仍然可用的通道，调 update([]) 让它抹掉自己。
            #    分支通道正是靠这一步做到"一个超步只活一步"。
            written = {c for _, c, _ in writes}
            for chan, ch in channels.items():
                if chan in written:
                    continue
                if ch.is_available() and ch.update([]):
                    versions[chan] = next_version
                    if ch.is_available():
                        updated_channels.add(chan)

            step += 1
            if debug:
                print(f"  step {step}: {ran} -> "
                      f"changed={[c for _, c, _ in writes]}")

            self._last_values = _state_projection(channels, reducers, extra_values)
            yield ",".join(ran), step, {
                c: channels[c].get() for c in dict.fromkeys(c for _, c, _ in writes)
                if c in reducers and c in channels and channels[c].is_available()
            }

            if saver and thread_id:
                upcoming = _prepare_next_tasks(
                    channels, versions, seen, self.nodes,
                    trigger_to_nodes=self.trigger_to_nodes,
                    updated_channels=updated_channels,
                )
                saver.put(thread_id, Checkpoint(
                    id=saver.new_id(), step=step,
                    values=_state_projection(channels, reducers, extra_values),
                    next=tuple(t.name for t in upcoming),
                    tasks=[t.send if t.is_push else t.name for t in upcoming],
                    channel_values={c: ch.checkpoint() for c, ch in channels.items()},
                    channel_versions=dict(versions),
                    versions_seen={k: dict(v) for k, v in seen.items()},
                ))

        _ctx()["slot"] = None


class _NodeError:
    def __init__(self, exc: BaseException):
        self.exc = exc


# ---------------------------------------------------------------------------
# 触发判定
# ---------------------------------------------------------------------------

def _triggers(channels: dict[str, BaseChannel], versions: dict[str, int],
              seen: dict[str, int] | None, trig: list[str]) -> bool:
    """
    节点该被唤醒吗？

    对每个触发通道 chan：
        通道可用 **且** 版本比该节点的水位线新。

    seen 为 None 表示这个节点**从没跑过** —— 此时只要通道可用就触发
    （分支通道存的值通常是 None 但它"可用"，这正是入口能被唤醒的原因）。
    """
    if seen is None:
        for chan in trig:
            ch = channels.get(chan)
            if ch is not None and ch.is_available():
                return True
        return False
    for chan in trig:
        ch = channels.get(chan)
        if ch is None or not ch.is_available():
            continue
        if versions.get(chan, 0) > seen.get(chan, 0):
            return True
    return False


def _prepare_next_tasks(channels: dict[str, BaseChannel], versions: dict[str, int],
                        seen: dict[str, dict[str, int]],
                        nodes: dict[str, NodeSpec],
                        trigger_to_nodes: Mapping[str, list[str]] | None = None,
                        updated_channels: set[str] | None = None) -> list[Task]:
    """
    算出这一超步要跑哪些任务。

    两条轨道（对齐 LangGraph 的 prepare_next_tasks）：
      PUSH —— __pregel_tasks 里的每个 Send 变成一个任务
      PULL —— 订阅通道有新版本的节点

    有 updated_channels 和 trigger_to_nodes 时，只检查「变化通道的订阅者」。
    这是候选节点优化；版本与水位线仍由 _triggers 做最终判定。
    """
    tasks: list[Task] = []

    # --- PUSH 轨：Send 队列 ---
    tc = channels.get(TASKS)
    if tc is not None and tc.is_available():
        for idx, packet in enumerate(tc.get()):
            if not isinstance(packet, Send):
                continue
            if packet.node not in nodes:
                continue
            tasks.append(Task(name=packet.node, send=packet, idx=idx))

    # --- PULL 轨：通道版本触发 ---
    if updated_channels is not None and trigger_to_nodes is not None:
        candidates = {
            name
            for chan in updated_channels
            for name in trigger_to_nodes.get(chan, ())
        }
    else:
        candidates = set(nodes)

    # 按节点名排序，保证同一批任务的顺序是确定的（恢复时槽位才对得上）
    for name in sorted(candidates):
        spec = nodes[name]
        if spec.is_start:
            continue
        node_seen = seen.get(name)
        if _triggers(channels, versions, node_seen, spec.triggers):
            tasks.append(Task(name=name, idx=0))

    return tasks


def _next_version(versions: dict[str, int]) -> int:
    """全局版本计数器：所有通道共用一个递增序列（对齐 LangGraph）。"""
    return max(versions.values(), default=0) + 1


def _apply_writes(channels: dict[str, BaseChannel], versions: dict[str, int],
                  writes: list[tuple[str, str, Any]],
                  next_version: int | None = None) -> set[str]:
    """
    把这一超步的写入合并进通道，并推进被写通道的版本号。

    版本推进是**触发**的全部机制：写一次 = 版本前进一步 = 下游水位线过期。
    同一超步内所有变化通道共享同一个新版本号（对齐 LangGraph）。

    返回本步更新且仍可用的通道，供下一轮通过 trigger_to_nodes 找订阅者。
    """
    by_chan: dict[str, list[tuple[str, Any]]] = {}
    for node, chan, val in writes:
        by_chan.setdefault(chan, []).append((node, val))

    version = _next_version(versions) if next_version is None else next_version
    updated: set[str] = set()
    for chan, items in by_chan.items():
        ch = channels.get(chan)
        if ch is None:
            raise ValueError(f"节点写了未知通道 {chan!r}")
        vals = [v for _, v in items]
        try:
            changed = ch.update(vals)
        except ConcurrentWriteError as exc:
            who = [n for n, _ in items]
            raise ConcurrentWriteError(
                f"同一超步有多个节点（{who}）写字段 {ch.key!r}；"
                f"并发写必须给它配 reducer（如 operator.add / merge_dict）"
                f"（LangGraph 对应错误：At key {ch.key!r}: "
                f"Can receive only one value per step.）"
            ) from exc
        if changed:
            versions[chan] = version
            if ch.is_available():
                updated.add(chan)
    return updated


# ---------------------------------------------------------------------------
# 状态读写
# ---------------------------------------------------------------------------

def _read_state(channels: dict[str, BaseChannel], reducers: Mapping[str, Reducer],
                extra_values: Mapping[str, Any]) -> dict:
    """
    读出这一超步的状态快照。所有节点看到同一份 —— 这就是同步屏障。
    """
    state = {f: channels[f].get() for f in reducers
             if f in channels and channels[f].is_available()}
    state.update(extra_values)
    return state


def _state_projection(channels: dict[str, BaseChannel],
                      reducers: Mapping[str, Reducer],
                      extra_values: Mapping[str, Any] | None = None) -> dict:
    """
    状态通道 -> 扁平的状态字典（给 get_state / checkpoint.values 用）。

    只含**状态字段**和只读的额外输入键，不含 branch:to:* / join:* /
    __pregel_tasks 这些触发通道 —— 它们是内核实现细节，不该泄漏到用户状态里。
    """
    out: dict[str, Any] = {}
    for f in reducers:
        if f in channels and channels[f].is_available():
            out[f] = channels[f].get()
    out.update(extra_values or {})
    return out


def _write_state_channels(reducers: Mapping[str, Reducer], channels: dict[str, BaseChannel],
                          update: dict, *, node: str) -> None:
    """
    把 update 写进状态通道（走各自的 reducer）。

    写未声明的字段要报错 —— 这是作者在"表态"：这个字段该用哪个 reducer。
    """
    for key, val in update.items():
        if key not in reducers:
            raise ValueError(
                f"节点 {node!r} 写了未声明的字段 {key!r}；已声明字段：{sorted(reducers)}"
            )
        ch = channels.get(key)
        if ch is None:
            ch = channel_for_field(key, reducers[key])
            channels[key] = ch
        ch.update([val])


# ---------------------------------------------------------------------------
# 并发执行
# ---------------------------------------------------------------------------

def _run_task(name: str, fn: Callable, arg: Any, retry: Any, slot: str,
              resume: dict | None = None):
    """在一个线程里跑节点。线程本地要重新注入 slot / resume（线程池不继承调用方上下文）。"""
    _ctx()["slot"] = slot
    _ctx()["resume"] = dict(resume or {})
    try:
        return _run_with_retry(name, fn, arg, retry)
    except GraphInterrupt as gi:
        return gi
    except Exception as exc:  # noqa: BLE001
        return _NodeError(exc)


def _execute_batch(nodes: dict[str, NodeSpec], runs, snapshot: dict):
    """
    同一超步内的节点并发执行。

    注意：并发只发生在"单步内"，跨超步仍然是严格的 BSP 同步屏障，
    因此结果与执行顺序无关（reducer 决定合并语义）。
    """
    def arg_for(task):
        return snapshot if not task.is_push else _wrap_send(task.send, snapshot)

    # 恢复值必须显式带进工作线程：threading.local 不会被线程池继承
    resume = dict(_ctx().get("resume") or {})

    if len(runs) == 1:
        (task, slot), = runs
        spec = nodes[task.name]
        return [_run_task(task.name, spec.fn, arg_for(task), spec.retry, slot, resume)]

    with ThreadPoolExecutor(max_workers=min(16, len(runs))) as pool:
        futures = []
        for (task, slot) in runs:
            spec = nodes[task.name]
            futures.append(pool.submit(_run_task, task.name, spec.fn, arg_for(task),
                                       spec.retry, slot, resume))
        return [f.result() for f in futures]


def _wrap_send(send: Send, values: State) -> Any:
    """Send 的 arg 作为节点入参，同时把主状态挂到 __state__ 供只读访问。"""
    if isinstance(send.arg, Mapping):
        return {**values, **send.arg, "__state__": values}
    return _SendView(send.arg, values)


class _SendView(dict):
    def __init__(self, arg, state):
        super().__init__(state)
        self.arg = arg


# ---------------------------------------------------------------------------
# 节点返回值 -> 通道写入
# ---------------------------------------------------------------------------

def _normalize(node: str, out: Any, snapshot: dict, nodes: dict[str, NodeSpec],
               channels: dict[str, BaseChannel],
               reducers: Mapping[str, Reducer]) -> list[tuple[str, Any]]:
    """
    把节点的返回值翻译成一串「(通道, 值)」写入。

    这是"边 = 一次通道写入"落地的地方：
      - 返回 dict            -> 写状态通道
      - Command(goto=X)      -> 写 branch:to:X（触发 X）
      - Command(update=...)  -> 写状态通道
      - Send(...)            -> 写 __pregel_tasks（PUSH 轨）
      - 静态边 / 条件边      -> 写 branch:to:<目标> / join:... 通道
    """
    spec = nodes.get(node)
    writes: list[tuple[str, Any]] = []
    update: dict = {}

    # ---- Command：更新状态 + 决定去哪，两件事一起做 ----
    if isinstance(out, Command):
        update = dict(out.update or {})
        goto = out.goto
        if goto is not None:
            targets = list(goto) if isinstance(goto, (list, tuple)) else [goto]
            for t in targets:
                if isinstance(t, Send):
                    writes.append((TASKS, t))
                elif t != END:
                    chan = BRANCH_TO.format(t)
                    if chan not in channels:
                        raise ValueError(f"Command(goto={t!r}) 指向未知节点 {t!r}")
                    writes.append((chan, None))
        writes.extend(_state_writes(update, node, reducers))
        # Command **不抑制**静态边与条件边（对齐真实 LangGraph 的实测行为）
        writes.extend(_control_writes(spec, snapshot, update, channels, reducers))
        return writes

    # ---- 直接返回 list[Send]：等价于"条件边里做扇出" ----
    # 这是 minigraph 的扩展（真实 LangGraph 会抛 InvalidUpdateError）。
    # 它**抑制**本节点的静态边与条件边：返回 Send 就意味着"我自己在派活"。
    if isinstance(out, (list, tuple)) and all(isinstance(x, Send) for x in out):
        return [(TASKS, s) for s in out]

    # ---- 普通返回值 ----
    if out is not None:
        if not isinstance(out, Mapping):
            raise TypeError(f"节点 {node!r} 必须返回 dict / None / Command，收到 {type(out).__name__}")
        update = dict(out)

    writes.extend(_state_writes(update, node, reducers))
    writes.extend(_control_writes(spec, snapshot, update, channels, reducers))
    return writes


def _state_writes(update: Mapping[str, Any], node: str,
                  reducers: Mapping[str, Reducer]) -> list[tuple[str, Any]]:
    """
    状态字段 -> 状态通道写入。

    写未声明的字段要报错：这是作者在"表态" —— 这个字段该用哪个 reducer。
    （输入里的未声明键是另一回事，它们只读、不许写，见 _run。）

    `{"key": None}` 视作"这次不改"（跳过）。真 LangGraph 会把 None 写进通道，
    这里跳过是为了并行友好 —— 两个并行分支各写一个字段时不该互相打架。
    """
    unknown = [k for k, v in update.items() if k not in reducers and v is not None]
    if unknown:
        raise ValueError(
            f"节点 {node!r} 写了未声明的字段 {sorted(unknown)}；已声明字段：{sorted(reducers)}"
        )
    return [(k, v) for k, v in update.items() if v is not None]


def _fresh_view(snapshot: dict, update: Mapping[str, Any],
                reducers: Mapping[str, Reducer]) -> dict:
    """
    算出「本节点写入合并之后」的状态视图，给条件边看。

    关键点：合并必须**经过 reducer**。节点写 {"n": 1}、reducer 是
    operator.add 时，视图里的 n 应该是 old+1，而不是 1 —— 条件边据此
    才能正确判断"还要不要再来一轮"。
    """
    view = dict(snapshot)
    for key, val in update.items():
        if val is None:
            continue
        reducer = reducers.get(key)
        if reducer is None:
            continue
        old = view.get(key, MISSING)
        # 与 BinaryOperatorAggregate.update 的首次写入捷径保持一致：
        # operator.add(MISSING, 1) 并没有意义，首次值应当直接是 1。
        view[key] = val if old is MISSING else reducer(old, val)
    return view


def _control_writes(spec: NodeSpec | None, snapshot: dict, update: Mapping[str, Any],
                    channels: dict[str, BaseChannel],
                    reducers: Mapping[str, Reducer]) -> list[tuple[str, Any]]:
    """
    控制流写入：静态边 + 条件边。

    静态边 -> 写 branch:to:<目标>；屏障边 -> 写 join:... 通道（值是源节点自己的名字）。
    条件边 -> 路径函数看到的是「本节点更新合并进快照之后」的状态。
    """
    if spec is None:
        return []
    writes: list[tuple[str, Any]] = []

    # 静态 writers
    for chan in spec.writers:
        if chan.startswith("join:"):
            writes.append((chan, spec.name))      # "我到了"
        else:
            writes.append((chan, None))

    # 条件边
    for br in spec.branches:
        # 条件边看到的是「本节点的写入**经过 reducer 合并之后**」的状态。
        # 不能直接用原始 update 覆盖快照：节点写 {"n": 1} 而 reducer 是
        # operator.add 时，合并结果是 old+1，而不是 1。
        # （对齐 LangGraph 条件边的 fresh=True 读：先把写入应用到通道副本上再读。）
        view = _fresh_view(snapshot, update, reducers)
        decision = br["path"](view)
        picks = decision if isinstance(decision, (list, tuple, set)) else [decision]
        for p in picks:
            if isinstance(p, Send):
                writes.append((TASKS, p))
                continue
            if p == END:
                continue
            target = br["map"].get(p, p) if br["map"] else p
            if target == END:
                continue
            chan = BRANCH_TO.format(target)
            if chan not in channels:
                raise ValueError(f"条件边指向未知节点 {target!r}")
            writes.append((chan, None))

    return writes


def _run_with_retry(node: str, fn: Callable, state: Any, retry: Any):
    if retry is None:
        return fn(state)
    policy = retry if isinstance(retry, Mapping) else {"max_attempts": int(retry)}
    attempts = int(policy.get("max_attempts", 3))
    backoff = float(policy.get("backoff", 0.0))
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn(state)
        except GraphInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001
            last = exc
            if i + 1 < attempts and backoff:
                time.sleep(backoff * (2 ** i))
    raise RuntimeError(f"节点 {node!r} 重试 {attempts} 次仍失败：{last}") from last
