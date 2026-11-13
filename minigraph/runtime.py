"""
§5 Pregel 超步循环（通道触发版）

这一节是整个框架的**内核**。它不再是"查边列表决定下一批跑谁"，
而是"看通道版本决定谁该醒"。

## 一个超步的骨架

    ① 算触发   哪些节点的触发通道有新版本了？
    ② 读快照   本超步共享的那一份状态
    ③ 并发跑   同一批任务并发执行
    ④ 收写入   把节点返回值翻译成「(通道, 值)」
    ⑤ 落盘     应用写入、推版本号、推水位线
    ⑥ 清旧     没人写的触发通道自己抹掉

前三步是"读"，后三步是"写"。**⑤⑥ 内部的顺序不能换**，
每一步为什么在那儿，写在对应函数的 docstring 里。

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
from dataclasses import dataclass, field
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


@dataclass
class _Hit:
    """本超步里挂起（interrupt）的那个任务。"""

    task: Task
    slot: str
    value: Any


@dataclass
class _RuntimeState:
    """
    一次运行中的全部可变状态。

    这几张表必须**一起**移动 —— 算触发的、写通道的、算快照的都要看它们，
    拆成三四个参数到处传只会把调用点撑长，所以打包成一个对象。

    channels  通道当前值（数据通道存状态，触发通道存"信号"）
    versions  通道版本号，单调递增；触发全靠它
    seen      每节点的水位线：它上次跑时看到的版本
    updated   上一步变了的通道；下一轮算触发时只看它们的订阅者
              （None = 不做增量优化，全量扫描所有节点）
    extra     未声明的输入键（只读，不许写）
    step      当前超步号
    """

    nodes: dict[str, NodeSpec]
    reducers: Mapping[str, Reducer]
    channels: dict[str, BaseChannel]
    versions: dict[str, int]
    seen: dict[str, dict[str, int]]
    extra: dict[str, Any] = field(default_factory=dict)
    updated: set[str] | None = None
    step: int = 0

    # -- 读 ---------------------------------------------------------------
    def read(self) -> dict:
        """本超步的状态快照：所有节点看到同一份 —— 这就是同步屏障。"""
        return _read_state(self.channels, self.reducers, self.extra)

    def projection(self) -> dict:
        """扁平状态字典（给 get_state / checkpoint.values 用）。"""
        return _state_projection(self.channels, self.reducers, self.extra)

    def next_version(self) -> int:
        """全局版本计数器：所有通道共用一个递增序列（对齐 LangGraph）。"""
        return _next_version(self.versions)

    # -- ⑤ 推水位线 -------------------------------------------------------
    def advance_watermarks(self, tasks: list[Task]) -> None:
        """
        本步跑过的节点，把水位线推到**它当时看到的版本**。

        注意顺序：必须用**写入前**的版本，而不是写入后的版本。
        否则节点自己写 branch:to:自己 就会把自己的触发信号抹掉，
        循环（回边）就再也转不起来。
        （对齐 LangGraph apply_writes：versions_seen 用的是
         checkpoint["channel_versions"][chan] 的写入前取值。）
        """
        for task in tasks:
            # PUSH 的触发源是保留通道 __pregel_tasks，不是目标节点的
            # branch:to:X；因此它不能推进 X 的水位线（对齐真实 LangGraph）。
            if task.is_push:
                self.seen.setdefault(task.name, {})
                continue
            spec = self.nodes.get(task.name)
            if spec is None or not spec.triggers:
                continue
            node_seen = self.seen.setdefault(task.name, {})
            for chan in spec.triggers:
                if chan in self.versions:
                    node_seen[chan] = self.versions[chan]

    # -- ⑤ 消费触发通道 ---------------------------------------------------
    def consume_triggers(self, tasks: list[Task], next_version: int) -> None:
        """
        订阅者跑完了，把触发通道"消费"掉。

        绝大多数通道无需处理；NamedBarrierValue 在这里清空 seen、
        重新武装，下一轮汇聚才能再次等齐并触发。
        """
        for task in tasks:
            if task.is_push:
                continue
            spec = self.nodes.get(task.name)
            if spec is None:
                continue
            for chan in spec.triggers:
                ch = self.channels.get(chan)
                if ch is not None and ch.consume():
                    self.versions[chan] = next_version
                    if ch.is_available():
                        self.updated.add(chan)

    # -- ⑥ 清理过期通道 ---------------------------------------------------
    def expire_unwritten(self, written: set[str], next_version: int) -> None:
        """
        本步没人写、但仍然可用的通道，调 update([]) 让它抹掉自己。

        分支通道正是靠这一步做到"一个超步只活一步"。
        """
        for chan, ch in self.channels.items():
            if chan in written:
                continue
            if ch.is_available() and ch.update([]):
                self.versions[chan] = next_version
                if ch.is_available():
                    self.updated.add(chan)

    # -- 调试 -------------------------------------------------------------
    def dump(self, tag: str) -> None:
        """
        把三张表打出来 —— 调试超步循环时最有用的一张图。

        只打**触发通道**（branch:to:* / join:* / __pregel_tasks）：
        数据通道（状态字段）一旦被写就**始终可用**，而且从不参与触发判定，
        混进来只会淹没真正要看的信息 —— 谁被唤醒了、水位线推到哪了。
        """
        trig = {c: ch for c, ch in self.channels.items() if c not in self.reducers}
        print(f"  {tag}")
        print("    versions:", {k: v for k, v in sorted(self.versions.items())
                              if k not in self.reducers})
        print("    seen    :", {k: dict(sorted(v.items())) for k, v in sorted(self.seen.items())})
        print("    avail   :", {c: trig[c].get() for c in sorted(trig)
                               if trig[c].is_available()})


class CompiledGraph:
    def __init__(self, graph: StateGraph, checkpointer: InMemorySaver | None = None):
        self.graph = graph
        self.reducers = graph.reducers
        # 字段类型：给 reducer 通道算初值（typ()）。裸 dict 形态的 schema 没有。
        self.field_types = graph.field_types
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

        在通道模型下，"改写状态"= 往状态通道里写（走各自的 reducer）。

        `as_node` 表示"假装这条更新是**哪个节点写的"。既然是它写的，
        被点亮的就是**它的出边**，而不是它自己：

          - 静态边 -> 写 branch:to:<后继>
          - 条件边 -> 按**合并后**的状态重新求值，写被选中的目标

        于是接下来跑的是 as_node 的**后继**。
        （对齐真实 LangGraph：update_state 执行的是 as_node 的 writers，
        不是 as_node 自己 —— 详见 langgraph/pregel/main.py 的
        bulk_update_state：`writers = self.nodes[as_node].flat_writers`。）

        不给 as_node 时，按水位线推断"最后一个写过状态的节点"
        （对齐 LangGraph 的 _infer_as_node）；几个节点同时写过、说不清是谁
        写的，就报错请显式指定。
        """
        thread_id = config["thread_id"]
        if not self.checkpointer:
            raise ValueError("update_state 需要 checkpointer")
        snap = self.checkpointer.get(thread_id, config.get("checkpoint_id"))
        if snap is None:
            # LangGraph 允许在还没有快照的 thread 上 update_state：
            # 以空快照为底，这次的 values 就是初始状态。
            snap = Checkpoint(id=self.checkpointer.new_id(), step=-1,
                              values={}, next=(), tasks=[])
        else:
            snap = snap.copy()

        channels, versions, seen = self._restore_channels(snap)
        extra_values = {k: v for k, v in snap.values.items() if k not in self.reducers}
        # ① 把 values 写进状态通道
        _write_state_channels(self.reducers, channels, values,
                              node=as_node or "<update_state>",
                              types=self.field_types)

        # ② 决定"接下来跑谁"
        if as_node is None:
            as_node = self._infer_as_node(seen)
            if as_node is None:
                raise ValueError(
                    "无法推断 as_node（多个节点同时写过状态）；请显式指定 as_node"
                )
        if as_node not in self.nodes:
            raise ValueError(f"未知节点 {as_node!r}")
        spec = self.nodes[as_node]

        # ③ 点亮 as_node 的出边。条件边看到的是「values 合并进旧状态之后」
        #    的视图 —— 和节点正常跑完时看到的是同一份东西。
        writes = _control_writes(spec, dict(snap.values), values,
                                 channels, self.reducers)
        next_version = _next_version(versions)
        updated = _apply_writes(channels, versions,
                                [(as_node, c, v) for c, v in writes],
                                next_version=next_version)

        # ④ 本次更新没写到的触发通道自清。否则上一次挂起留下的旧信号会把
        #    无关节点也唤醒（对齐 apply_writes 的 "channels that weren't
        #    updated in this step are notified of a new step"）。
        rt = _RuntimeState(
            nodes=self.nodes, reducers=self.reducers, channels=channels,
            versions=versions, seen=seen, extra=extra_values,
            updated=set(updated), step=snap.step,
        )
        rt.expire_unwritten({c for c, _ in writes}, next_version)

        # ⑤ next = 被这次写入点亮的节点（as_node 自己不会被点亮）
        upcoming = self._next_tasks(rt)

        snap.id = self.checkpointer.new_id()
        snap.step += 1
        snap.interrupt = None
        snap.interrupts = []
        snap.resumes = {}
        snap.values = _state_projection(channels, self.reducers, extra_values)
        snap.channel_values = {c: ch.checkpoint() for c, ch in channels.items()}
        snap.channel_versions = dict(versions)
        snap.versions_seen = {k: dict(v) for k, v in seen.items()}
        snap.next = tuple(t.name for t in upcoming)
        snap.tasks = [t.send if t.is_push else t.name for t in upcoming]
        self.checkpointer.put(thread_id, snap)
        return snap.id

    def _infer_as_node(self, seen: dict[str, dict[str, int]]) -> str | None:
        """
        推断"最后一个写过状态的节点"（对齐 LangGraph 的 _infer_as_node）。

        判据：节点水位线里**最高的那个版本**属于谁，谁就是最后跑的那个。
        最高版本有两个节点并列 -> 说不清是谁写的 -> 返回 None，
        让调用方要求显式指定 as_node。

        一个节点都还没跑过时，入口伪节点 __start__ 算"最后写的那个"
        （它的出边就是所有入口点）。
        """
        if len(self.nodes) == 1:
            return next(iter(self.nodes))
        marks = sorted(
            (v, name)
            for name, node_seen in seen.items()
            if name in self.nodes
            for v in node_seen.values()
        )
        if not marks:
            return START if START in self.nodes else None
        if len(marks) == 1 or marks[-1][0] != marks[-2][0]:
            return marks[-1][1]
        return None

    # -- 通道状态的恢复 -----------------------------------------------------
    def _restore_channels(self, snap: Checkpoint):
        """
        从 checkpoint 重建运行期的通道状态。

        通道的"值"和"版本"是两回事：
          - 值   存在 channel_values 里
          - 版本  存在 channel_versions 里（单调递增，触发靠它）
          - 水位  存在 versions_seen 里（每个节点各存一份）
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

    # -- 开场：全新运行 or 从 checkpoint 恢复 ------------------------------
    def _begin(self, input: State | None, snap: Checkpoint | None,
               cfg: dict) -> _RuntimeState:
        """
        准备一次运行的初始状态。

        返回的 _RuntimeState 里 `updated` 为 None 表示"下一轮算触发时
        全量扫描所有节点"（恢复场景还没建起 changed-channel 索引）。
        """
        reducers = self.reducers
        has_resume = "resume" in cfg
        resume_val = cfg.get("resume")
        debug = cfg.get("debug", False)

        if input is None and snap is not None:
            # ---------- 从 checkpoint 恢复 ----------
            channels, versions, seen = self._restore_channels(snap)
            extra_values = {
                k: v for k, v in snap.values.items() if k not in reducers
            }
            rt = _RuntimeState(
                nodes=self.nodes, reducers=reducers,
                channels=channels, versions=versions, seen=seen,
                extra=extra_values, updated=None, step=snap.step,
            )
            # checkpoint 目前不保存 updated_channels；恢复后的第一轮检查
            # 所有节点，之后仍走 changed-channel 反向索引。
            if snap.interrupt:
                pending = list(snap.interrupts or [snap.interrupt])
                # 已经答复过的值要**保留**：整步回滚意味着挂起的节点会重跑，
                # 重跑时还得拿到当初那个值。
                store = {k: list(v) for k, v in (snap.resumes or {}).items()}
                if not has_resume and not store:
                    raise RuntimeError(
                        f"该 thread 停在 interrupt({snap.interrupt['value']!r})，"
                        f"需要 config={{'resume': ...}} 才能继续"
                    )
                if has_resume:
                    store = _merge_resume(store, pending, resume_val)
                _ctx()["resume"] = store
                if debug:
                    print(f"[resume] step={rt.step} from interrupt at "
                          f"{[p['node'] for p in pending]}")
            else:
                _ctx()["resume"] = {}
                if debug:
                    print(f"[resume] step={rt.step} versions={versions}")
            return rt

        # ---------- 全新运行 ----------
        # 输入即初始状态。注意：不像节点那样强制校验——输入里带些
        # 图内只读的键（如 user_id）是允许的，只有"写"才要求声明。
        channels = {c: ch.copy() for c, ch in self.channels.items()}
        versions = {c: 0 for c in channels}
        raw = dict(input or {})
        # 输入分两拨：已声明的写进状态通道（走 reducer），
        # 未声明的留作 extra_values —— 它们能被节点读到，但不允许被写。
        extra_values = {k: v for k, v in raw.items() if k not in reducers}
        _write_state_channels(
            reducers, channels,
            {k: v for k, v in raw.items() if k in reducers},
            node=START,
            types=self.field_types,
        )
        rt = _RuntimeState(
            nodes=self.nodes, reducers=reducers,
            channels=channels, versions=versions, seen={},
            extra=extra_values, updated=set(), step=0,
        )
        _ctx()["resume"] = {}
        if debug:
            print(f"[start] {raw}")

        # 入口：__start__ 写所有入口点的触发通道
        start_spec = self.nodes.get(START)
        if start_spec:
            start_version = rt.next_version()
            for chan in start_spec.writers:
                rt.channels[chan].update([None])
                rt.versions[chan] = start_version
                rt.updated.add(chan)
            rt.seen[START] = {start_spec.triggers[0]: rt.versions[start_spec.triggers[0]]}
        return rt

    # -- 主循环 -------------------------------------------------------------
    def _run(self, input: State | None, cfg: dict):
        """
        超步循环。骨架只有下面这几行，其余都交给 _step_* 系列：

            ① 算触发 -> ② 读快照 -> ③ 并发跑 -> ④ 收写入 -> ⑤⑥ 落盘

        ⑤⑥ 内部的顺序不能换，每一步为什么在那儿写在对应函数的 docstring 里。
        """
        saver, thread_id = self.checkpointer, cfg.get("thread_id")
        debug = cfg.get("debug", False)
        limit = cfg.get("recursion_limit", 50)

        self._interrupted = None
        snap = saver.get(thread_id, cfg.get("checkpoint_id")) if (saver and thread_id) else None

        rt = self._begin(input, snap, cfg)

        self._last_values = rt.projection()
        while True:
            # ① 算触发：上一轮哪些通道变了，它们的订阅者里谁该醒
            tasks = self._next_tasks(rt)
            if not tasks:
                break
            # 只有"还有任务要跑"才消耗步数预算。图恰好在第 limit 步结束
            # 是正常完成，不应在下一轮发现没有任务前被误判为死循环。
            if rt.step >= limit:
                raise RuntimeError(f"超过 recursion_limit={limit}，疑似死循环")

            if debug:
                rt.dump(f"step {rt.step + 1} 触发前")

            # ② 读快照：同一超步的所有节点看到同一份状态
            snapshot = rt.read()

            # ③ 并发跑本批任务
            runs = self._assign_slots(tasks, rt.step, snap)
            # 本超步开始前的 resume 队列快照。挂起时整步回滚，已经被
            # 消费掉的 resume 值必须还原 —— 否则重跑时取不到。
            store_before = {k: list(v) for k, v in (_ctx().get("resume") or {}).items()}
            results = _execute_batch(self.nodes, runs, snapshot)

            # ④ 收写入：把节点返回值翻译成「(节点, 通道, 值)」
            writes, ran, hits = self._collect_writes(runs, results, snapshot)

            # 有节点挂起 -> 整步作废：不应用任何写入、不推进版本、不动水位线。
            # 于是恢复时算触发会得出**同一批任务**，整批重跑。
            if hits:
                self._suspend(rt, runs, hits, store_before, saver, thread_id)
                yield hits[0].task.name, rt.step + 1, {}
                return

            # ⑤⑥ 落盘：推水位线、消费屏障、应用写入、清理过期通道
            self._advance(rt, tasks, writes)

            rt.step += 1
            if debug:
                print(f"  step {rt.step}: {ran} -> changed={[c for _, c, _ in writes]}")

            self._last_values = rt.projection()
            yield ",".join(ran), rt.step, {
                c: rt.channels[c].get() for c in dict.fromkeys(c for _, c, _ in writes)
                if c in self.reducers and c in rt.channels and rt.channels[c].is_available()
            }

            if saver and thread_id:
                self._save(rt, saver, thread_id)

        _ctx()["slot"] = None

    # -- ① 算触发 ----------------------------------------------------------
    def _next_tasks(self, rt: _RuntimeState) -> list[Task]:
        """① 这一超步要跑哪些任务。"""
        return _prepare_next_tasks(
            rt.channels, rt.versions, rt.seen, self.nodes,
            trigger_to_nodes=self.trigger_to_nodes,
            updated_channels=rt.updated,
        )

    # -- ③ 分配槽位 --------------------------------------------------------
    def _assign_slots(self, tasks: list[Task], step: int,
                      snap: Checkpoint | None) -> list[tuple[Task, str]]:
        """
        给每个任务分配一个**槽位**（slot）。

        槽位是 interrupt / resume 的握手地址：节点内部调 interrupt() 时
        并不知道自己是谁，只能靠当前槽位去取 resume 值。

        恢复时，整批任务必须落回它们当初的槽位，否则槽位里的 step 号
        会漂移，之前答复过的 resume 值就配不上槽位了。
        """
        recorded = {
            (r["name"], r.get("idx", 0), bool(r.get("push", False))): r["slot"]
            for r in (snap.slots if snap and snap.slots else [])
        }
        runs: list[tuple[Task, str]] = []
        for task in tasks:
            kind = "push" if task.is_push else "pull"
            slot = recorded.get((task.name, task.idx, task.is_push))
            if slot is None:
                slot = f"{task.name}@{step + 1}#{kind}:{task.idx}"
            runs.append((task, slot))
        return runs

    # -- ④ 收写入 ----------------------------------------------------------
    def _collect_writes(self, runs, results, snapshot: dict):
        """
        ④ 把节点返回值翻译成「(节点, 通道, 值)」。

        同时挑出本步**所有**挂起的任务（有人调了 interrupt()）。
        多个节点并行挂起时，它们都要被记下来 —— 少记一个，
        那个节点重跑时会永远等不到自己的 resume 值。
        """
        writes: list[tuple[str, str, Any]] = []
        ran: list[str] = []
        hits: list[_Hit] = []
        for (task, slot), out in zip(runs, results):
            name = task.name
            if isinstance(out, GraphInterrupt):
                # 挂起 = 本超步作废：已提交的写入会被丢弃，恢复后整批重跑。
                # 这正是 BSP 屏障（"一步要么全提交，要么全不提交"）的含义，
                # 也是 checkpoint 存在**超步级**的根本原因 —— 所以节点要幂等。
                hits.append(_Hit(task=task, slot=slot, value=out.value))
                continue
            if isinstance(out, _NodeError):
                raise RuntimeError(f"节点 {name!r} 执行失败：{out.exc}") from out.exc
            ran.append(name)
            node_writes = _normalize(name, out, snapshot, self.nodes, self.channels,
                                     self.reducers)
            writes.extend((name, c, v) for c, v in node_writes)
        return writes, ran, hits

    # -- ⑤⑥ 落盘 -----------------------------------------------------------
    def _advance(self, rt: _RuntimeState, tasks: list[Task],
                 writes: list[tuple[str, str, Any]]) -> None:
        """
        ⑤⑥ 把这一超步的写入落到通道上，并推进版本号。

        四小步，顺序不能换：

          1. 推水位线  用**写入前**的版本（见 advance_watermarks）
          2. 消费屏障  NamedBarrierValue 重新武装（见 consume_triggers）
          3. 应用写入  本步所有写入共享同一个新版本号
          4. 清旧通道  没人写的触发通道自己抹掉（见 expire_unwritten）
        """
        rt.advance_watermarks(tasks)
        next_version = rt.next_version()
        rt.updated = set()               # 重新收集"这一步变了哪些通道"
        rt.consume_triggers(tasks, next_version)
        rt.updated |= _apply_writes(
            rt.channels, rt.versions, writes, next_version=next_version
        )
        rt.expire_unwritten({c for _, c, _ in writes}, next_version)

    # -- 挂起 --------------------------------------------------------------
    def _suspend(self, rt: _RuntimeState, runs: list[tuple[Task, str]],
                 hits: list[_Hit], store_before: dict[str, list[Any]],
                 saver, thread_id: str | None) -> None:
        """
        挂起：冻结在 interrupt 处，存一份 checkpoint，然后结束这次运行。

        冻结的是**超步边界**：本步的写入一律不应用、版本一律不推进、
        水位线一律不动。于是恢复时算触发会得出**同一批任务**，整批重跑。

        存三样和恢复有关的东西：
          interrupts  本步**所有**挂起点（多个节点同时挂起时要全记）
          resumes     已收集到的 resume 值（整步回滚，重跑时还要用）
          slots       本批任务的槽位分配（重跑时槽位不能漂移）
        """
        tasks = [t for t, _ in runs]
        this_step = rt.step + 1          # 挂起这一步的编号 = 它本该成为的超步号
        records = [
            {"slot": h.slot, "node": h.task.name, "idx": h.task.idx,
             "push": h.task.is_push, "step": this_step, "value": h.value}
            for h in hits
        ]
        ckpt = Checkpoint(
            id=saver.new_id() if saver else "ephemeral",
            step=this_step, values=rt.projection(),
            next=tuple(t.name for t in tasks),
            tasks=[t.send if t.is_push else t.name for t in tasks],
            interrupt=records[0] if records else None,
            interrupts=records,
            resumes={k: list(v) for k, v in store_before.items()},
            slots=[{"name": t.name, "idx": t.idx, "push": t.is_push, "slot": s}
                   for (t, s) in runs],
            channel_values={c: ch.checkpoint() for c, ch in rt.channels.items()},
            channel_versions=dict(rt.versions),
            versions_seen={k: dict(v) for k, v in rt.seen.items()},
        )
        if saver and thread_id:
            saver.put(thread_id, ckpt)
        self._last_values = rt.projection()
        self._interrupted = hits[0].value
        self._interrupts = [h.value for h in hits]

    # -- 存 checkpoint ------------------------------------------------------
    def _save(self, rt: _RuntimeState, saver, thread_id: str) -> None:
        """把本步的结束状态存成 checkpoint（给 get_state / 恢复用）。"""
        upcoming = self._next_tasks(rt)
        saver.put(thread_id, Checkpoint(
            id=saver.new_id(), step=rt.step,
            values=rt.projection(),
            next=tuple(t.name for t in upcoming),
            tasks=[t.send if t.is_push else t.name for t in upcoming],
            channel_values={c: ch.checkpoint() for c, ch in rt.channels.items()},
            channel_versions=dict(rt.versions),
            versions_seen={k: dict(v) for k, v in rt.seen.items()},
        ))


class _NodeError:
    def __init__(self, exc: BaseException):
        self.exc = exc


# ---------------------------------------------------------------------------
# 触发判定
# ---------------------------------------------------------------------------

def _merge_resume(store: dict[str, list[Any]], pending: list[dict[str, Any]],
                  value: Any) -> dict[str, list[Any]]:
    """
    把用户带来的 resume 值并进"槽位 -> 值队列"。

    `value` 有三种形态：

      - **标量**：给第一个待恢复的中断（对齐 LangGraph 的
        "a single value with which to resume the next interrupt"）。
      - **字典，键是槽位（"b@2#push:1"）或节点名（"b"）**：minigraph 的扩展，
        一次答复多个挂起点。节点名会命中该节点**所有**待恢复的槽位；
        值本身可以是列表 —— 对应"一个节点里连续多次 interrupt()"。
      - **字典，键对不上任何节点名/槽位**：整个字典就是**一个**答复值，
        原样交给 interrupt()。

    第三种是 LangGraph 那边最常见的用法：中断的请求本身是个 dict
    （`{"action": "approve"}`），`Command(resume=X)` 里 interrupt() 拿到的
    就是 X 本身（对齐真实 LangGraph：resume 值**原样透传**，不做节点匹配）。
    少了这一条，LangGraph 风格的答复会被当成"按节点名批量答复"而**静默丢掉**
    —— 节点于是原地再挂起一次，看起来像"答复没生效"。

    代价是有一个歧义：payload 的键**恰好**叫某个待恢复节点名时，
    按扩展解释（键 = 节点名）。这是有意取舍 —— 扩展是文档化的一等公民。

    队列而不是单值：整步回滚会让节点重跑，已经答复过的值必须留着；
    一个节点多次 interrupt() 时也要按调用顺序逐个取。
    """
    if isinstance(value, Mapping):
        # 只有键真的对得上某个待恢复的节点/槽位，才按"批量答复"解释。
        matched = [k for k in value
                   if any(p["slot"] == k or p["node"] == k for p in pending)]
        if matched:
            for key in matched:
                val = value[key]
                targets = [p for p in pending if p["slot"] == key] or \
                          [p for p in pending if p["node"] == key]
                for p in targets:
                    queue = store.setdefault(p["slot"], [])
                    if isinstance(val, (list, tuple)):
                        queue.extend(val)
                    else:
                        queue.append(val)
            return store
        # 键对不上 -> 整个字典是一个答复值，原样透传给 interrupt()
    for p in pending:
        store.setdefault(p["slot"], []).append(value)
        break
    return store


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
        # __start__ 是伪节点，它只负责在开局写入口信号，本身不是任务。
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
                          update: dict, *, node: str,
                          types: Mapping[str, Any] | None = None) -> None:
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
            ch = channel_for_field(key, reducers[key], (types or {}).get(key))
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
    （输入里的未声明键是另一回事，它们只读、不许写，见 _begin。）

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
