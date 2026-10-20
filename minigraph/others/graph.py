"""
minigraph —— 用约 600 行复刻 LangGraph 的核心机制。

它不追求 API 全覆盖，只保留"少了它 LangGraph 就不成立"的部分：
状态+reducer、超步调度、条件边/Command 路由、Send 扇出、checkpoint 与 interrupt。

对照关系（真 LangGraph -> mini 版）：
    StateGraph / CompiledGraph    -> StateGraph / CompiledGraph
    Annotation.Root               -> State(dict of reducers)
    add_node / add_edge           -> 同名
    add_conditional_edges         -> 同名
    START / END                   -> 同名
    Command(goto=..., update=..)  -> 同名
    Send(node, arg)               -> 同名
    interrupt() / Command(resume) -> 同名
    InMemorySaver                 -> InMemorySaver
    Pregel 超步循环                -> CompiledGraph._run
"""

from __future__ import annotations

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

# ----------------------------------------------------------------------------
# 0. 哨兵与类型别名
# ----------------------------------------------------------------------------

START = "__start__"
END = "__end__"

Reducer = Callable[[Any, Any], Any]  # (旧值, 新值) -> 合并后的值


class _Missing:
    def __repr__(self) -> str:  # pragma: no cover
        return "<missing>"


MISSING = _Missing()

_CTX = threading.local()


def _ctx() -> dict[str, Any]:
    """每次运行（一次 invoke / resume）一份上下文，供 interrupt() 读取。"""
    if not hasattr(_CTX, "data"):
        _CTX.data = {"resume": {}, "slot": None}
    return _CTX.data


# ----------------------------------------------------------------------------
# 1. 状态：字段 -> reducer
# ----------------------------------------------------------------------------
# 这是 LangGraph 最本质的设计：
#   节点是"无状态函数"，状态只存在于图这一层；
#   节点返回的不是新状态，而是一组"增量（update）"，
#   由 reducer 决定如何合并回主状态。
# 默认 reducer 是 last-write-wins；因此同一超步里两个并行节点
# 写同一个字段会直接报错（防止静默覆盖），这是很多人第一次
# 用并行分支时踩的坑。
# ----------------------------------------------------------------------------

LAST_VALUE: Reducer = lambda old, new: new


def add_messages(old, new):
    """消息列表 reducer：按 id 追加/更新，语义对齐 langgraph 的 add_messages。"""
    old = [] if old is MISSING else old
    new = new if isinstance(new, (list, tuple)) else [new]
    out = list(old)
    for msg in new:
        mid = msg.get("id") if isinstance(msg, Mapping) else getattr(msg, "id", None)
        idx = next((i for i, m in enumerate(out)
                    if (m.get("id") if isinstance(m, Mapping) else getattr(m, "id", None)) == mid), None
                   ) if mid else None
        if idx is None:
            out.append(msg)
        else:
            out[idx] = msg
    return out


def merge_dict(old, new):
    if old is MISSING:
        return dict(new or {})
    merged = dict(old)
    merged.update(new or {})
    return merged


class State(dict):
    """
    声明式状态。用法：

        class S(State):
            messages: Annotated[list, add_messages]   # 自定义合并
            total:    Annotated[int, operator.add]    # 累加（并行安全）
            query:    str                             # 缺省 last-write-wins
    """

    reducers: dict[str, Reducer] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        reducers: dict[str, Reducer] = {}
        for base in reversed(cls.__mro__[1:]):
            reducers.update(getattr(base, "reducers", {}))
        for name, ann in cls.__dict__.get("__annotations__", {}).items():
            if name.startswith("_") or name == "reducers":
                continue
            reducers[name] = _extract_reducer(ann) or LAST_VALUE
        cls.reducers = reducers

    @classmethod
    def schema(cls) -> dict[str, Reducer]:
        return dict(cls.reducers)


def _extract_reducer(ann) -> Reducer | None:
    meta = getattr(ann, "__metadata__", None)
    if not meta:
        return None
    first = meta[0]
    if callable(first):
        return first
    if isinstance(first, tuple) and first and callable(first[0]):
        return first[0]
    return None


def _schema(obj: Mapping[str, Reducer] | type[State] | None) -> dict[str, Reducer]:
    if obj is None:
        return {}
    if isinstance(obj, type) and issubclass(obj, State):
        return obj.schema()
    return dict(obj)


# ----------------------------------------------------------------------------
# 2. 控制流原语
# ----------------------------------------------------------------------------


@dataclass
class Command:
    """节点直接决定下一跳，可顺带提交 update（LangGraph 的 Command）。"""

    goto: str | Sequence[str] | None = None
    update: dict[str, Any] | None = None
    resume: Any = None


@dataclass
class Send:
    """动态扇出：把一份独立输入派给某节点，构造并行任务。"""

    node: str
    arg: Any


class GraphInterrupt(Exception):
    def __init__(self, value: Any):
        self.value = value
        super().__init__(f"interrupt: {value!r}")


def interrupt(value: Any) -> Any:
    """
    节点内调用即暂停整张图，value 作为"请求"暴露给调用方。
    恢复时本函数返回 Command(resume=...) 传入的值；
    注意节点会从头重跑，所以 interrupt 之前的副作用必须幂等。
    """
    slot = _ctx().get("slot")
    if slot is None:
        raise RuntimeError("interrupt() 只能在图节点内调用")
    store = _ctx()["resume"]
    if slot in store:
        return store.pop(slot)
    raise GraphInterrupt(value)


# ----------------------------------------------------------------------------
# 3. 检查点
# ----------------------------------------------------------------------------


@dataclass
class Checkpoint:
    id: str
    step: int
    values: dict[str, Any]
    next: tuple[str, ...]
    # 待执行任务，保留 Send 以便恢复后能重建并行分支
    tasks: list[Any] = field(default_factory=list)
    interrupt: dict[str, Any] | None = None

    def copy(self) -> "Checkpoint":
        import copy as _c

        return Checkpoint(
            id=self.id, step=self.step, values=_c.deepcopy(self.values),
            next=tuple(self.next), tasks=_c.deepcopy(self.tasks),
            interrupt=_c.deepcopy(self.interrupt),
        )


class InMemorySaver:
    """按 thread_id 追加保存 checkpoint；就是 LangGraph 的 InMemorySaver。"""

    def __init__(self) -> None:
        self.storage: dict[str, list[Checkpoint]] = {}
        self._n = 0

    def new_id(self) -> str:
        self._n += 1
        return f"ckpt-{self._n:04d}-{random.randint(0, 0xFFFF):04x}"

    def put(self, thread_id: str, ckpt: Checkpoint) -> str:
        self.storage.setdefault(thread_id, []).append(ckpt)
        return ckpt.id

    def list(self, thread_id: str) -> list[Checkpoint]:
        return self.storage.get(thread_id, [])

    def get(self, thread_id: str, checkpoint_id: str | None = None) -> Checkpoint | None:
        history = self.list(thread_id)
        if not history:
            return None
        if checkpoint_id is None:
            return history[-1]
        for ckpt in history:  # 支持回到任意历史点重新分支
            if ckpt.id == checkpoint_id:
                return ckpt
        raise KeyError(f"thread {thread_id} 下找不到 checkpoint {checkpoint_id}")


# ----------------------------------------------------------------------------
# 4. 图构建
# ----------------------------------------------------------------------------


class StateGraph:
    def __init__(self, state_schema: Mapping[str, Reducer] | type[State] | None = None):
        self.reducers = _schema(state_schema)
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[tuple[str, str]] = []
        self.branches: list[dict[str, Any]] = []
        self.entry: str | None = None

    def add_node(self, name: str, action, **options) -> "StateGraph":
        if name in (START, END):
            raise ValueError(f"{name} 是保留名")
        if name in self.nodes:
            raise ValueError(f"节点 {name!r} 已存在")
        if isinstance(action, Mapping):
            fn, opts = action["fn"], dict(action)
        else:
            fn, opts = action, {}
        opts.update(options)
        self.nodes[name] = {"fn": fn, **opts}
        return self

    def add_edge(self, src: str, dst: str) -> "StateGraph":
        if src != START and src not in self.nodes:
            raise ValueError(f"未知源节点 {src!r}")
        if dst != END and dst not in self.nodes:
            raise ValueError(f"未知目标节点 {dst!r}")
        if src == START:
            self.entry = dst
        self.edges.append((src, dst))
        return self

    def add_conditional_edges(
        self, src: str, path: Callable[[State], Any], path_map: Mapping[str, str] | None = None
    ) -> "StateGraph":
        """path 返回 key（或 key 列表 / Send），经 path_map 翻译成节点名。"""
        if src not in self.nodes:
            raise ValueError(f"未知源节点 {src!r}")
        self.branches.append({"source": src, "path": path, "map": dict(path_map or {})})
        return self

    def set_entry_point(self, name: str) -> "StateGraph":
        return self.add_edge(START, name)

    def set_finish_point(self, name: str) -> "StateGraph":
        return self.add_edge(name, END)

    def compile(self, checkpointer: InMemorySaver | None = None) -> "CompiledGraph":
        return CompiledGraph(self, checkpointer)


# ----------------------------------------------------------------------------
# 5. Pregel 超步循环
# ----------------------------------------------------------------------------


class CompiledGraph:
    def __init__(self, graph: StateGraph, checkpointer: InMemorySaver | None = None):
        self.graph = graph
        self.reducers = graph.reducers
        self.checkpointer = checkpointer
        self._validate()

    def _validate(self) -> None:
        used = set()
        for src, dst in self.graph.edges:
            used.update((src, dst))
        used.update(b["source"] for b in self.graph.branches)
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
        as_node 指定"假装是哪个节点写的"，决定接下来从哪继续；
        不指定则沿用当前待执行任务。
        """
        thread_id = config["thread_id"]
        if not self.checkpointer:
            raise ValueError("update_state 需要 checkpointer")
        snap = self.get_state(config).copy()
        _apply(self.reducers, snap.values, values, node=as_node or "<update_state>")
        snap.id = self.checkpointer.new_id()
        snap.step += 1
        snap.interrupt = None
        if as_node:
            snap.tasks = [as_node]
            snap.next = (as_node,)
        self.checkpointer.put(thread_id, snap)
        return snap.id

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

        if input is None and snap is not None:
            # ---------- 从 checkpoint 恢复 ----------
            values = snap.values.copy()
            tasks: list[Any] = list(snap.tasks)
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
                    print(f"[resume] step={step} tasks={_names(tasks)}")
        else:
            # ---------- 全新运行 ----------
            # 输入即初始状态。注意：不像节点那样强制校验——输入里带些
            # 图内只读的键（如 user_id）是允许的，只有"写"才要求声明。
            values = dict(input or {})
            tasks = [graph.entry] if graph.entry else []
            step = 0
            _ctx()["resume"] = {}
            if debug:
                print(f"[start] {values}")

        # 恢复时：被中断的任务必须落回它当初的槽位，才能取到 resume 值
        pending_node = pending_slot = None
        pending_idx = 0
        pending_step = None
        if snap is not None and snap.interrupt:
            pending_node = snap.interrupt["node"]
            pending_slot = snap.interrupt["slot"]
            pending_idx = snap.interrupt.get("idx", 0)
            pending_step = snap.interrupt["step"]

        # ---------- 超步循环（Pregel / BSP）----------
        # 每个超步：① 并行执行本批任务 ② 收齐写入 ③ 用 reducer 一次性合并
        # ④ 根据合并后的状态决定下一批任务。这就是 LangGraph 的执行内核。
        while tasks:
            if step >= limit:
                raise RuntimeError(f"超过 recursion_limit={limit}，疑似死循环")
            # 恢复时重新落回挂起所在的那一步，槽位才能对上
            step = pending_step if pending_step is not None else step + 1
            pending_step = None

            batch = list(tasks)
            snapshot = dict(values)          # 本超步所有节点看到的都是同一快照
            runs: list[tuple[Any, str, str]] = []
            for idx, task in enumerate(batch):
                name = task.node if isinstance(task, Send) else task
                # 槽位必须唯一：同名节点在同一超步可能出现多次（循环回到自己），
                # 所以用 步号 + 批内序号；恢复时沿用挂起检查点里记下的那个槽位。
                slot = pending_slot if (pending_node == name and idx == pending_idx) \
                    else f"{name}@{step}#{idx}"
                runs.append((task, name, slot))

            results = _execute_batch(graph, runs, snapshot)

            writes: list[tuple[str, str, Any]] = []
            nxt: list[Any] = []
            ran: list[str] = []
            hit = None
            hit_node = None
            hit_slot = None
            hit_idx = 0

            for idx, (task, name, slot) in enumerate(runs):
                out = results[idx]
                if isinstance(out, GraphInterrupt):
                    # 挂起 = 本超步作废：已提交的写入会被丢弃，恢复后整批重跑。
                    # 这正是 BSP 屏障（"一步要么全提交，要么全不提交"）的含义，
                    # 也是 checkpoint 存在**超步级**的根本原因 —— 所以节点要幂等。
                    hit, hit_node, hit_slot, hit_idx = out, name, slot, idx
                    continue
                if isinstance(out, _NodeError):
                    raise RuntimeError(f"节点 {name!r} 执行失败：{out.exc}") from out.exc
                ran.append(name)
                update, routes = _normalize(name, out, snapshot, graph)
                writes.extend((name, k, v) for k, v in update.items() if not _is_noop(out, k))
                nxt.extend(routes)

            # 本超步全部写入在此一次性合并 —— 并行分支就是在这里汇流的
            if writes and hit is None:
                values = _merge_writes(reducers, values, writes)

            if hit is not None:                   # ---------- 挂起 ----------
                ckpt = Checkpoint(
                    id=saver.new_id() if saver else "ephemeral",
                    step=step, values=values,
                    next=tuple(_names(batch)), tasks=list(batch),
                    interrupt={"slot": hit_slot, "node": hit_node, "idx": hit_idx,
                               "step": step, "value": hit.value},
                )
                if saver and thread_id:
                    saver.put(thread_id, ckpt)
                self._last_values = values
                self._interrupted = hit.value
                yield hit_node, step, {}
                return

            # 汇合语义：同一超步多个分支指向同一节点只调度一次（fan-in）
            nxt = _dedupe_tasks(nxt)

            if debug:
                print(f"  step {step}: {ran} -> {_names(nxt)}")

            self._last_values = values
            yield ",".join(ran), step, {
                k: values.get(k) for k in dict.fromkeys(k for _, k, _ in writes)
            }

            if saver and thread_id:
                saver.put(thread_id, Checkpoint(
                    id=saver.new_id(), step=step, values=values.copy(),
                    next=tuple(_names(nxt)), tasks=list(nxt),
                ))
            tasks = nxt
            _ctx()["resume"] = {}

        _ctx()["slot"] = None


class _NodeError:
    def __init__(self, exc: BaseException):
        self.exc = exc


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


def _execute_batch(graph: StateGraph, runs, snapshot: dict):
    """
    同一超步内的节点并发执行。
    注意：并发只发生在"单步内"，跨超步仍然是严格的 BSP 同步屏障，
    因此结果与执行顺序无关（reducer 决定合并语义）。
    """
    def arg_for(task):
        return snapshot if not isinstance(task, Send) else _wrap_send(task, snapshot)

    # 恢复值必须显式带进工作线程：threading.local 不会被线程池继承
    resume = dict(_ctx().get("resume") or {})

    if len(runs) == 1:
        task, name, slot = runs[0]
        return [_run_task(name, graph.nodes[name]["fn"], arg_for(task),
                          graph.nodes[name].get("retry"), slot, resume)]

    with ThreadPoolExecutor(max_workers=min(16, len(runs))) as pool:
        futures = [
            pool.submit(_run_task, name, graph.nodes[name]["fn"], arg_for(task),
                        graph.nodes[name].get("retry"), slot, resume)
            for task, name, slot in runs
        ]
        return [f.result() for f in futures]


def _dedupe_tasks(tasks: list[Any]) -> list[Any]:
    """保留首个同名节点任务，Send 不去重（它们携带不同 arg）。"""
    seen: set[str] = set()
    out: list[Any] = []
    for t in tasks:
        if isinstance(t, Send):
            out.append(t)
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _names(tasks) -> list[str]:
    return [t.node if isinstance(t, Send) else t for t in tasks]


def _is_noop(out: Any, key: str) -> bool:
    """
    节点返回 {key: None} 在很多图里表示"这次不改"，不应参与并发冲突判定。
    真 LangGraph 会把 None 写入通道；这里为了并行友好，视作空更新。
    """
    return isinstance(out, Mapping) and out.get(key) is None


def _wrap_send(send: Send, values: State) -> Any:
    """Send 的 arg 作为节点入参，同时把主状态挂到 __state__ 供只读访问。"""
    if isinstance(send.arg, Mapping):
        return {**values, **send.arg, "__state__": values}
    return _SendView(send.arg, values)


class _SendView(dict):
    def __init__(self, arg, state):
        super().__init__(state)
        self.arg = arg


def _normalize(node: str, out: Any, values: State, graph: StateGraph) -> tuple[dict, list[Any]]:
    """拆出 (update, 下一跳列表)，并解析 Command。"""
    update: dict = {}
    if isinstance(out, Command):
        update = dict(out.update or {})
        goto = out.goto
        routes: list[Any] = []
        if goto is not None:
            routes = list(goto) if isinstance(goto, (list, tuple)) else [goto]
        routes = [r for r in routes if r != END]
        return update, routes

    # 直接返回 list[Send]：等价于"条件边里做扇出"，LangGraph 里常见写法
    if isinstance(out, (list, tuple)) and all(isinstance(x, Send) for x in out):
        return {}, list(out)

    if out is not None:
        if not isinstance(out, Mapping):
            raise TypeError(f"节点 {node!r} 必须返回 dict / None / Command，收到 {type(out).__name__}")
        update = dict(out)

    # 静态边
    statics = [dst for src, dst in graph.edges if src == node]
    cond = next((b for b in graph.branches if b["source"] == node), None)
    if cond is None:
        return update, [d for d in statics if d != END]

    # 条件边：path 看到的是"合并之后"的状态更合理，所以这里用 update 覆盖
    view = {**values, **update}
    decision = cond["path"](view)
    picks = decision if isinstance(decision, (list, tuple, set)) else [decision]
    routes = []
    for p in picks:
        if isinstance(p, Send):
            routes.append(p)
            continue
        if p == END:
            continue
        routes.append(cond["map"].get(p, p) if cond["map"] else p)
    return update, [r for r in routes if r != END]


def _apply(reducers: Mapping[str, Reducer], values: dict, update: dict, *, node: str) -> None:
    for key, new_val in update.items():
        reducer = reducers.get(key)
        if reducer is None:
            raise ValueError(
                f"节点 {node!r} 写了未声明的字段 {key!r}；已声明字段：{sorted(reducers)}"
            )
        values[key] = reducer(values.get(key, MISSING), new_val)


def _merge_writes(reducers: Mapping[str, Reducer], base: dict,
                  writes: list[tuple[str, str, Any]]) -> dict:
    """
    同一超步内多个节点可能写同一 key：
      - 有显式 reducer -> 逐个 reduce（并行安全）
      - 只有默认 last-write-wins 且并发写 -> 报错（LangGraph 的行为，
        因为"最后一次写入"在并行下是不确定的）
    """
    by_key: dict[str, list[tuple[str, Any]]] = {}
    for node, key, val in writes:
        by_key.setdefault(key, []).append((node, val))

    unknown = set(by_key) - set(reducers)
    if reducers and unknown:
        who = {k: [n for n, _ in by_key[k]] for k in unknown}
        raise ValueError(f"节点写了未声明的状态字段 {sorted(unknown)}（写入者 {who}）；已声明 {sorted(reducers)}")

    out = dict(base)
    for key, items in by_key.items():
        reducer = reducers.get(key, LAST_VALUE)
        if len(items) > 1 and reducer is LAST_VALUE:
            who = [n for n, _ in items]
            raise RuntimeError(
                f"同一超步有多个节点（{who}）写字段 {key!r}；"
                f"并发写必须给它配 reducer（如 operator.add / merge_dict）"
            )
        acc = out.get(key, MISSING)
        for _, v in items:
            acc = reducer(acc, v)
        out[key] = acc
    return out


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
