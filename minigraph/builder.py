"""
§4 图构建：把"图"编译成"通道网络"

这一节是整个重写的**编译期**。核心思想：

    **图不是被"调度"的，图是被编译成通道网络的。**

编译产物有三样：

  1. **channels**  通道表。状态字段各占一个数据通道；每个节点各占一个
                   触发通道 `branch:to:X`；多源汇聚占一个屏障通道 `join:a+b:c`。
  2. **triggers**  每个节点订阅了哪些通道（它能被哪些通道唤醒）。
  3. **writers**  每个节点会写哪些通道（它的边去哪儿）。

运行时（runtime.py）只跟这三样打交道，完全不再看"边列表"。

对照真实 LangGraph：langgraph/graph/state.py 的
attach_node / attach_edge / attach_branch，做的是同一件事。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from .channels import (
    BaseChannel,
    EphemeralValue,
    NamedBarrierValue,
    Topic,
    channel_for_field,
)
from .constants import BRANCH_TO, END, JOIN, Reducer, START, TASKS
from .state import State, _schema

if TYPE_CHECKING:  # 运行时真正 import 留给 .compile() 内部
    from .checkpoint import InMemorySaver
    from .runtime import CompiledGraph


@dataclass
class NodeSpec:
    """一个节点编译后的规格。"""

    name: str
    fn: Callable
    retry: Any = None

    # 订阅了哪些通道 —— 任一通道有新版本就唤醒本节点
    triggers: list[str] = field(default_factory=list)

    # 会写哪些通道（静态边编译来的）
    writers: list[str] = field(default_factory=list)

    # 条件边：运行时才决定写哪些通道
    branches: list[dict[str, Any]] = field(default_factory=list)

    # 该节点是否是 __start__ 伪节点
    is_start: bool = False


class StateGraph:
    def __init__(self, state_schema: Mapping[str, Reducer] | type[State] | None = None):
        self.reducers = _schema(state_schema)
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[tuple[str, str]] = []
        # 多源汇聚边：add_edge(["a","b"], "c")
        self.waiting_edges: list[tuple[tuple[str, ...], str]] = []
        self.branches: list[dict[str, Any]] = []
        # 入口点可以有多个：add_edge(START,"a") 和 add_edge(START,"b") 都要保留
        self.entry_points: list[str] = []

    # -- 兼容旧属性：entry 现在是 entry_points 的第一个 -------------------
    @property
    def entry(self) -> str | None:
        return self.entry_points[0] if self.entry_points else None

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

    def add_edge(self, src: str | Sequence[str], dst: str) -> "StateGraph":
        """
        加一条边。

        - `add_edge("a", "b")`  -> 编译成 a 写 "branch:to:b"，b 订阅它
        - `add_edge(["a","b"], "c")` -> 编译成屏障通道 "join:a+b:c"，
          a 和 b 各写自己的名字，c 等齐后只被触发**一次**
        - `add_edge(START, "a")` -> a 是入口点。**可以有多个入口**，
          它们会并行跑（之前这里后者会覆盖前者，导致前面的入口被静默丢弃）
        """
        if isinstance(src, str):
            if src != START and src not in self.nodes:
                raise ValueError(f"未知源节点 {src!r}")
            if dst != END and dst not in self.nodes:
                raise ValueError(f"未知目标节点 {dst!r}")
            if src == START:
                # 累加而不是覆盖：多个入口点要一起跑
                if dst not in self.entry_points:
                    self.entry_points.append(dst)
            self.edges.append((src, dst))
            return self

        # 多源 -> 屏障边
        for s in src:
            if s != START and s not in self.nodes:
                raise ValueError(f"未知源节点 {s!r}")
        if dst != END and dst not in self.nodes:
            raise ValueError(f"未知目标节点 {dst!r}")
        self.waiting_edges.append((tuple(src), dst))
        return self

    def add_conditional_edges(
        self,
        src: str,
        path: Callable[[State], Any],
        path_map: Mapping[str, str] | Sequence[str] | None = None,
    ) -> "StateGraph":
        """
        path 返回 key（或 key 列表 / Send），经 path_map 翻译成节点名。

        与 LangGraph 一致，path_map 既可以给 {"continue": "worker"} 这样的映射，
        也可以给 ["worker"] 这样的列表（每个名字映射到自身）。
        """
        if src not in self.nodes:
            raise ValueError(f"未知源节点 {src!r}")
        if isinstance(path_map, Mapping) or path_map is None:
            mapping = dict(path_map or {})
        elif isinstance(path_map, (str, bytes)):
            raise TypeError("path_map 必须是映射或节点名列表，不能是字符串")
        else:
            mapping = {name: name for name in path_map}
        self.branches.append({"source": src, "path": path, "map": mapping})
        return self

    def set_entry_point(self, name: str) -> "StateGraph":
        return self.add_edge(START, name)

    def set_finish_point(self, name: str) -> "StateGraph":
        return self.add_edge(name, END)

    # -- 编译 -------------------------------------------------------------
    def _compile(self) -> tuple[dict[str, BaseChannel], dict[str, NodeSpec]]:
        """
        把图编译成 (通道表, 节点规格表)。

        这是「边 = 一次通道写入」真正落地的地方。
        """
        channels: dict[str, BaseChannel] = {}
        specs: dict[str, NodeSpec] = {}

        # ① 数据通道：每个状态字段一个
        for field, reducer in self.reducers.items():
            channels[field] = channel_for_field(field, reducer)

        # ② 触发通道：每个节点一个 branch:to:X，节点订阅它
        for name in self.nodes:
            chan = BRANCH_TO.format(name)
            # guard=False：两个并行节点同时 Command(goto="x") 是合法的
            channels[chan] = EphemeralValue(chan, guard=False)
            specs[name] = NodeSpec(
                name=name,
                fn=self.nodes[name]["fn"],
                retry=self.nodes[name].get("retry"),
                triggers=[chan],
            )

        # ③ 入口伪节点：__start__ 写所有入口点的触发通道
        start_chan = START
        channels[start_chan] = EphemeralValue(start_chan)
        specs[START] = NodeSpec(
            name=START,
            fn=None,
            triggers=[start_chan],
            writers=[BRANCH_TO.format(x) for x in self.entry_points],
            is_start=True,
        )

        # ④ 静态边：src 写 branch:to:dst
        for src, dst in self.edges:
            if src == START:
                continue                      # 入口已在 ③ 处理
            if dst == END:
                continue                      # 结束边不需要通道
            specs[src].writers.append(BRANCH_TO.format(dst))

        # ⑤ 多源屏障：add_edge(["a","b"], "c")
        for starts, end in self.waiting_edges:
            if end == END:
                continue
            chan = JOIN.format("+".join(starts) + ":" + end)
            channels[chan] = NamedBarrierValue(chan, names=set(starts))
            specs[end].triggers.append(chan)
            for s in starts:
                # 写入的值是"源自己的名字"，作为"我到了"的信号
                specs[s].writers.append(chan)

        # ⑥ 条件边挂到源节点上
        for br in self.branches:
            specs[br["source"]].branches.append(br)

        # ⑦ Send 队列通道
        channels[TASKS] = Topic(TASKS, accumulate=False)

        return channels, specs

    def compile(self, checkpointer: "InMemorySaver | None" = None) -> "CompiledGraph":
        # 局部 import 切断与 runtime / checkpoint 的顶层 import 环
        from .checkpoint import InMemorySaver
        from .runtime import CompiledGraph

        return CompiledGraph(self, checkpointer)
