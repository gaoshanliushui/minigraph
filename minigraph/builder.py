"""
§4 图构建
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Mapping

from .constants import END, Reducer, START
from .state import State, _schema

if TYPE_CHECKING:  # 运行时真正 import 留给 .compile() 内部
    from .checkpoint import InMemorySaver
    from .runtime import CompiledGraph


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

    def compile(self, checkpointer: "InMemorySaver | None" = None) -> "CompiledGraph":
        # 局部 import 切断与 runtime / checkpoint 的顶层 import 环
        from .checkpoint import InMemorySaver
        from .runtime import CompiledGraph

        return CompiledGraph(self, checkpointer)