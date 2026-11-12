"""
差分场景定义。

每个场景描述一张图，build() 收到 Backend 就能把同样的图搭到任意一套后端上。
这样"两张图是不是同一张图"就由构造代码本身保证，而不是靠人工核对。

覆盖的场景都是**内核语义的分水岭**，不是边角料：

  - 线性流转 / 条件循环       基本骨架
  - 两条独立边汇聚             目标节点该跑几次
  - add_edge([a,b],c) 屏障    多源汇聚该跑几次
  - START 多入口               入口会不会被静默丢弃
  - Command(goto=)            动态路由
  - Send 扇出                 PUSH 轨
  - reducer 混用              合并语义
  - 并发写无 reducer 字段      冲突检测
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable

from .backends import Backend


@dataclass
class Scenario:
    name: str
    annotations: dict[str, Any]
    build: Callable[[Backend], Any]          # 收到后端，返回搭好的 builder
    input: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    # 这个场景在两边是否**预期**行为不同（记录已知的、有意为之的差异）
    known_divergence: str | None = None


# ---------------------------------------------------------------------------
# 1. 线性流转
# ---------------------------------------------------------------------------
def _linear(b: Backend):
    g = b.state_graph_cls(b.make_state({
        "n": Annotated[int, operator.add],
        "log": Annotated[list, operator.add],
    }))
    g.add_node("a", lambda s: {"n": 1, "log": ["a"]})
    g.add_node("b", lambda s: {"n": 1, "log": ["b"]})
    g.add_node("c", lambda s: {"n": 1, "log": ["c"]})
    g.add_edge(b.START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", b.END)
    return g


# ---------------------------------------------------------------------------
# 2. 条件循环（回边靠版本推进重新触发）
# ---------------------------------------------------------------------------
def _conditional_loop(b: Backend):
    g = b.state_graph_cls(b.make_state({
        "n": Annotated[int, operator.add],
        "log": Annotated[list, operator.add],
    }))
    g.add_node("inc", lambda s: {"n": 1, "log": ["inc"]})
    g.add_node("check", lambda s: {"log": ["check"]})
    g.add_edge(b.START, "inc")
    g.add_edge("inc", "check")
    g.add_conditional_edges(
        "check",
        lambda s: "again" if s["n"] < 3 else "stop",
        {"again": "inc", "stop": b.END},
    )
    return g


# ---------------------------------------------------------------------------
# 3. 两条独立边在同一超步写入 -> 目标节点只跑一次
# ---------------------------------------------------------------------------
def _two_edges_to_same_target(b: Backend):
    g = b.state_graph_cls(b.make_state({
        "log": Annotated[list, operator.add],
    }))
    g.add_node("fork", lambda s: {"log": ["fork"]})
    g.add_node("a", lambda s: {"log": ["a"]})
    g.add_node("b", lambda s: {"log": ["b"]})
    g.add_node("c", lambda s: {"log": ["c"]})
    g.add_edge(b.START, "fork")
    g.add_edge("fork", "a")
    g.add_edge("fork", "b")
    g.add_edge("a", "c")
    g.add_edge("b", "c")
    g.add_edge("c", b.END)
    return g


# ---------------------------------------------------------------------------
# 3b. 两条独立边在不同超步写入 -> 目标节点跑两次
# ---------------------------------------------------------------------------
def _asymmetric_fanin(b: Backend):
    g = b.state_graph_cls(b.make_state({
        "log": Annotated[list, operator.add],
    }))
    g.add_node("fork", lambda s: {"log": ["fork"]})
    g.add_node("a", lambda s: {"log": ["a"]})
    g.add_node("b1", lambda s: {"log": ["b1"]})
    g.add_node("b2", lambda s: {"log": ["b2"]})
    g.add_node("c", lambda s: {"log": ["c"]})
    g.add_edge(b.START, "fork")
    g.add_edge("fork", "a")
    g.add_edge("fork", "b1")
    g.add_edge("a", "c")
    g.add_edge("b1", "b2")
    g.add_edge("b2", "c")
    g.add_edge("c", b.END)
    return g


# ---------------------------------------------------------------------------
# 4. add_edge([a,b],c) 屏障 -> 目标节点跑一次
# ---------------------------------------------------------------------------
def _barrier_join(b: Backend):
    g = b.state_graph_cls(b.make_state({
        "log": Annotated[list, operator.add],
    }))
    g.add_node("fork", lambda s: {"log": ["fork"]})
    g.add_node("a", lambda s: {"log": ["a"]})
    g.add_node("b", lambda s: {"log": ["b"]})
    g.add_node("c", lambda s: {"log": ["c"]})
    g.add_edge(b.START, "fork")
    g.add_edge("fork", "a")
    g.add_edge("fork", "b")
    g.add_edge(["a", "b"], "c")     # <- 屏障
    g.add_edge("c", b.END)
    return g


# ---------------------------------------------------------------------------
# 5. START 多入口（曾经会静默丢弃第一个入口）
# ---------------------------------------------------------------------------
def _start_multi_edge(b: Backend):
    g = b.state_graph_cls(b.make_state({
        "log": Annotated[list, operator.add],
    }))
    g.add_node("a", lambda s: {"log": ["a"]})
    g.add_node("b", lambda s: {"log": ["b"]})
    g.add_edge(b.START, "a")
    g.add_edge(b.START, "b")
    g.add_edge("a", b.END)
    g.add_edge("b", b.END)
    return g


# ---------------------------------------------------------------------------
# 6. Command(goto=) 动态路由
# ---------------------------------------------------------------------------
def _command_goto(b: Backend):
    g = b.state_graph_cls(b.make_state({
        "score": int,
        "log": Annotated[list, operator.add],
    }))
    g.add_node(
        "router",
        lambda s: b.Command(
            goto="right" if s["score"] > 5 else "left",
            update={"log": ["router"]},
        ),
    )
    g.add_node("left", lambda s: {"log": ["left"]})
    g.add_node("right", lambda s: {"log": ["right"]})
    g.add_edge(b.START, "router")
    g.add_edge("left", b.END)
    g.add_edge("right", b.END)
    return g


# ---------------------------------------------------------------------------
# 6b. Command(goto=) 与静态边同时生效
# ---------------------------------------------------------------------------
def _command_goto_with_static_edge(b: Backend):
    g = b.state_graph_cls(b.make_state({
        "log": Annotated[list, operator.add],
    }))
    g.add_node("router", lambda s: b.Command(goto="dynamic", update={"log": ["router"]}))
    g.add_node("static", lambda s: {"log": ["static"]})
    g.add_node("dynamic", lambda s: {"log": ["dynamic"]})
    g.add_edge(b.START, "router")
    g.add_edge("router", "static")
    g.add_edge("static", b.END)
    g.add_edge("dynamic", b.END)
    return g


# ---------------------------------------------------------------------------
# 7. Send 扇出（条件边发出）
# ---------------------------------------------------------------------------
def _send_fanout(b: Backend):
    g = b.state_graph_cls(b.make_state({
        "topics": list,
        "results": Annotated[list, operator.add],
    }))
    g.add_node("plan", lambda s: {"topics": ["a", "b", "c"]})
    g.add_node("worker", lambda s: {"results": [s["t"]]})

    def dispatch(s):
        return [b.Send("worker", {"t": t}) for t in s["topics"]]

    g.add_edge(b.START, "plan")
    g.add_conditional_edges("plan", dispatch, ["worker"])
    # worker 是条件边的终端目标，不需要再画一条 worker -> END。
    return g


# ---------------------------------------------------------------------------
# 8. reducer 混用
# ---------------------------------------------------------------------------
def _reducers_mix(b: Backend):
    g = b.state_graph_cls(b.make_state({
        "total": Annotated[int, operator.add],
        "log": Annotated[list, operator.add],
        "name": str,
    }))
    g.add_node("a", lambda s: {"total": 1, "log": ["a"], "name": "n1"})
    g.add_node("b", lambda s: {"total": 2, "log": ["b"], "name": "n2"})
    g.add_edge(b.START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", b.END)
    return g


# ---------------------------------------------------------------------------
# 9. 并行分支各写各的字段（不该冲突）
# ---------------------------------------------------------------------------
def _parallel_distinct_fields(b: Backend):
    g = b.state_graph_cls(b.make_state({
        "a": Annotated[list, operator.add],
        "b": Annotated[list, operator.add],
    }))
    g.add_node("fork", lambda s: {})
    g.add_node("x", lambda s: {"a": ["x"]})
    g.add_node("y", lambda s: {"b": ["y"]})
    g.add_edge(b.START, "fork")
    g.add_edge("fork", "x")
    g.add_edge("fork", "y")
    g.add_edge("x", b.END)
    g.add_edge("y", b.END)
    return g


# ---------------------------------------------------------------------------
# 10. 并发写同一个无 reducer 字段 -> 两边都该报错
# ---------------------------------------------------------------------------
def _concurrent_lww_conflict(b: Backend):
    g = b.state_graph_cls(b.make_state({"v": str}))
    g.add_node("fork", lambda s: {})
    g.add_node("x", lambda s: {"v": "x"})
    g.add_node("y", lambda s: {"v": "y"})
    g.add_edge(b.START, "fork")
    g.add_edge("fork", "x")
    g.add_edge("fork", "y")
    g.add_edge("x", b.END)
    g.add_edge("y", b.END)
    return g


# ---------------------------------------------------------------------------
# 11. 自环（recursion_limit 应该能兜住）
# ---------------------------------------------------------------------------
def _self_loop(b: Backend):
    g = b.state_graph_cls(b.make_state({"n": Annotated[int, operator.add]}))
    g.add_node("n", lambda s: {"n": 1})
    g.add_edge(b.START, "n")
    g.add_edge("n", "n")
    return g


# ---------------------------------------------------------------------------
# 12. 扇出后汇聚（Send + 静态边）
# ---------------------------------------------------------------------------
def _fanout_then_join(b: Backend):
    g = b.state_graph_cls(b.make_state({
        "topics": list,
        "results": Annotated[list, operator.add],
    }))
    g.add_node("plan", lambda s: {"topics": ["a", "b"]})
    g.add_node("worker", lambda s: {"results": [s["t"]]})
    g.add_node("agg", lambda s: {"results": ["agg"]})

    def dispatch(s):
        return [b.Send("worker", {"t": t}) for t in s["topics"]]

    g.add_edge(b.START, "plan")
    g.add_conditional_edges("plan", dispatch, ["worker"])
    g.add_edge("worker", "agg")
    g.add_edge("agg", b.END)
    return g


SCENARIOS: list[Scenario] = [
    Scenario("linear", {"n": Annotated[int, operator.add],
                        "log": Annotated[list, operator.add]},
             _linear, {"n": 0, "log": []}),
    Scenario("conditional_loop", {"n": Annotated[int, operator.add],
                                  "log": Annotated[list, operator.add]},
             _conditional_loop, {"n": 0, "log": []}),
    Scenario("two_edges_same_superstep", {"log": Annotated[list, operator.add]},
             _two_edges_to_same_target, {"log": []}),
    Scenario("asymmetric_fanin", {"log": Annotated[list, operator.add]},
             _asymmetric_fanin, {"log": []}),
    Scenario("barrier_join", {"log": Annotated[list, operator.add]},
             _barrier_join, {"log": []}),
    Scenario("start_multi_edge", {"log": Annotated[list, operator.add]},
             _start_multi_edge, {"log": []}),
    Scenario("command_goto", {"score": int, "log": Annotated[list, operator.add]},
             _command_goto, {"score": 9, "log": []}),
    Scenario("command_goto_with_static_edge", {"log": Annotated[list, operator.add]},
             _command_goto_with_static_edge, {"log": []}),
    Scenario("send_fanout", {"topics": list, "results": Annotated[list, operator.add]},
             _send_fanout, {"topics": [], "results": []}),
    Scenario("reducers_mix", {"total": Annotated[int, operator.add],
                               "log": Annotated[list, operator.add], "name": str},
             _reducers_mix, {"total": 0, "log": [], "name": ""}),
    Scenario("parallel_distinct_fields", {"a": Annotated[list, operator.add],
                                          "b": Annotated[list, operator.add]},
             _parallel_distinct_fields, {"a": [], "b": []}),
    Scenario("concurrent_lww_conflict", {"v": str},
             _concurrent_lww_conflict, {"v": ""}),
    Scenario("self_loop", {"n": Annotated[int, operator.add]},
             _self_loop, {"n": 0}, config={"recursion_limit": 5}),
    Scenario("fanout_then_join", {"topics": list, "results": Annotated[list, operator.add]},
             _fanout_then_join, {"topics": [], "results": []}),
]
