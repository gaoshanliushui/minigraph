"""
检查点层面的差分场景：update_state / interrupt / 时间旅行。

超步循环本身的差分在 tests/test_diff_langgraph.py；这里比的是**落盘之后**
的行为 —— 人工改写状态、挂起恢复、回到历史快照重跑。

这些场景要两套后端各自带 checkpointer 跑，所以沿用 scenarios.py 的写法：
每个场景写一个 build(backend)，同一份构造代码搭出两张图，保证可比。

比对的是三个维度：
  - next      改写之后接下来该跑哪些节点（**这是 as_node 语义的分水岭**）
  - values    改写之后的状态
  - final     继续跑完之后的最终状态
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable

from .backends import Backend

SCHEMA = {
    "n": Annotated[int, operator.add],
    "log": Annotated[list, operator.add],
}


@dataclass
class CheckpointScenario:
    name: str
    build: Callable[[Backend], Any]       # 收到后端，返回编好的 app（带 checkpointer）
    run: Callable[[Backend, Any, str], Any]  # (backend, app, thread_id) -> 可比较的结果


# ---------------------------------------------------------------------------
# 图：三种形态
# ---------------------------------------------------------------------------
def _chain(b: Backend):
    """a -> b -> c（c 是终点）。"""
    g = b.state_graph_cls(b.make_state(SCHEMA))
    for nm in "abc":
        g.add_node(nm, (lambda n: (lambda s: {"n": 1, "log": [n]}))(nm))
    g.add_edge(b.START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", b.END)
    return g.compile(checkpointer=b.checkpointer_cls())


def _chain_with_conditional(b: Backend):
    """a -> b，b 的条件边按 n 的大小选 c 或 d。"""
    g = b.state_graph_cls(b.make_state(SCHEMA))
    g.add_node("a", lambda s: {"n": 1, "log": ["a"]})
    g.add_node("b", lambda s: {"log": ["b"]})
    g.add_node("c", lambda s: {"log": ["c"]})
    g.add_node("d", lambda s: {"log": ["d"]})
    g.add_edge(b.START, "a")
    g.add_edge("a", "b")
    g.add_conditional_edges("b", lambda s: "c" if s["n"] > 10 else "d",
                            {"c": "c", "d": "d"})
    g.add_edge("c", b.END)
    g.add_edge("d", b.END)
    return g.compile(checkpointer=b.checkpointer_cls())


def _hitl_payload(b: Backend):
    """a -> b(挂起等人批，答复本身是个 dict) -> c。"""
    g = b.state_graph_cls(b.make_state(SCHEMA))

    def ask(s):
        decision = b.interrupt({"请回复": "approve | reject"})
        return {"n": 10, "log": [f"b:{decision['action']}"]}

    g.add_node("a", lambda s: {"n": 1, "log": ["a"]})
    g.add_node("b", ask)
    g.add_node("c", lambda s: {"n": 100, "log": ["c"]})
    g.add_edge(b.START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", b.END)
    return g.compile(checkpointer=b.checkpointer_cls())


def _hitl(b: Backend):
    """a -> b(挂起等人批) -> c。"""
    g = b.state_graph_cls(b.make_state(SCHEMA))

    def ask(s):
        ans = b.interrupt({"ask": s["n"]})
        return {"n": 10, "log": [f"b:{ans}"]}

    g.add_node("a", lambda s: {"n": 1, "log": ["a"]})
    g.add_node("b", ask)
    g.add_node("c", lambda s: {"n": 100, "log": ["c"]})
    g.add_edge(b.START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", b.END)
    return g.compile(checkpointer=b.checkpointer_cls())


# ---------------------------------------------------------------------------
# 场景
# ---------------------------------------------------------------------------
def _run_update_state(b: Backend, app, tid: str, as_node: str | None,
                      values: dict) -> dict:
    cfg = b.config(tid)
    app.invoke({"n": 0, "log": []}, cfg)
    app.update_state(cfg, values, **({"as_node": as_node} if as_node else {}))
    st = app.get_state(cfg)
    return {"next": tuple(st.next or ()), "values": dict(st.values)}


def _case_update_state_successor(b, app, tid):
    """as_node 指向中间节点 -> 跑的是它的**后继**。"""
    return _run_update_state(b, app, tid, "b", {"n": 100})


def _case_update_state_terminal(b, app, tid):
    """as_node 指向终点节点 -> 它的出边是 END -> 图直接结束。"""
    return _run_update_state(b, app, tid, "c", {"n": 100})


def _case_update_state_conditional(b, app, tid):
    """as_node 带条件边 -> 条件边按**合并后**的状态重新求值。"""
    return _run_update_state(b, app, tid, "b", {"n": 100})


def _case_update_state_no_as_node(b, app, tid):
    """不给 as_node -> 从水位线推断"最后一个写过状态的节点"。"""
    return _run_update_state(b, app, tid, None, {"n": 100})


def _case_update_state_fresh_thread(b, app, tid):
    """thread 上还没有任何 checkpoint -> 以空快照为底照样能改。"""
    cfg = b.config(tid)
    app.update_state(cfg, {"n": 1}, as_node="a")
    st = app.get_state(cfg)
    return {"next": tuple(st.next or ()), "values": dict(st.values)}


def _case_update_state_then_run(b, app, tid):
    """改写之后继续跑完，看最终状态。"""
    cfg = b.config(tid)
    app.invoke({"n": 0, "log": []}, cfg)
    app.update_state(cfg, {"n": 100}, as_node="b")
    st = app.get_state(cfg)
    out = app.invoke(None, cfg)
    return {"next": tuple(st.next or ()), "final": dict(out)}


def _case_time_travel(b, app, tid):
    """回到第 1 步的快照，改道重跑。"""
    cfg = b.config(tid)
    app.invoke({"n": 0, "log": []}, cfg)
    hist = b.history(app, cfg)
    step1 = next(c for s, c in hist if s == 1)
    app.update_state(step1, {"n": 100}, as_node="a")
    out = app.invoke(None, cfg)
    st = app.get_state(cfg)
    return {"next": tuple(st.next or ()), "final": dict(out)}


def _case_interrupt_resume(b, app, tid):
    """挂起 -> 答复 -> 跑完。"""
    cfg = b.config(tid)
    app.invoke({"n": 0, "log": []}, cfg)
    st = app.get_state(cfg)
    suspended = {"next": tuple(st.next or ()), "values": dict(st.values)}
    b.resume(app, tid, "yes")
    final = app.get_state(cfg)
    return {"suspended": suspended, "final": dict(final.values)}


def _case_interrupt_resume_payload(b, app, tid):
    """答复本身是个 dict 时，整个 dict 原样交给 interrupt()。"""
    cfg = b.config(tid)
    app.invoke({"n": 0, "log": []}, cfg)
    st = app.get_state(cfg)
    suspended = {"next": tuple(st.next or ()), "values": dict(st.values)}
    b.resume(app, tid, {"action": "approve"})
    final = app.get_state(cfg)
    return {"suspended": suspended, "final": dict(final.values)}


SCENARIOS: list[CheckpointScenario] = [
    CheckpointScenario("update_state_successor", _chain,
                       _case_update_state_successor),
    CheckpointScenario("update_state_terminal", _chain,
                       _case_update_state_terminal),
    CheckpointScenario("update_state_conditional", _chain_with_conditional,
                       _case_update_state_conditional),
    CheckpointScenario("update_state_no_as_node", _chain,
                       _case_update_state_no_as_node),
    CheckpointScenario("update_state_fresh_thread", _chain,
                       _case_update_state_fresh_thread),
    CheckpointScenario("update_state_then_run", _chain,
                       _case_update_state_then_run),
    CheckpointScenario("time_travel", _chain, _case_time_travel),
    CheckpointScenario("interrupt_resume", _hitl, _case_interrupt_resume),
    CheckpointScenario("interrupt_resume_payload", _hitl_payload,
                       _case_interrupt_resume_payload),
]
