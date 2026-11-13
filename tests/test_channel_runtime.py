"""通道内核的回归测试：覆盖旧测试没有触达的触发、恢复与屏障语义。"""

import copy
import operator
from typing import Annotated

import pytest

from minigraph.builder import StateGraph
from minigraph.channels import (
    BinaryOperatorAggregate,
    EphemeralValue,
    LastValue,
    NamedBarrierValue,
    channel_for_field,
)
from minigraph.checkpoint import InMemorySaver
from minigraph.constants import MISSING, START, END
from minigraph.context import interrupt
from minigraph.control import Send
from minigraph.state import LAST_VALUE, State


def test_missing_is_a_singleton_across_copy_and_deepcopy():
    assert copy.copy(MISSING) is MISSING
    assert copy.deepcopy(MISSING) is MISSING


def test_reducer_determines_data_channel_type():
    assert isinstance(channel_for_field("v", None), LastValue)
    assert isinstance(channel_for_field("v", LAST_VALUE), LastValue)
    assert isinstance(channel_for_field("n", operator.add), BinaryOperatorAggregate)


def test_reducer_channel_starts_at_the_type_zero_value():
    """
    reducer 字段的通道初值是 typ()（list->[]、int->0），和 LangGraph 一致。

    于是节点读一个**还没被写过**的字段拿到的是零值，而不是 KeyError；
    拿不到类型时（比如裸 dict 形态声明的 schema）才退化成 MISSING。
    """
    ch = channel_for_field("log", operator.add, list)
    assert ch.is_available()
    assert ch.get() == []
    assert channel_for_field("n", operator.add, int).get() == 0
    # 没有类型 -> MISSING -> 不可用
    assert not channel_for_field("x", operator.add).is_available()


def test_reading_an_unwritten_reducer_field_yields_the_zero_value():
    """节点读一个从没被写过的 reducer 字段：拿到零值，不是 KeyError。"""

    class S(State):
        n: Annotated[int, operator.add]
        log: Annotated[list, operator.add]

    g = StateGraph(S)
    g.add_node("a", lambda s: {"log": list(s["log"]) + ["read"], "n": s["n"] + 1})
    g.add_edge(START, "a")
    g.add_edge("a", END)
    app = g.compile()
    out = app.invoke({})
    assert out["log"] == ["read"]
    assert out["n"] == 1


def test_ephemeral_value_clears_and_barrier_rearms_after_consume():
    branch = EphemeralValue("branch:to:a", guard=False)
    assert branch.update([None]) is True
    assert branch.is_available()
    assert branch.update([]) is True
    assert not branch.is_available()

    barrier = NamedBarrierValue("join:a+b:c", {"a", "b"})
    barrier.update(["a"])
    assert not barrier.is_available()
    barrier.update(["b"])
    assert barrier.is_available()
    assert barrier.consume() is True
    assert not barrier.is_available()
    barrier.update(["a", "b"])
    assert barrier.is_available()


def test_start_can_fan_out_to_multiple_entry_points():
    class S(State):
        log: Annotated[list, operator.add]

    g = StateGraph(S)
    g.add_node("a", lambda s: {"log": ["a"]})
    g.add_node("b", lambda s: {"log": ["b"]})
    g.add_edge(START, "a")
    g.add_edge(START, "b")
    g.add_edge("a", END)
    g.add_edge("b", END)
    app = g.compile()
    assert app.trigger_to_nodes["branch:to:a"] == ["a"]
    assert app.trigger_to_nodes["branch:to:b"] == ["b"]
    out = app.invoke({"log": []})
    assert out["log"] == ["a", "b"]


def test_asymmetric_fanin_triggers_target_twice_without_barrier():
    class S(State):
        log: Annotated[list, operator.add]

    g = StateGraph(S)
    g.add_node("fork", lambda s: {"log": ["fork"]})
    g.add_node("a", lambda s: {"log": ["a"]})
    g.add_node("b1", lambda s: {"log": ["b1"]})
    g.add_node("b2", lambda s: {"log": ["b2"]})
    g.add_node("c", lambda s: {"log": ["c"]})
    g.add_edge(START, "fork")
    g.add_edge("fork", "a")
    g.add_edge("fork", "b1")
    g.add_edge("a", "c")
    g.add_edge("b1", "b2")
    g.add_edge("b2", "c")
    g.add_edge("c", END)
    out = g.compile().invoke({"log": []})
    assert out["log"].count("c") == 2


def test_barrier_runs_target_once_and_rearms_on_the_next_round():
    class S(State):
        n: Annotated[int, operator.add]
        log: Annotated[list, operator.add]

    g = StateGraph(S)
    g.add_node("inc", lambda s: {"n": 1, "log": ["inc"]})
    g.add_node("a", lambda s: {"log": ["a"]})
    g.add_node("b", lambda s: {"log": ["b"]})
    g.add_node("join", lambda s: {"log": ["join"]})
    g.add_edge(START, "inc")
    g.add_edge("inc", "a")
    g.add_edge("inc", "b")
    g.add_edge(["a", "b"], "join")
    g.add_conditional_edges(
        "join", lambda s: "again" if s["n"] < 2 else "done",
        {"again": "inc", "done": END},
    )
    out = g.compile().invoke({"n": 0, "log": []})
    assert out["n"] == 2
    assert out["log"].count("join") == 2


def test_push_task_does_not_advance_the_targets_branch_watermark():
    class S(State):
        topics: list
        results: Annotated[list, operator.add]

    g = StateGraph(S)
    g.add_node("plan", lambda s: {"topics": ["a", "b"]})
    g.add_node("worker", lambda s: {"results": [s["t"]]})
    g.add_edge(START, "plan")
    g.add_conditional_edges(
        "plan", lambda s: [Send("worker", {"t": t}) for t in s["topics"]], ["worker"]
    )
    g.add_edge("worker", END)
    app = g.compile(InMemorySaver())
    cfg = {"thread_id": "push"}
    assert app.invoke({"topics": [], "results": []}, cfg)["results"] == ["a", "b"]
    assert app.get_state(cfg).versions_seen.get("worker", {}) == {}


def test_stream_updates_contain_state_fields_not_trigger_channels():
    class S(State):
        log: Annotated[list, operator.add]

    g = StateGraph(S)
    g.add_node("a", lambda s: {"log": ["a"]})
    g.add_node("b", lambda s: {"log": ["b"]})
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    chunks = [(node, update) for node, _step, update in g.compile().stream({"log": []})]
    assert chunks == [("a", {"log": ["a"]}), ("b", {"log": ["a", "b"]})]


def test_undeclared_input_keys_survive_interrupt_and_resume():
    class S(State):
        log: Annotated[list, operator.add]

    def review(s):
        answer = interrupt("continue?")
        return {"log": [answer]}

    g = StateGraph(S)
    g.add_node("review", review)
    g.add_node("finish", lambda s: {"log": [s["request_id"]]})
    g.add_edge(START, "review")
    g.add_edge("review", "finish")
    g.add_edge("finish", END)
    app = g.compile(InMemorySaver())
    cfg = {"thread_id": "extras"}

    first = app.invoke({"request_id": "r-1", "log": []}, cfg)
    assert first["request_id"] == "r-1"
    final = app.invoke(None, {**cfg, "resume": "ok"})
    assert final["log"] == ["ok", "r-1"]
    assert final["request_id"] == "r-1"


def test_conditional_edge_handles_first_write_to_reducer_field():
    class S(State):
        n: Annotated[int, operator.add]

    g = StateGraph(S)
    g.add_node("inc", lambda s: {"n": 1})
    g.add_edge(START, "inc")
    g.add_conditional_edges(
        "inc", lambda s: "again" if s["n"] < 2 else "done",
        {"again": "inc", "done": END},
    )
    assert g.compile().invoke({})["n"] == 2


def test_checkpoint_next_describes_the_upcoming_superstep():
    class S(State):
        log: Annotated[list, operator.add]

    g = StateGraph(S)
    g.add_node("a", lambda s: {"log": ["a"]})
    g.add_node("b", lambda s: {"log": ["b"]})
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    app = g.compile(InMemorySaver())
    cfg = {"thread_id": "next"}
    app.invoke({"log": []}, cfg)
    first, second = app.get_state_history(cfg)
    assert first.next == ("b",)
    assert first.tasks == ["b"]
    assert second.next == ()
    assert second.tasks == []


def test_graph_may_finish_exactly_at_recursion_limit():
    class S(State):
        n: Annotated[int, operator.add]

    g = StateGraph(S)
    g.add_node("once", lambda s: {"n": 1})
    g.add_edge(START, "once")
    assert g.compile().invoke({"n": 0}, {"recursion_limit": 1})["n"] == 1


def test_conditional_target_may_be_a_terminal_node():
    class S(State):
        log: Annotated[list, operator.add]

    g = StateGraph(S)
    g.add_node("start", lambda s: {"log": ["start"]})
    g.add_node("worker", lambda s: {"log": ["worker"]})
    g.add_edge(START, "start")
    g.add_conditional_edges("start", lambda s: "worker", ["worker"])
    assert g.compile().invoke({"log": []})["log"] == ["start", "worker"]


def test_conditional_path_map_accepts_a_node_name_list():
    class S(State):
        log: Annotated[list, operator.add]

    g = StateGraph(S)
    g.add_node("start", lambda s: {"log": ["start"]})
    g.add_node("worker", lambda s: {"log": ["worker"]})
    g.add_edge(START, "start")
    g.add_conditional_edges("start", lambda s: "worker", ["worker"])
    g.add_edge("worker", END)
    assert g.compile().invoke({"log": []})["log"] == ["start", "worker"]

    with pytest.raises(TypeError, match="path_map"):
        g.add_conditional_edges("start", lambda s: "worker", "worker")
