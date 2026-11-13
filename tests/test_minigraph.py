"""minigraph 的行为测试：每条用例对应 LangGraph 的一个核心机制。"""

import operator
from typing import Annotated

import pytest

from minigraph.builder import StateGraph
from minigraph.checkpoint import InMemorySaver
from minigraph.constants import END, START
from minigraph.context import interrupt
from minigraph.control import Command, Send
from minigraph.state import LAST_VALUE, State, add_messages, merge_dict


# ---------------------------------------------------------------- 1. 基本流转
def test_linear_and_conditional_loop():
    class S(State):
        n: Annotated[int, operator.add]
        trace: Annotated[list, operator.add]

    g = StateGraph(S)
    g.add_node("inc", lambda s: {"n": 1, "trace": ["inc"]})
    g.add_node("check", lambda s: {"trace": ["check"]})
    g.add_edge(START, "inc")
    g.add_edge("inc", "check")
    g.add_conditional_edges(
        "check", lambda s: "again" if s["n"] < 3 else "stop", {"again": "inc", "stop": END}
    )
    out = g.compile().invoke({"n": 0, "trace": []})
    assert out["n"] == 3
    assert out["trace"].count("inc") == 3


# ---------------------------------------------------------------- 2. reducer
def test_reducer_merges_partial_updates():
    class S(State):
        msgs: Annotated[list, add_messages]
        meta: Annotated[dict, merge_dict]
        name: str  # last-write-wins

    g = StateGraph(S)
    g.add_node("a", lambda s: {"msgs": [{"id": 1, "c": "hi"}], "meta": {"x": 1}, "name": "n1"})
    g.add_node("b", lambda s: {"msgs": [{"id": 2, "c": "yo"}], "meta": {"y": 2}, "name": "n2"})
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    out = g.compile().invoke({"msgs": [], "meta": {}, "name": ""})
    assert [m["c"] for m in out["msgs"]] == ["hi", "yo"]
    assert out["meta"] == {"x": 1, "y": 2}
    assert out["name"] == "n2"


def test_add_messages_updates_by_id_instead_of_appending():
    class S(State):
        msgs: Annotated[list, add_messages]

    g = StateGraph(S)
    g.add_node("a", lambda s: {"msgs": [{"id": 1, "c": "v1"}]})
    g.add_node("b", lambda s: {"msgs": [{"id": 1, "c": "v2"}]})
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    out = g.compile().invoke({"msgs": []})
    assert out["msgs"] == [{"id": 1, "c": "v2"}]


# ---------------------------------------------------------------- 3. 并行 / Send
def test_send_fan_out_collects_parallel_results():
    class S(State):
        topics: Annotated[list, operator.add]
        results: Annotated[list, operator.add]

    def fan(s):
        return [Send("worker", {"topic": t}) for t in s["topics"]]

    g = StateGraph(S)
    g.add_node("plan", lambda s: {"topics": ["a", "b", "c"]})
    g.add_node("fan", fan)
    g.add_node("worker", lambda s: {"results": [s["topic"].upper()]})
    g.add_edge(START, "plan")
    g.add_edge("plan", "fan")
    g.add_edge("fan", "worker")
    g.add_edge("worker", END)
    out = g.compile().invoke({"topics": [], "results": []})
    assert sorted(out["results"]) == ["A", "B", "C"]


def test_parallel_branch_pointing_at_end_does_not_crash():
    """常见写法：两个并行节点一个接 aggregator、一个直接 END。"""

    class S(State):
        a: Annotated[list, operator.add]
        b: Annotated[list, operator.add]

    g = StateGraph(S)
    g.add_node("fork", lambda s: {})
    g.add_node("x", lambda s: {"a": ["x"]})
    g.add_node("y", lambda s: {"b": ["y"]})
    g.add_edge(START, "fork")
    g.add_edge("fork", "x")
    g.add_edge("fork", "y")
    g.add_edge("x", END)
    g.add_edge("y", END)
    out = g.compile().invoke({"a": [], "b": []})
    assert out == {"a": ["x"], "b": ["y"]}


def test_concurrent_write_without_reducer_is_rejected():
    """同一超步两个节点写同一个 last-write-wins 字段 -> 必须报错。"""

    class S(State):
        v: str

    g = StateGraph(S)
    g.add_node("fork", lambda s: {})
    g.add_node("x", lambda s: {"v": "x"})
    g.add_node("y", lambda s: {"v": "y"})
    g.add_edge(START, "fork")
    g.add_edge("fork", "x")
    g.add_edge("fork", "y")
    g.add_edge("x", END)
    g.add_edge("y", END)
    with pytest.raises(RuntimeError, match="同一超步"):
        g.compile().invoke({"v": ""})


# ---------------------------------------------------------------- 4. Command
def test_command_does_goto_and_update():
    class S(State):
        path: Annotated[list, operator.add]

    def router(s):
        # 节点自己决定去哪，并同时提交状态
        return Command(goto="right" if s["score"] > 5 else "left", update={"path": ["router"]})

    g = StateGraph(S)
    g.add_node("router", router)
    g.add_node("left", lambda s: {"path": ["left"]})
    g.add_node("right", lambda s: {"path": ["right"]})
    g.add_edge(START, "router")
    g.add_edge("left", END)
    g.add_edge("right", END)
    app = g.compile()
    assert app.invoke({"score": 9, "path": []})["path"] == ["router", "right"]
    assert app.invoke({"score": 1, "path": []})["path"] == ["router", "left"]


# ---------------------------------------------------------------- 5. 检查点
def test_checkpoint_history_and_time_travel():
    class S(State):
        trace: Annotated[list, operator.add]

    g = StateGraph(S)
    g.add_node("a", lambda s: {"trace": ["a"]})
    g.add_node("b", lambda s: {"trace": ["b"]})
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    saver = InMemorySaver()
    app = g.compile(checkpointer=saver)
    cfg = {"thread_id": "t"}
    app.invoke({"trace": []}, cfg)

    history = app.get_state_history(cfg)
    assert [c.step for c in history] == [1, 2]

    # 回到第 1 步的状态，改道再跑一次。
    # as_node = "a" 表示"假装是 a 写的" -> 被点亮的是 **a 的出边**（a -> b），
    # 于是 b 被重新触发。对齐真实 LangGraph：update_state 跑的是 as_node 的
    # writers（出边），不是 as_node 自己。
    first = history[0]
    app.update_state({"thread_id": "t", "checkpoint_id": first.id}, {"trace": ["injected"]}, as_node="a")
    out = app.invoke(None, {"thread_id": "t"})
    assert out["trace"] == ["a", "injected", "b"]


def test_update_state_as_node_runs_the_successor_not_the_node():
    """as_node 是终点节点时，图直接结束 —— 因为 b 的出边是 END。"""
    class S(State):
        trace: Annotated[list, operator.add]

    g = StateGraph(S)
    g.add_node("a", lambda s: {"trace": ["a"]})
    g.add_node("b", lambda s: {"trace": ["b"]})
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    app = g.compile(checkpointer=InMemorySaver())
    cfg = {"thread_id": "t"}
    app.invoke({"trace": []}, cfg)

    first = app.get_state_history(cfg)[0]
    app.update_state({"thread_id": "t", "checkpoint_id": first.id},
                     {"trace": ["injected"]}, as_node="b")
    # b 的出边是 END -> 没有任何后继被点亮 -> 图到此为止
    assert app.get_state(cfg).next == ()
    out = app.invoke(None, {"thread_id": "t"})
    assert out["trace"] == ["a", "injected"]


# ---------------------------------------------------------------- 6. interrupt
def build_approval_graph():
    class S(State):
        draft: str
        log: Annotated[list, operator.add]

    def write(s):
        return {"draft": "初稿", "log": ["write"]}

    def review(s):
        answer = interrupt({"draft": s["draft"], "ask": "是否批准？"})
        return {"log": [f"review:{answer}"]}

    g = StateGraph(S)
    g.add_node("write", write)
    g.add_node("review", review)
    g.add_node("publish", lambda s: {"log": ["publish"]})
    g.add_edge(START, "write")
    g.add_edge("write", "review")
    g.add_edge("review", "publish")
    g.add_edge("publish", END)
    return g


def test_interrupt_suspends_then_resumes():
    app = build_approval_graph().compile(checkpointer=InMemorySaver())
    cfg = {"thread_id": "conv-1"}

    first = app.invoke({"draft": "", "log": []}, cfg)
    assert first == {"draft": "初稿", "log": ["write"]}      # review 尚未提交
    assert app.interrupted == {"draft": "初稿", "ask": "是否批准？"}
    assert app.get_state(cfg).interrupt is not None

    final = app.invoke(None, {**cfg, "resume": "批准"})
    assert final["log"] == ["write", "review:批准", "publish"]
    assert app.get_state(cfg).interrupt is None


def test_resume_without_input_raises():
    app = build_approval_graph().compile(checkpointer=InMemorySaver())
    cfg = {"thread_id": "c"}
    app.invoke({"draft": "", "log": []}, cfg)
    with pytest.raises(RuntimeError, match="interrupt"):
        app.invoke(None, cfg)


# ---------------------------------------------------------------- 7. 约束
def test_unknown_state_key_is_rejected():
    class S(State):
        a: int

    g = StateGraph(S)
    g.add_node("n", lambda s: {"oops": 1})
    g.add_edge(START, "n")
    with pytest.raises(ValueError, match="未声明"):
        g.compile().invoke({"a": 0})


def test_recursion_limit():
    class S(State):
        a: int

    g = StateGraph(S)
    g.add_node("n", lambda s: {"a": s["a"] + 1})
    g.add_edge(START, "n")
    g.add_edge("n", "n")   # 自环
    with pytest.raises(RuntimeError, match="recursion_limit"):
        g.compile().invoke({"a": 0}, {"recursion_limit": 5})


def test_retry_policy():
    class S(State):
        ok: bool

    calls = {"n": 0}

    def flaky(s):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("boom")
        return {"ok": True}

    g = StateGraph(S)
    g.add_node("flaky", {"fn": flaky, "retry": {"max_attempts": 3}})
    g.add_edge(START, "flaky")
    assert g.compile().invoke({"ok": False})["ok"] is True
    assert calls["n"] == 3


def test_interrupt_can_fire_multiple_times_in_a_loop():
    """同一节点在循环里反复挂起，每次都要取到对应的 resume 值（回归测试）。"""

    class S(State):
        round: Annotated[int, operator.add]
        answers: Annotated[list, operator.add]

    def ask(s):
        v = interrupt(f"第 {s['round'] + 1} 次询问")
        return {"round": 1, "answers": [v]}

    g = StateGraph(S)
    g.add_node("ask", ask)
    g.add_edge(START, "ask")
    g.add_conditional_edges("ask", lambda s: "more" if s["round"] < 3 else "done",
                            {"more": "ask", "done": END})
    app = g.compile(InMemorySaver())
    cfg = {"thread_id": "loop"}

    app.invoke({"round": 0, "answers": []}, cfg)
    assert app.interrupted == "第 1 次询问"
    app.invoke(None, {**cfg, "resume": "A"}, )
    assert app.interrupted == "第 2 次询问"
    app.invoke(None, {**cfg, "resume": "B"})
    assert app.interrupted == "第 3 次询问"
    out = app.invoke(None, {**cfg, "resume": "C"})
    assert out["answers"] == ["A", "B", "C"]


def test_parallel_execution_is_actually_concurrent():
    import time

    class S(State):
        hits: Annotated[list, operator.add]

    def slow(s):
        time.sleep(0.1)
        return {"hits": [1]}

    g = StateGraph(S)
    g.add_node("fork", lambda s: [Send("slow", {}), Send("slow", {}), Send("slow", {})])
    g.add_node("slow", slow)
    g.add_edge(START, "fork")
    g.add_edge("fork", "slow")
    g.add_edge("slow", END)
    t0 = time.time()
    out = g.compile().invoke({"hits": []})
    assert len(out["hits"]) == 3
    assert time.time() - t0 < 0.25      # 串行需要 0.3s+


def test_interrupt_inside_parallel_batch_rolls_back_the_superstep():
    """并行批里有人挂起 -> 本超步作废；恢复后整批原子重跑，结果既不丢也不重复。"""

    class S(State):
        log: Annotated[list, operator.add]

    def fast(s):
        return {"log": ["fast"]}

    def slow(s):
        v = interrupt("wait")
        return {"log": [f"slow:{v}"]}

    g = StateGraph(S)
    g.add_node("fork", lambda s: [Send("fast", {}), Send("slow", {})])
    g.add_node("fast", fast)
    g.add_node("slow", slow)
    g.add_edge(START, "fork")
    g.add_edge("fork", "fast")
    g.add_edge("fork", "slow")
    g.add_edge("fast", "slow")     # 让 fast / slow 落在同一超步（慢的挂起）
    g.add_edge("slow", END)

    app = g.compile(InMemorySaver())
    cfg = {"thread_id": "batch"}

    app.invoke({"log": []}, cfg)
    assert app.get_state(cfg).values["log"] == []      # 整步回滚，没有半截写入

    out = app.invoke(None, {**cfg, "resume": "ok"})
    assert out["log"] == ["fast", "slow:ok"]           # 不丢、不重复


# ---------------------------------------------------------------- 6b. 多个中断
def _two_interrupting_graph():
    """两个节点在同一超步里各自 interrupt。"""

    class S(State):
        log: Annotated[list, operator.add]

    g = StateGraph(S)
    g.add_node("fork", lambda s: [Send("a", {}), Send("b", {})])
    g.add_node("a", lambda s: {"log": [f"a:{interrupt('need-a')}"]})
    g.add_node("b", lambda s: {"log": [f"b:{interrupt('need-b')}"]})
    g.add_edge(START, "fork")
    g.add_edge("fork", "a")
    g.add_edge("fork", "b")
    g.add_edge("a", END)
    g.add_edge("b", END)
    return g


def test_multiple_interrupts_are_all_recorded():
    """同一超步里的多个挂起点必须**全部**记下来（只记一个会让另一个永远等不到值）。"""
    app = _two_interrupting_graph().compile(InMemorySaver())
    cfg = {"thread_id": "multi"}

    app.invoke({"log": []}, cfg)
    pending = app.get_state(cfg).interrupts
    assert [p["node"] for p in pending] == ["a", "b"]
    assert [p["value"] for p in pending] == ["need-a", "need-b"]


def test_multiple_interrupts_resumed_with_a_mapping():
    """用 dict 一次性答复多个挂起点（对齐 LangGraph 的 resume map）。"""
    app = _two_interrupting_graph().compile(InMemorySaver())
    cfg = {"thread_id": "multi-map"}

    app.invoke({"log": []}, cfg)
    out = app.invoke(None, {**cfg, "resume": {"a": "A", "b": "B"}})
    assert out["log"] == ["a:A", "b:B"]
    assert app.get_state(cfg).interrupt is None


def test_multiple_interrupts_resumed_one_by_one():
    """标量 resume 逐个答复：每次答复一个，剩下的继续挂着，不会死循环。"""
    app = _two_interrupting_graph().compile(InMemorySaver())
    cfg = {"thread_id": "multi-scalar"}

    app.invoke({"log": []}, cfg)
    app.invoke(None, {**cfg, "resume": "v2"})
    assert app.get_state(cfg).interrupt["node"] == "b"    # a 已答复，b 还在等

    out = app.invoke(None, {**cfg, "resume": "v3"})
    assert out["log"] == ["a:v2", "b:v3"]
    assert app.get_state(cfg).interrupt is None


def test_resume_payload_dict_is_passed_through_verbatim():
    """
    LangGraph 风格的答复：resume 的值本身是个 dict，原样交给 interrupt()。

    真实 LangGraph 的 Command(resume=X) 里，interrupt() 拿到的就是 X 本身，
    不做"键是不是节点名"的匹配。少了这条，{"action": "approve"} 这种
    payload 会被误当成"按节点名批量答复"而静默丢掉 —— 节点原地再挂一次。
    """

    class S(State):
        log: Annotated[list, operator.add]

    def ask(s):
        decision = interrupt({"请回复": "approve | reject"})
        return {"log": [decision["action"]]}

    g = StateGraph(S)
    g.add_node("ask", ask)
    g.add_edge(START, "ask")
    g.add_edge("ask", END)

    app = g.compile(InMemorySaver())
    cfg = {"thread_id": "payload"}
    app.invoke({"log": []}, cfg)
    out = app.invoke(None, {**cfg, "resume": {"action": "approve"}})
    assert out["log"] == ["approve"]
    assert app.get_state(cfg).interrupt is None


def test_one_node_interrupting_twice_consumes_values_in_order():
    """同一节点里连续两次 interrupt：值按调用顺序出队（对齐 LangGraph）。"""

    class S(State):
        log: Annotated[list, operator.add]

    def twice(s):
        v1 = interrupt("第一次")
        v2 = interrupt("第二次")
        return {"log": [f"{v1}/{v2}"]}

    g = StateGraph(S)
    g.add_node("ask", twice)
    g.add_edge(START, "ask")
    g.add_edge("ask", END)

    app = g.compile(InMemorySaver())
    cfg = {"thread_id": "twice"}
    app.invoke({"log": []}, cfg)
    out = app.invoke(None, {**cfg, "resume": {"ask": ["V1", "V2"]}})
    assert out["log"] == ["V1/V2"]


# ---------------------------------------------------------------- 8. 状态 schema
def test_state_schema_reads_own_annotations():
    """State 子类的 reducer 必须从**本类自己**的注解里提取。

    Python 3.14 起 (PEP 649) 注解是惰性求值的，类字典里没有
    "__annotations__" 这一项 —— 取错地方会**静默**得到空 schema，
    表现为运行时每个字段写入都报"写了未声明的字段"。
    """
    class S(State):
        n: Annotated[int, operator.add]
        plain: str

    assert set(S.reducers) == {"n", "plain"}
    assert S.reducers["n"] is operator.add
    assert S.reducers["plain"] is LAST_VALUE      # 没写 Annotated -> last-write-wins


def test_state_schema_created_via_type():
    """用 type(...) 造 schema（差分测试后端就是这么造的）也要能取到 reducer。"""
    S = type("S", (State,), {"__annotations__": {"n": Annotated[int, operator.add]}})
    assert S.reducers["n"] is operator.add


# ---------------------------------------------------------------- 9. 触发规则
def test_self_loop_via_command_keeps_running():
    """节点写**自己的**触发通道时，靠的是"水位线记写入前的版本"这条规则。

    若水位线记的是写入后的版本，节点就把自己这一轮的触发信号抹掉了，
    回边转不起来 —— 循环只跑一轮就停。
    """
    class S(State):
        n: Annotated[int, operator.add]

    seen_n: list[int] = []

    def spin(state):
        seen_n.append(state.get("n", 0))
        if state.get("n", 0) < 3:
            return Command(goto="spin", update={"n": 1})
        return {"n": 1}

    g = StateGraph(S)
    g.add_node("spin", spin)
    g.add_edge(START, "spin")

    out = g.compile().invoke({"n": 0})
    assert seen_n == [0, 1, 2, 3]         # 靠版本推进自己唤醒自己，转了三轮
    assert out["n"] == 4

