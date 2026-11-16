"""
DeltaChannel 测试：checkpoint 里只存增量，状态靠重放重建。

对照 LangGraph 的 langgraph/channels/delta.py + langgraph/pregel/_checkpoint.py。

核心机制：
  - checkpoint 不存 delta 通道的值（只存哨兵 / 快照）
  - 每步的增量写入记进 pending_writes
  - 恢复时沿祖先链找「种子 + 增量写入」，重放重建状态
  - 两个计数器（写入次数 / 超步数）决定何时落完整快照
"""

import operator
from typing import Annotated

import pytest

from minigraph.builder import StateGraph
from minigraph.checkpoint import InMemorySaver
from minigraph.checkpoint_sqlite import SqliteSaver
from minigraph.channels import DeltaChannel, DeltaSnapshot
from minigraph.constants import END, START
from minigraph.context import interrupt
from minigraph.state import State, add_messages


def _ids(snap):
    return [m["id"] for m in snap.values.get("messages", [])]


def _build(saver, snapshot_frequency=1000):
    """一个最简单的图：START -> a -> END，a 写一条消息。"""

    class S(State):
        messages: Annotated[
            list, DeltaChannel(add_messages, list,
                               snapshot_frequency=snapshot_frequency)
        ]

    def a(state):
        return {"messages": [{"id": "a1", "role": "assistant", "content": "hi"}]}

    g = StateGraph(S)
    g.add_node("a", a)
    g.add_edge(START, "a")
    g.add_edge("a", END)
    return g.compile(checkpointer=saver)


# ---------------------------------------------------------------- 1. 基本语义
def test_delta_channel_compiles_to_delta_channel():
    """声明了 DeltaChannel 的字段，编译出来就是 DeltaChannel。"""
    app = _build(InMemorySaver())
    assert isinstance(app.channels["messages"], DeltaChannel)


def test_delta_value_not_stored_in_checkpoint():
    """checkpoint 的 channel_values 里**没有**这个通道 —— 值不落盘。"""
    app = _build(InMemorySaver())
    app.invoke({"messages": [{"id": "h1", "role": "human", "content": "q"}]},
               {"thread_id": "t"})
    snap = app.get_state({"thread_id": "t"})
    assert "messages" not in snap.channel_values


def test_delta_value_not_stored_in_values():
    """
    values 里也不该有完整值 —— 那是省空间的大头。

    values 是"扁平状态投影"，delta 通道的值由「种子 + 增量写入」表达，
    再存一遍就是白占。
    """
    app = _build(InMemorySaver())
    app.invoke({"messages": [{"id": "h1", "role": "human", "content": "q"}]},
               {"thread_id": "t"})
    snap = app.get_state({"thread_id": "t"})
    # get_state 会把 delta 字段补回来（用户看到的是完整状态）
    assert _ids(snap) == ["h1", "a1"]
    # 但**存下去**的那份不含它
    raw = app.checkpointer.list("t")[-1]
    assert "messages" not in raw.values


def test_pending_writes_recorded():
    """每步的增量写入要记进 pending_writes —— 重放靠它。"""
    app = _build(InMemorySaver())
    app.invoke({"messages": [{"id": "h1", "role": "human", "content": "q"}]},
               {"thread_id": "t"})
    raw = app.checkpointer.list("t")[-1]
    chans = [c for c, _ in raw.pending_writes]
    assert chans == ["messages", "messages"]   # 输入 + 节点写入


# ---------------------------------------------------------------- 2. 重建
def test_state_reconstructed_across_invocations():
    """同一个 thread 反复 invoke，历史一条条累积 —— 靠重放重建。"""
    app = _build(InMemorySaver())
    cfg = {"thread_id": "t"}
    for i in range(1, 4):
        app.invoke({"messages": [{"id": f"h{i}", "role": "human",
                                  "content": "q"}]}, cfg)
    # a1 是同一个 id，add_messages 按 id 更新 -> 位置不变
    assert _ids(app.get_state(cfg)) == ["h1", "a1", "h2", "h3"]


def test_state_reconstructed_with_sqlite_saver():
    """SQLite saver 走同一条重建路径。"""
    import tempfile, os

    tmp = tempfile.mkdtemp()
    app = _build(SqliteSaver(os.path.join(tmp, "t.db")))
    cfg = {"thread_id": "t"}
    for i in range(1, 4):
        app.invoke({"messages": [{"id": f"h{i}", "role": "human",
                                  "content": "q"}]}, cfg)
    assert _ids(app.get_state(cfg)) == ["h1", "a1", "h2", "h3"]


def test_replay_order_is_old_to_new():
    """重放的顺序必须是旧->新 —— 顺序错了 reducer 算出来的值就不对。"""
    app = _build(InMemorySaver())
    cfg = {"thread_id": "t"}
    app.invoke({"messages": [{"id": "h1", "role": "human", "content": "q"}]}, cfg)
    app.invoke({"messages": [{"id": "h2", "role": "human", "content": "q"}]}, cfg)
    # h1 在 h2 之前
    ids = _ids(app.get_state(cfg))
    assert ids.index("h1") < ids.index("h2")


# ---------------------------------------------------------------- 3. 快照节奏
def test_snapshot_taken_at_snapshot_frequency():
    """写入次数达到 snapshot_frequency -> 落一份完整快照。"""
    app = _build(InMemorySaver(), snapshot_frequency=2)
    cfg = {"thread_id": "t"}
    for i in range(1, 5):
        app.invoke({"messages": [{"id": f"h{i}", "role": "human",
                                  "content": "q"}]}, cfg)
    history = app.checkpointer.list("t")
    # 快照以 DeltaSnapshot 的形式落进 channel_values
    snaps = [c for c in history
             if isinstance(c.channel_values.get("messages"), DeltaSnapshot)]
    assert snaps, "应该有快照"
    # 落了快照的通道，计数器归零
    assert any(c.delta_counters.get("messages") == (0, 0) for c in history)


def test_snapshot_keeps_state_correct():
    """落了快照之后，状态仍然要重建得对。"""
    app = _build(InMemorySaver(), snapshot_frequency=2)
    cfg = {"thread_id": "t"}
    for i in range(1, 6):
        app.invoke({"messages": [{"id": f"h{i}", "role": "human",
                                  "content": "q"}]}, cfg)
    assert _ids(app.get_state(cfg)) == ["h1", "a1", "h2", "h3", "h4", "h5"]


def test_no_snapshot_means_channel_absent():
    """没到快照频率 -> 通道根本不出现（省空间）。"""
    app = _build(InMemorySaver(), snapshot_frequency=1000)
    cfg = {"thread_id": "t"}
    app.invoke({"messages": [{"id": "h1", "role": "human", "content": "q"}]}, cfg)
    raw = app.checkpointer.list("t")[-1]
    assert "messages" not in raw.channel_values


def test_delta_saves_space():
    """
    同样的图，DeltaChannel 的 checkpoint 明显更小。

    对照组用普通的 add_messages（BinaryOperatorAggregate）——
    它每份 checkpoint 都存一遍完整状态，所以是 O(n²)。
    """
    import tempfile, os

    class Plain(State):
        messages: Annotated[list, add_messages]

    class Delta(State):
        messages: Annotated[list, DeltaChannel(add_messages, list)]

    def make(schema_cls, saver):
        def a(state):
            return {"messages": [{"id": "a1", "role": "assistant",
                                  "content": "x" * 100}]}

        g = StateGraph(schema_cls)
        g.add_node("a", a)
        g.add_edge(START, "a")
        g.add_edge("a", END)
        return g.compile(checkpointer=saver)

    sizes = {}
    for label, schema_cls in (("plain", Plain), ("delta", Delta)):
        tmp = tempfile.mkdtemp()
        saver = SqliteSaver(os.path.join(tmp, "t.db"))
        app = make(schema_cls, saver)
        cfg = {"thread_id": "t"}
        for i in range(20):
            app.invoke({"messages": [{"id": f"h{i}", "role": "human",
                                      "content": "y" * 100}]}, cfg)
        sizes[label] = saver.conn.execute(
            "SELECT SUM(LENGTH(data)) FROM checkpoints").fetchone()[0]

    assert sizes["delta"] < sizes["plain"]


# ---------------------------------------------------------------- 4. 其他路径
def test_update_state_preserves_delta_history():
    """update_state 之后，历史不能丢。"""
    app = _build(InMemorySaver())
    cfg = {"thread_id": "t"}
    app.invoke({"messages": [{"id": "h1", "role": "human", "content": "q"}]}, cfg)
    app.update_state(cfg, {"messages": [{"id": "manual", "role": "human",
                                          "content": "m"}]}, as_node="a")
    assert "manual" in _ids(app.get_state(cfg))
    assert "h1" in _ids(app.get_state(cfg))
    # 继续跑
    app.invoke({"messages": [{"id": "h2", "role": "human", "content": "q"}]}, cfg)
    ids = _ids(app.get_state(cfg))
    assert "manual" in ids and "h2" in ids and "h1" in ids


def test_interrupt_resume_preserves_delta_history():
    """挂起 -> 恢复，输入那条消息不能丢。"""
    class S(State):
        messages: Annotated[list, DeltaChannel(add_messages, list)]

    def a(state):
        ans = interrupt({"ask": "ok?"})
        return {"messages": [{"id": "a1", "role": "assistant",
                              "content": f"got {ans}"}]}

    g = StateGraph(S)
    g.add_node("a", a)
    g.add_edge(START, "a")
    g.add_edge("a", END)
    app = g.compile(checkpointer=InMemorySaver())
    cfg = {"thread_id": "t"}

    app.invoke({"messages": [{"id": "h1", "role": "human", "content": "q"}]}, cfg)
    # 挂起时输入已经进去了
    assert _ids(app.get_state(cfg)) == ["h1"]
    app.invoke(None, {**cfg, "resume": "yes"})
    assert _ids(app.get_state(cfg)) == ["h1", "a1"]


def test_state_history_shows_delta_values():
    """get_state_history 里的每一份快照，delta 字段都要补齐。"""
    app = _build(InMemorySaver())
    cfg = {"thread_id": "t"}
    for i in range(1, 4):
        app.invoke({"messages": [{"id": f"h{i}", "role": "human",
                                  "content": "q"}]}, cfg)
    history = app.get_state_history(cfg)
    assert len(history) >= 3
    for snap in history:
        assert "messages" in snap.values


# ---------------------------------------------------------------- 5. 其它 reducer
def test_delta_with_operator_add():
    """operator.add 也满足分结合并不变性，可以用在 DeltaChannel 上。"""

    class S(State):
        total: Annotated[int, DeltaChannel(operator.add, int)]

    def a(state):
        return {"total": 1}

    g = StateGraph(S)
    g.add_node("a", a)
    g.add_edge(START, "a")
    g.add_edge("a", END)
    app = g.compile(checkpointer=InMemorySaver())
    cfg = {"thread_id": "t"}
    for _ in range(3):
        app.invoke({"total": 0}, cfg)
    assert app.get_state(cfg).values["total"] == 3


def test_snapshot_frequency_must_be_positive():
    with pytest.raises(ValueError, match="snapshot_frequency"):
        DeltaChannel(add_messages, list, snapshot_frequency=0)


def test_seed_writes_not_double_applied():
    """
    找到种子的那份 checkpoint，它自己的写入**不能**再重放一遍。

    种子值 = 那个时刻的完整状态，已经包含了它自己的写入。再重放就是
    重复计算 —— 计数器这种**不幂等**的 reducer 会直接算错。

    （用 add_messages 测不出来：它按 id 覆盖，重复应用看不出差别。
    所以这里用 operator.add。）
    """
    import tempfile, os

    class S(State):
        total: Annotated[int, DeltaChannel(operator.add, int,
                                           snapshot_frequency=2)]

    def a(state):
        return {"total": 10}

    for saver in (InMemorySaver(),
                  SqliteSaver(os.path.join(tempfile.mkdtemp(), "t.db"))):
        g = StateGraph(S)
        g.add_node("a", a)
        g.add_edge(START, "a")
        g.add_edge("a", END)
        app = g.compile(checkpointer=saver)
        cfg = {"thread_id": "t"}
        for _ in range(5):
            app.invoke({"total": 1}, cfg)
        # 5 次输入(1) + 5 次节点写入(10) = 55
        assert app.get_state(cfg).values["total"] == 55, \
            f"{type(saver).__name__}: total 不对"


def test_superstep_bound_forces_snapshot(monkeypatch):
    """
    距上次快照过了太多超步 -> 强制落快照。

    为什么需要这个上界：只按"写入次数"计数的话，一个**很久没被写**的
    通道，重放链会一直变长（每份 checkpoint 都记着它还没被重放的那段）。
    这个上界保证重放深度有天花板。
    """
    import minigraph.runtime as rt_mod

    monkeypatch.setattr(rt_mod, "DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT", 3)

    class S(State):
        messages: Annotated[list, DeltaChannel(add_messages, list,
                                                snapshot_frequency=1000)]
        steps: Annotated[int, operator.add]

    def a(state):
        return {"messages": [{"id": "a1", "role": "assistant", "content": "hi"}]}

    def b(state):
        return {"steps": 1}          # 不碰 messages

    g = StateGraph(S)
    g.add_node("a", a)
    g.add_node("b", b)
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    app = g.compile(checkpointer=InMemorySaver())
    cfg = {"thread_id": "t"}
    for i in range(1, 5):
        app.invoke({"messages": [{"id": f"h{i}", "role": "human",
                                  "content": "q"}], "steps": 0}, cfg)

    history = app.checkpointer.list("t")
    # 写入次数远没到 1000，但上界到了 -> 一定有快照
    assert any(isinstance(c.channel_values.get("messages"), DeltaSnapshot)
               for c in history)
    # 状态仍然正确
    assert _ids(app.get_state(cfg)) == ["h1", "a1", "h2", "h3", "h4"]


# ---------------------------------------------------------------- 6. 批量 reducer
def test_batch_reducer_style():
    """
    LangGraph 的 DeltaChannel reducer 签名是 `(state, list[writes]) -> new_state`。

    minigraph 用 BatchReducer 显式标记这类 reducer（普通 callable 一律
    当二元 reducer 处理）—— 靠 arity 判断不可靠。
    """
    from minigraph.channels import BatchReducer

    def merge(state, writes):
        out = list(state)
        for w in writes:
            out.extend(w)
        return out

    class S(State):
        items: Annotated[list, DeltaChannel(BatchReducer(merge), list)]

    def a(state):
        return {"items": ["x", "y"]}

    g = StateGraph(S)
    g.add_node("a", a)
    g.add_edge(START, "a")
    g.add_edge("a", END)
    app = g.compile(checkpointer=InMemorySaver())
    cfg = {"thread_id": "t"}
    app.invoke({"items": ["seed"]}, cfg)
    app.invoke({"items": ["p"]}, cfg)
    # 节点 a 每次 invoke 都会跑一遍，所以 x/y 出现两次
    assert app.get_state(cfg).values["items"] == ["seed", "x", "y", "p", "x", "y"]


def test_batch_reducer_reconstructed_across_invocations():
    """批量 reducer 也要能靠重放重建。"""
    from minigraph.channels import BatchReducer

    def merge(state, writes):
        out = list(state)
        for w in writes:
            out.extend(w)
        return out

    class S(State):
        items: Annotated[list, DeltaChannel(BatchReducer(merge), list)]

    def a(state):
        return {"items": ["x"]}

    g = StateGraph(S)
    g.add_node("a", a)
    g.add_edge(START, "a")
    g.add_edge("a", END)
    app = g.compile(checkpointer=InMemorySaver())
    cfg = {"thread_id": "t"}
    for i in range(3):
        app.invoke({"items": [f"i{i}"]}, cfg)
    assert app.get_state(cfg).values["items"] == ["i0", "x", "i1", "x", "i2", "x"]
