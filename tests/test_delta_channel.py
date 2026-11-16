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
    """
    每步的**全部**写入都记进 pending_writes —— 数据通道和触发通道都记
    （对齐 LangGraph 的 writes 表：branch:to:X = null 也留痕）。
    """
    app = _build(InMemorySaver())
    app.invoke({"messages": [{"id": "h1", "role": "human", "content": "q"}]},
               {"thread_id": "t"})
    raw = app.checkpointer.list("t")[-1]
    # 顺序 = 应用顺序：输入 -> __start__ 点亮入口 -> 节点写入
    assert [c for c, _ in raw.pending_writes] == [
        "messages", "branch:to:a", "messages"]
    # 触发信号没有载荷（版本才重要），值是 None
    assert raw.pending_writes[1][1] is None
    # delta 重放只按通道名挑 messages 那几条，控制写入不参与
    h = app.checkpointer.get_delta_channel_history("t", ["messages"])
    assert len(h["messages"]["writes"]) == 2


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


# ------------------------------------------------------- 7. 存储结构：writes 表 + 父链
def _fresh_savers():
    import os, tempfile
    return [InMemorySaver(),
            SqliteSaver(os.path.join(tempfile.mkdtemp(), "t.db"))]


def test_pending_writes_stored_in_writes_table():
    """
    SqliteSaver 把增量写入存进独立的 writes 表，不塞 checkpoint 的 JSON blob。

    对齐 LangGraph 的 checkpoints + writes 两表结构：blob 只装内核状态，
    写入日志按 (checkpoint_id, idx) 有序存放。读出来时 pending_writes
    拼回 Checkpoint —— 它从存储字段变成了**查询视图**。
    """
    import json, os, tempfile

    saver = SqliteSaver(os.path.join(tempfile.mkdtemp(), "t.db"))
    app = _build(saver)
    app.invoke({"messages": [{"id": "h1", "role": "human", "content": "q"}]},
               {"thread_id": "t"})

    raw = saver.conn.execute("SELECT data FROM checkpoints").fetchone()[0]
    assert "pending_writes" not in json.loads(raw), "blob 里不该再有写入日志"

    rows = saver.conn.execute(
        "SELECT channel, idx FROM writes ORDER BY idx").fetchall()
    assert [r[0] for r in rows] == ["messages", "branch:to:a", "messages"]
    assert [r[1] for r in rows] == [0, 1, 2], "idx 保住同一步内多条写入的顺序"

    # 视图拼回：拿到的 Checkpoint 仍是自包含的
    assert [c for c, _ in saver.get("t").pending_writes] == [
        "messages", "branch:to:a", "messages"]


def test_parent_chain_links_linear_history():
    """线性历史：每份的 parent 是前一份，首份为 None。"""
    for saver in _fresh_savers():
        app = _build(saver)
        cfg = {"thread_id": "t"}
        for i in range(1, 4):
            app.invoke({"messages": [{"id": f"h{i}", "role": "human",
                                      "content": "q"}]}, cfg)
        hist = saver.list("t")
        assert hist[0].parent_id is None, f"{type(saver).__name__}"
        for prev, cur in zip(hist, hist[1:]):
            assert cur.parent_id == prev.id, f"{type(saver).__name__}"


def test_update_state_writes_delta_snapshot():
    """
    update_state 写了 delta 通道 -> 落一份 DeltaSnapshot。

    曾经这里写的是 `if chan in rt.input_writes` —— input_writes 是
    [(通道, 值)] 元组列表，拿字符串去 in 永远为 False，于是**从不落快照**。
    线性历史下靠重放碰巧也是对的，只有分叉时才暴露。
    """
    app = _build(InMemorySaver())
    cfg = {"thread_id": "t"}
    app.invoke({"messages": [{"id": "h1", "role": "human", "content": "q"}]}, cfg)
    app.update_state(cfg, {"messages": [{"id": "manual", "role": "human",
                                          "content": "m"}]}, as_node="a")
    raw = app.checkpointer.list("t")[-1]
    assert isinstance(raw.channel_values.get("messages"), DeltaSnapshot)


def test_fork_delta_replay_isolated():
    """
    从历史点分叉后，两条线的 delta 重放**互不串**。

    存储里两条线的 checkpoint 是交错追加的。若按写入顺序（seq）往回扫，
    分叉线会把主线的 h2/h3 也重放进来。必须沿 parent 链走 ——
    只有 on-path 祖先参与（对齐 LangGraph 的 parent_checkpoint_id）。
    """
    for saver in _fresh_savers():
        app = _build(saver)
        cfg = {"thread_id": "t"}
        for i in (1, 2, 3):
            app.invoke({"messages": [{"id": f"h{i}", "role": "human",
                                      "content": "q"}]}, cfg)
        hist = saver.list("t")
        snap1, snap3 = hist[0], hist[-1]

        # 从 snap1 分叉：manual 只属于分叉线
        app.update_state({**cfg, "checkpoint_id": snap1.id},
                         {"messages": [{"id": "manual", "role": "human",
                                        "content": "m"}]}, as_node="a")
        assert _ids(app.get_state(cfg)) == ["h1", "a1", "manual"], \
            f"{type(saver).__name__}: 分叉点就混进了主线的消息"

        # 分叉线继续跑（产生不写快照的 checkpoint -> 必须靠重放）
        app.invoke({"messages": [{"id": "h9", "role": "human",
                                  "content": "q"}]}, cfg)
        assert _ids(app.get_state(cfg)) == ["h1", "a1", "manual", "h9"], \
            f"{type(saver).__name__}: 分叉线重放串到了主线"

        # 主线从 snap3 继续：不该看见 manual / h9
        app.invoke({"messages": [{"id": "h10", "role": "human",
                                  "content": "q"}]},
                   {**cfg, "checkpoint_id": snap3.id})
        last = saver.list("t")[-1]
        assert last.parent_id == snap3.id, "新 checkpoint 的父节点是恢复源"
        assert _ids(app.get_state({**cfg, "checkpoint_id": last.id})) == \
            ["h1", "a1", "h2", "h3", "h10"], \
            f"{type(saver).__name__}: 主线重放串到了分叉线"


def test_legacy_db_migrates_and_replays():
    """
    旧格式的库（无 parent 列、pending_writes 在 data blob 里）打开即用：

      - 自动 ALTER 加 parent_checkpoint_id 列
      - 旧数据是线性历史，按 seq 顺序回填 parent
      - writes 表查不到时，回退读 data blob 里的 pending_writes
      - 旧数据 + 新写入混合后，重放依然正确
    """
    import json, os, sqlite3, tempfile, time
    from minigraph.checkpoint_sqlite import _ckpt_to_dict, _encode

    # ① 造两份旧格式数据（pending_writes 放进 data blob）
    src = SqliteSaver(os.path.join(tempfile.mkdtemp(), "src.db"))
    app = _build(src)
    cfg = {"thread_id": "t"}
    for i in (1, 2):
        app.invoke({"messages": [{"id": f"h{i}", "role": "human",
                                  "content": "q"}]}, cfg)
    old_rows = []
    for ck in src.list("t"):
        d = _ckpt_to_dict(ck)
        d["pending_writes"] = ck.pending_writes
        old_rows.append((ck.id, ck.step,
                         json.dumps(_encode(d), ensure_ascii=False).encode()))

    # ② 旧 schema：没有 parent_checkpoint_id 列，也没有 writes 表
    db = os.path.join(tempfile.mkdtemp(), "old.db")
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE checkpoints (
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT NOT NULL, checkpoint_id TEXT NOT NULL,
            step INTEGER NOT NULL, data BLOB NOT NULL,
            created_at REAL NOT NULL)""")
    for cid, step, blob in old_rows:
        conn.execute(
            "INSERT INTO checkpoints (thread_id, checkpoint_id, step,"
            " data, created_at) VALUES ('t', ?, ?, ?, ?)",
            (cid, step, blob, time.time()))
    conn.commit()
    conn.close()

    # ③ 打开 = 迁移
    saver = SqliteSaver(db)
    hist = saver.list("t")
    assert hist[0].parent_id is None
    assert hist[1].parent_id == hist[0].id, "旧线性历史按 seq 回填 parent"
    assert len(hist[-1].pending_writes) == 3, "旧格式 pending_writes 回退读取"

    # ④ 旧数据能重建；继续写入（新格式）后依然正确
    app2 = _build(saver)
    assert _ids(app2.get_state(cfg)) == ["h1", "a1", "h2"]
    app2.invoke({"messages": [{"id": "h3", "role": "human", "content": "q"}]}, cfg)
    assert _ids(app2.get_state(cfg)) == ["h1", "a1", "h2", "h3"]
