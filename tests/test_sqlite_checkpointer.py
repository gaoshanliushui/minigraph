"""
SQLite checkpointer：落盘、跨进程恢复、隔离性、时间旅行。

关键约束：`SqliteSaver` 是 `InMemorySaver` 的**直接替身**。
所以行为类断言都用 `saver` fixture 跑两遍（内存版 + sqlite 版），
两边必须给出同样的结果 —— 任何"sqlite 版行为不同"都会在这里暴露。
"""

import operator
import sqlite3
from typing import Annotated

import pytest

from minigraph.builder import StateGraph
from minigraph.checkpoint import Checkpoint, InMemorySaver
from minigraph.checkpoint_sqlite import SqliteSaver
from minigraph.constants import END, MISSING, START
from minigraph.context import interrupt
from minigraph.control import Send
from minigraph.state import State


class S(State):
    log: Annotated[list, operator.add]
    draft: str


def build_graph():
    g = StateGraph(S)
    g.add_node("write", lambda s: {"draft": "初稿", "log": ["write"]})
    g.add_node("review", lambda s: {"log": [f"review:{interrupt('是否批准？')}"]})
    g.add_node("publish", lambda s: {"log": ["publish"]})
    g.add_edge(START, "write")
    g.add_edge("write", "review")
    g.add_edge("review", "publish")
    g.add_edge("publish", END)
    return g


def build_send_graph():
    g = StateGraph(S)
    g.add_node("fork", lambda s: [Send("worker", {"i": 1}), Send("worker", {"i": 2})])
    g.add_node("worker", lambda s: {"log": [f"w{s['i']}"]})
    g.add_edge(START, "fork")
    g.add_edge("fork", "worker")
    g.add_edge("worker", END)
    return g


def build_simple_graph():
    """没有 interrupt 的线性图 —— 用来测"同一 thread 多次调用"的语义。

    build_graph() 中间有个 interrupt()，跑到那儿就冻住了；测"追加语义"
    时不想被 HITL 那条线干扰，所以单独建一个干净的图。
    """
    g = StateGraph(S)
    g.add_node("echo", lambda s: {"log": [f"echo:{s['draft']}"], "draft": "done"})
    g.add_edge(START, "echo")
    g.add_edge("echo", END)
    return g


@pytest.fixture(params=["memory", "sqlite"])
def saver(request, tmp_path):
    """同一套断言对两种 saver 各跑一遍。"""
    if request.param == "memory":
        s = InMemorySaver()
    else:
        s = SqliteSaver(str(tmp_path / "cp.db"))
    yield s
    if hasattr(s, "close"):
        s.close()


# ---------------------------------------------------------------- 基本读写
def test_put_and_get_round_trip(saver):
    ck = Checkpoint(id=saver.new_id(), step=1, values={"a": 1},
                    next=("x",), tasks=["x"])
    saver.put("t1", ck)

    back = saver.get("t1")
    assert back.id == ck.id
    assert back.values == {"a": 1}
    assert back.next == ("x",)


def test_get_returns_none_for_unknown_thread():
    saver = SqliteSaver(":memory:")
    assert saver.get("nope") is None
    with pytest.raises(KeyError):
        saver.get("nope", "ckpt-0000-0000")


def test_list_keeps_insertion_order(saver):
    ids = []
    for i in range(5):
        ck = Checkpoint(id=saver.new_id(), step=i, values={"i": i}, next=(), tasks=[])
        saver.put("t", ck)
        ids.append(ck.id)
    history = saver.list("t")
    assert [c.id for c in history] == ids
    assert [c.step for c in history] == [0, 1, 2, 3, 4]


def test_threads_are_isolated(saver):
    saver.put("a", Checkpoint(id=saver.new_id(), step=1, values={"x": 1}, next=(), tasks=[]))
    saver.put("b", Checkpoint(id=saver.new_id(), step=1, values={"x": 2}, next=(), tasks=[]))
    assert saver.get("a").values == {"x": 1}
    assert saver.get("b").values == {"x": 2}


def test_get_a_specific_older_checkpoint(saver):
    first = Checkpoint(id=saver.new_id(), step=1, values={"v": 1}, next=(), tasks=[])
    saver.put("t", first)
    saver.put("t", Checkpoint(id=saver.new_id(), step=2, values={"v": 2}, next=(), tasks=[]))
    assert saver.get("t", first.id).values == {"v": 1}
    assert saver.get("t").values == {"v": 2}          # 不给 id -> 最新


# ---------------------------------------------------------------- 端到端
def test_graph_runs_with_sqlite_checkpointer(saver):
    app = build_graph().compile(checkpointer=saver)
    cfg = {"thread_id": "conv"}

    app.invoke({"log": [], "draft": ""}, cfg)
    assert app.get_state(cfg).interrupt is not None

    out = app.invoke(None, {**cfg, "resume": "批准"})
    assert out["log"] == ["write", "review:批准", "publish"]
    assert app.get_state(cfg).interrupt is None


def test_checkpoint_history_is_readable(saver):
    app = build_graph().compile(checkpointer=saver)
    cfg = {"thread_id": "conv"}
    app.invoke({"log": [], "draft": ""}, cfg)
    app.invoke(None, {**cfg, "resume": "批准"})

    hist = app.get_state_history(cfg)
    assert len(hist) >= 2
    assert hist[-1].interrupt is None
    assert hist[0].values["log"] == ["write"]      # 第一份快照 = 第一超步跑完
    assert hist[-1].values["log"] == ["write", "review:批准", "publish"]


def test_time_travel_to_an_earlier_checkpoint(saver):
    """回到历史 checkpoint 重新分支。"""
    app = build_graph().compile(checkpointer=saver)
    cfg = {"thread_id": "conv"}
    app.invoke({"log": [], "draft": ""}, cfg)

    first = app.get_state_history(cfg)[0]
    # 第一份快照还没挂起过，review 会重新挂起 —— 所以这里拿不到最终状态
    out = app.invoke(None, {**cfg, "checkpoint_id": first.id, "resume": "改主意"})
    assert out["log"] == ["write"]
    assert app.get_state(cfg).interrupt["node"] == "review"

    # 再答复一次就跑到头了
    out = app.invoke(None, {**cfg, "resume": "改主意"})
    assert out["log"] == ["write", "review:改主意", "publish"]


def test_update_state_with_sqlite_checkpointer(saver):
    app = build_graph().compile(checkpointer=saver)
    cfg = {"thread_id": "conv"}
    app.invoke({"log": [], "draft": ""}, cfg)

    cid = app.update_state(cfg, {"draft": "人工改"}, as_node="review")
    assert cid
    assert app.get_state(cfg).values["draft"] == "人工改"


# ---------------------------------------------------------------- sqlite 特有
def test_state_survives_process_restart(tmp_path):
    """换个 saver 实例（相当于重启进程）还能接着跑。"""
    db = str(tmp_path / "cp.db")

    saver = SqliteSaver(db)
    app = build_graph().compile(checkpointer=saver)
    cfg = {"thread_id": "conv"}
    app.invoke({"log": [], "draft": ""}, cfg)
    saver.close()

    # —— 重启 ——
    saver2 = SqliteSaver(db)
    app2 = build_graph().compile(checkpointer=saver2)
    out = app2.invoke(None, {**cfg, "resume": "批准"})
    assert out["log"] == ["write", "review:批准", "publish"]
    saver2.close()


def test_send_fanout_survives_restart(tmp_path):
    """Send 任务带的信息（arg）要能从库里读回来。"""
    db = str(tmp_path / "cp.db")
    saver = SqliteSaver(db)
    app = build_send_graph().compile(checkpointer=saver)
    cfg = {"thread_id": "s"}
    out = app.invoke({"log": [], "draft": ""}, cfg)
    assert sorted(out["log"]) == ["w1", "w2"]
    saver.close()

    saver2 = SqliteSaver(db)
    hist = saver2.list("s")
    # Send 对象必须原样读回来（否则恢复后并行分支会重建不出来）
    assert any(isinstance(t, Send) for c in hist for t in c.tasks)
    saver2.close()


# ------------------------------------------------- 同一 thread 多次调用
def test_new_input_on_an_existing_thread_is_appended(saver):
    """给已存在的 thread 传输入 = **追加**，不是替换。

    对齐真实 LangGraph：它也是先 channels_from_checkpoint() 恢复，
    再把 input 当作 INPUT 写进那些通道（langgraph/pregel/main.py:1793 起）。
    """
    app = build_simple_graph().compile(checkpointer=saver)
    cfg = {"thread_id": "conv"}

    out1 = app.invoke({"log": ["q1"], "draft": ""}, cfg)
    out2 = app.invoke({"log": ["q2"], "draft": ""}, cfg)

    # 两轮都留下了，而不是第二轮把第一轮顶掉
    assert out1["log"] == ["q1", "echo:"]
    assert out2["log"][:3] == ["q1", "echo:", "q2"]
    # 第二轮确实又跑了一遍节点
    assert out2["log"].count("echo:") == 2


def test_input_on_existing_thread_reruns_the_graph(saver):
    """追加输入之后，图要**再跑一遍** —— 不能因为水位线追上了就跳过。"""
    app = build_simple_graph().compile(checkpointer=saver)
    cfg = {"thread_id": "conv"}

    app.invoke({"log": [], "draft": ""}, cfg)
    first = app.get_state(cfg)

    app.invoke({"log": [], "draft": ""}, cfg)
    second = app.get_state(cfg)

    # 第二轮确实又跑了（超步数往前走了），不是原地不动
    assert second.step > first.step
    assert second.values["log"].count("echo:") == 2


def test_resume_without_input_does_not_rerun_the_entry(saver):
    """纯恢复（input=None）不该重复点亮入口通道。

    带输入 = 开新一轮对话；不带输入 = 接着上次继续。
    """
    app = build_graph().compile(checkpointer=saver)
    cfg = {"thread_id": "conv"}

    app.invoke({"log": [], "draft": ""}, cfg)
    before = app.get_state(cfg)

    app.invoke(None, {**cfg, "resume": "批准"})
    after = app.get_state(cfg)

    # 恢复后跑完了剩下的流程，但没有从头再来一遍
    assert after.values["log"].count("write") == 1
    assert after.step > before.step


def test_fresh_thread_starts_from_scratch(saver):
    """换个 thread_id 就是全新的对话 —— 不受别的 thread 影响。"""
    app = build_simple_graph().compile(checkpointer=saver)

    app.invoke({"log": ["q1"], "draft": ""}, {"thread_id": "a"})
    out = app.invoke({"log": ["q2"], "draft": ""}, {"thread_id": "b"})

    assert out["log"][:1] == ["q2"]
    assert "q1" not in out["log"]


def test_delete_thread():
    saver = SqliteSaver(":memory:")
    saver.put("a", Checkpoint(id=saver.new_id(), step=1, values={}, next=(), tasks=[]))
    saver.delete_thread("a")
    assert saver.list("a") == []
    assert saver.threads() == []


def test_accepts_an_existing_connection():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.put("t", Checkpoint(id=saver.new_id(), step=1, values={"a": 1},
                              next=(), tasks=[]))
    assert saver.get("t").values == {"a": 1}
    saver.close()


# ---------------------------------------------------------------- 序列化细节
def test_special_values_survive_round_trip():
    saver = SqliteSaver(":memory:")
    ck = Checkpoint(
        id=saver.new_id(), step=1,
        values={"s": {1, 2}, "t": (1, 2), "b": b"\x00\xff", "n": None},
        next=("a", "b"),
        tasks=[Send("w", {"i": 1})],
        channel_values={"c": MISSING, "j": {"x", "y"}},
        channel_versions={"c": 1},
        versions_seen={"n": {"branch:to:n": 2}},
    )
    saver.put("t", ck)
    back = saver.get("t")
    assert back.values["s"] == {1, 2}
    assert back.values["t"] == (1, 2)
    assert back.values["b"] == b"\x00\xff"
    assert back.next == ("a", "b")
    assert back.tasks == [Send("w", {"i": 1})]
    assert back.channel_values["c"] is MISSING          # 哨兵必须是同一个对象
    assert back.channel_values["j"] == {"x", "y"}


def test_user_dict_that_looks_like_a_tag_is_left_alone():
    """用户 state 里正好有 "__mg_type__" 键时，不能被误当成类型标签。"""
    saver = SqliteSaver(":memory:")
    ck = Checkpoint(id=saver.new_id(), step=1,
                    values={"__mg_type__": "tuple"},
                    next=(), tasks=[])
    saver.put("t", ck)
    assert saver.get("t").values == {"__mg_type__": "tuple"}


def test_task_writes_survive_round_trip():
    """挂起时存下的"已成功任务的写入"必须能原样读回来。"""
    saver = SqliteSaver(":memory:")
    ck = Checkpoint(
        id=saver.new_id(), step=1, values={}, next=(), tasks=[],
        task_writes={"effect@2#push:0": [("log", ["effect"]),
                                        ("branch:to:wait", None)]},
    )
    saver.put("t", ck)
    back = saver.get("t")
    assert back.task_writes == {
        "effect@2#push:0": [("log", ["effect"]), ("branch:to:wait", None)]}


def test_successful_sibling_is_not_reexecuted_with_sqlite(saver):
    """SQLite 版：挂起回滚后，跑成功的兄弟任务同样不该重跑。"""
    executed = []

    class T(State):
        log: Annotated[list, operator.add]

    def effect(s):
        executed.append("effect")
        return {"log": ["effect"]}

    def waits(s):
        return {"log": [f"wait:{interrupt('wait')}"]}

    g = StateGraph(T)
    g.add_node("fork", lambda s: [Send("effect", {}), Send("wait", {})])
    g.add_node("effect", effect)
    g.add_node("wait", waits)
    g.add_edge(START, "fork")
    g.add_edge("fork", "effect")
    g.add_edge("fork", "wait")
    g.add_edge("effect", "wait")
    g.add_edge("wait", END)

    app = g.compile(checkpointer=saver)
    cfg = {"thread_id": "sqlite-no-reexec"}
    app.invoke({"log": []}, cfg)
    assert executed == ["effect"]
    out = app.invoke(None, {**cfg, "resume": "ok"})
    assert executed == ["effect"]
    assert out["log"] == ["effect", "wait:ok"]


def test_unserializable_value_raises_clearly():
    class Weird:
        pass

    saver = SqliteSaver(":memory:")
    ck = Checkpoint(id=saver.new_id(), step=1,
                    values={"w": Weird()}, next=(), tasks=[])
    with pytest.raises(TypeError, match="无法序列化"):
        saver.put("t", ck)


def test_two_savers_agree_on_checkpoint_contents():
    """内存版和 sqlite 版存下来的 checkpoint 内容必须逐字段相同。"""
    def run(saver):
        app = build_graph().compile(checkpointer=saver)
        cfg = {"thread_id": "conv"}
        app.invoke({"log": [], "draft": ""}, cfg)
        app.invoke(None, {**cfg, "resume": "批准"})
        return [
            {k: getattr(c, k) for k in
             ("step", "values", "next", "tasks", "interrupt", "interrupts",
              "resumes", "slots", "channel_values", "channel_versions",
              "versions_seen")}
            for c in saver.list("conv")
        ]

    mem = run(InMemorySaver())
    sql = run(SqliteSaver(":memory:"))
    assert len(mem) == len(sql)
    for a, b in zip(mem, sql):
        assert a == b
