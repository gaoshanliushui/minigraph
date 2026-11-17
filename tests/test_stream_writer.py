"""stream_mode / get_stream_writer 的离线测试（不需要 LLM SDK）。

对齐 LangGraph 的 custom 模式：节点里 get_stream_writer() 写出去的数据，
调用方用 stream(stream_mode="custom") 实时收到。关键性质是**实时** ——
token 要在节点还没跑完的时候就流到消费者手里。
"""

import threading
import time

from minigraph.builder import StateGraph
from minigraph.constants import END, START
from minigraph.context import get_stream_writer
from minigraph.state import State


class S(State):
    n: int


def _build(fn):
    g = StateGraph(S)
    g.add_node("a", fn)
    g.add_edge(START, "a")
    g.add_edge("a", END)
    return g.compile()


def test_custom_mode_receives_writer_output():
    """节点里写出去的自定义数据，custom 模式按原样收到。"""

    def node(state):
        w = get_stream_writer()
        w({"type": "token", "text": "你好"})
        w({"type": "token", "text": "，世界"})
        return {"n": 1}

    app = _build(node)
    chunks = list(app.stream({"n": 0}, stream_mode="custom"))
    assert chunks == [{"type": "token", "text": "你好"},
                      {"type": "token", "text": "，世界"}]


def test_writer_is_noop_without_custom_mode():
    """没开 custom 模式时 writer 是 no-op：不报错，updates 照常。"""

    def node(state):
        get_stream_writer()({"type": "token", "text": "没人收"})
        return {"n": 1}

    app = _build(node)
    steps = list(app.stream({"n": 0}))          # 默认 updates
    assert len(steps) == 1
    node_name, step, update = steps[0]
    assert update["n"] == 1


def test_tokens_arrive_before_node_finishes():
    """实时性：token 必须在节点返回**之前**就能被消费者拿到。

    用事件同步：节点写完第一个 token 后阻塞等"消费者已收到"的信号，
    消费者收到第一个 token 才放行。若实现是把 token 憋到节点结束才
    一起吐，这里会死锁超时。
    """
    received = threading.Event()
    release = threading.Event()

    def node(state):
        w = get_stream_writer()
        w({"type": "token", "text": "first"})
        assert received.wait(timeout=5), "第一个 token 没能实时流出"
        release.set()
        w({"type": "token", "text": "second"})
        return {"n": 1}

    app = _build(node)
    seen = []
    for chunk in app.stream({"n": 0}, stream_mode="custom"):
        seen.append(chunk)
        if chunk["text"] == "first":
            received.set()          # 放行节点继续
        if chunk["text"] == "second":
            break
    assert release.is_set()
    assert [c["text"] for c in seen] == ["first", "second"]


def test_updates_and_custom_together():
    """多模式：updates 和 custom 一起收，每项带 mode 前缀。"""

    def node(state):
        w = get_stream_writer()
        w({"type": "token", "text": "hi"})
        return {"n": 42}

    app = _build(node)
    events = list(app.stream({"n": 0}, stream_mode=["updates", "custom"]))
    kinds = [m for m, _ in events]
    assert "custom" in kinds and "updates" in kinds
    custom = [p for m, p in events if m == "custom"]
    assert custom == [{"type": "token", "text": "hi"}]
    updates = [p for m, p in events if m == "updates"]
    assert updates[0][2]["n"] == 42


def test_values_mode_yields_full_state():
    """values 模式：每步产出完整 state 投影。"""

    def node(state):
        return {"n": state["n"] + 1}

    app = _build(node)
    values = list(app.stream({"n": 0}, stream_mode="values"))
    assert values[-1]["n"] == 1


def test_custom_writer_works_in_parallel_nodes():
    """并行节点各自用 writer：队列是线程安全的，数据不串。"""

    def make(tag):
        def node(state):
            w = get_stream_writer()
            for i in range(3):
                w({"node": tag, "i": i})
            return {}
        return node

    g = StateGraph(S)
    g.add_node("a", make("a"))
    g.add_node("b", make("b"))
    g.add_edge(START, "a")
    g.add_edge(START, "b")
    g.add_edge("a", END)
    g.add_edge("b", END)
    app = g.compile()

    chunks = list(app.stream({"n": 0}, stream_mode="custom"))
    assert len(chunks) == 6
    for tag in ("a", "b"):
        mine = [c["i"] for c in chunks if c["node"] == tag]
        assert mine == [0, 1, 2]           # 单个节点内部顺序保持
