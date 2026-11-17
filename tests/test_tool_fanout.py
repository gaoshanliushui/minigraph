"""工具调用并行（对齐 LangGraph ToolNode._func 的线程池执行与按序合并）。"""

import threading
import time

from deep_agent import create_deep_agent
from minigraph.tools import tool


def _text(text):
    return {"type": "text", "text": text}


def _call(call_id, name, args):
    return {"type": "tool_use", "id": call_id, "name": name, "input": args}


class FakeModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        assert self.responses, "fake model 脚本耗尽"
        return self.responses.pop(0)


def _results(out):
    res = {}
    for message in out["messages"]:
        content = message["content"]
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    res[block["tool_use_id"]] = block["content"]
    return res


def _tool_messages(out):
    """所有装着 tool_result 的 user 消息。"""
    return [m for m in out["messages"]
            if isinstance(m["content"], list)
            and any(isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in m["content"])]


def test_parallel_tool_calls_run_concurrently_and_keep_call_order():
    """N 个工具调用在线程池里并行；结果按原调用顺序拼成一条消息。"""
    barrier = threading.Barrier(3, timeout=5)
    threads = []

    @tool
    def slow(name: str) -> str:
        """Barrier 同步：三个调用必须同时在跑，否则测试超时。"""
        threads.append(threading.current_thread().name)
        barrier.wait()          # 串行执行时这里会 timeout
        time.sleep(0.02)
        return f"done:{name}"

    model = FakeModel([
        [_call("c1", "slow", {"name": "a"}),
         _call("c2", "slow", {"name": "b"}),
         _call("c3", "slow", {"name": "c"})],
        [_text("全部完成")],
    ])
    app = create_deep_agent(model, tools=[slow])
    out = app.invoke(
        {"messages": [{"id": "h1", "role": "user", "content": "并行"}]},
        config={"thread_id": "parallel-tools"})

    # 三个调用真的并发跑（Barrier 只在同时到达时才能通过）
    assert len(set(threads)) == 3

    # 结果按**原始调用顺序**排在同一条 user 消息里
    tool_messages = _tool_messages(out)
    assert len(tool_messages) == 1
    blocks = tool_messages[0]["content"]
    assert [b["tool_use_id"] for b in blocks] == ["c1", "c2", "c3"]
    assert [b["content"] for b in blocks] == [
        "done:a", "done:b", "done:c"]

    # 模型看到的完整历史符合 Anthropic 协议：
    # assistant(tool_use x3) -> user(tool_result x3)
    roles = [m["role"] for m in out["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert out["messages"][-1]["content"][0]["text"] == "全部完成"


def test_tool_results_are_assembled_into_one_user_message():
    """并行结果必须合并成一条 user 消息，而不是 N 条。"""
    @tool
    def echo(text: str) -> str:
        """回显文本。"""
        return text

    model = FakeModel([
        [_call("x", "echo", {"text": "one"}),
         _call("y", "echo", {"text": "two"})],
        [_text("done")],
    ])
    app = create_deep_agent(model, tools=[echo])
    out = app.invoke(
        {"messages": [{"id": "h1", "role": "user", "content": "go"}]},
        config={"thread_id": "one-tool-message"})

    assert len(_tool_messages(out)) == 1
    assert _results(out) == {"x": "one", "y": "two"}


def test_single_tool_call_still_works():
    @tool
    def add(a: int, b: int) -> int:
        """两数相加。"""
        return a + b

    model = FakeModel([
        [_call("1", "add", {"a": 2, "b": 3})],
        [_text("5")],
    ])
    app = create_deep_agent(model, tools=[add])
    out = app.invoke(
        {"messages": [{"id": "h1", "role": "user", "content": "加一下"}]},
        config={"thread_id": "single-tool"})

    assert _results(out)["1"] == "5"


def test_state_tools_write_concurrently_without_conflict():
    """并行 write_file 各自写不同路径：merge_dict 让它们安全合并。"""

    @tool
    def write(path: str, content: str) -> str:
        """写入文件。"""
        return f"wrote {path}"

    model = FakeModel([
        [_call("w1", "write", {"path": "/a.txt", "content": "A"}),
         _call("w2", "write", {"path": "/b.txt", "content": "B"})],
        [_text("done")],
    ])
    app = create_deep_agent(model, tools=[write])
    out = app.invoke(
        {"messages": [{"id": "h1", "role": "user", "content": "写"}]},
        config={"thread_id": "parallel-write"})

    assert _results(out) == {"w1": "wrote /a.txt", "w2": "wrote /b.txt"}


def test_builtin_state_tools_merge_under_parallel_dispatch():
    """内置的 write_file（写 state["files"]）并行时结果不丢。"""
    model = FakeModel([
        [_call("c1", "write_file", {"path": "/a.txt", "content": "A"}),
         _call("c2", "write_file", {"path": "/b.txt", "content": "B"})],
        [_text("done")],
    ])
    app = create_deep_agent(model)
    out = app.invoke(
        {"messages": [{"id": "h1", "role": "user", "content": "写"}]},
        config={"thread_id": "parallel-state-tool"})

    assert out["files"] == {"/a.txt": "A", "/b.txt": "B"}
    assert _results(out)["c1"] == "已写入 /a.txt（1 字符）"
    assert _results(out)["c2"] == "已写入 /b.txt（1 字符）"


def test_already_answered_tool_calls_are_not_dispatched_again():
    """历史里已有结果的调用不会被重复派发（对齐 pending_tool_calls 过滤）。"""
    calls = []

    @tool
    def ping() -> str:
        """只回一句话。"""
        calls.append(1)
        return "pong"

    model = FakeModel([
        [_call("t1", "ping", {})],
        [_text("done")],
    ])
    app = create_deep_agent(model, tools=[ping])
    out = app.invoke(
        {"messages": [{"id": "h1", "role": "user", "content": "ping"}]},
        config={"thread_id": "no-redi"})

    assert len(calls) == 1
    assert _results(out)["t1"] == "pong"


def test_tool_node_runs_once_per_call():
    """每个工具调用恰好执行一次（每次派发只调用一次）。"""
    counts = {}

    @tool
    def track(tag: str) -> str:
        """计数。"""
        counts[tag] = counts.get(tag, 0) + 1
        return tag

    model = FakeModel([
        [_call("a", "track", {"tag": "t"}),
         _call("b", "track", {"tag": "t"})],
        [_text("done")],
    ])
    app = create_deep_agent(model, tools=[track])
    app.invoke(
        {"messages": [{"id": "h1", "role": "user", "content": "go"}]},
        config={"thread_id": "count"})

    assert counts == {"t": 2}      # 两个调用各执行一次，没有重复执行


def test_hitl_batched_decisions_apply_to_parallel_tool_calls():
    """并行派发时，人工决定按 tool_call id 落到各自的调用上。"""
    from minigraph.checkpoint import InMemorySaver

    executed = []

    @tool
    def risky(tag: str) -> str:
        """需要审批。"""
        executed.append(tag)
        return f"ran:{tag}"

    model = FakeModel([
        [_call("r1", "risky", {"tag": "a"}),
         _call("r2", "risky", {"tag": "b"})],
        [_text("done")],
    ])
    app = create_deep_agent(
        model,
        tools=[risky],
        checkpointer=InMemorySaver(),
        interrupt_on={"risky": True},
    )
    cfg = {"thread_id": "hitl-parallel"}

    app.invoke({"messages": [{"id": "h1", "role": "user", "content": "go"}]},
               config=cfg)

    pending = app.interrupted
    assert pending is not None
    assert {a["name"] for a in pending["action_requests"]} == {"risky"}
    assert executed == []          # 审批前一个都没跑

    out = app.invoke(None, config={**cfg, "resume": {"decisions": [
        {"type": "approve"}, {"type": "reject", "message": "不批"}]}})

    assert sorted(executed) == ["a"]     # 只有被批准的那个真跑了
    results = _results(out)
    assert "ran:a" in results["r1"]
    assert "不批" in results["r2"]        # 被拒的也有结果，tool_use 不悬空
    assert len(_tool_messages(out)) == 1


def test_hitl_per_call_mode_with_parallel_calls_asks_one_at_a_time():
    """per_call 模式 + 并行派发：每个调用各自挂起，不会互相吞掉答复。"""
    from deep_agent.middleware import HumanInTheLoopMiddleware
    from minigraph.checkpoint import InMemorySaver

    executed = []

    @tool
    def risky(tag: str) -> str:
        """需要审批。"""
        executed.append(tag)
        return f"ran:{tag}"

    model = FakeModel([
        [_call("r1", "risky", {"tag": "a"}),
         _call("r2", "risky", {"tag": "b"})],
        [_text("done")],
    ])
    app = create_deep_agent(
        model,
        tools=[risky],
        middleware=[HumanInTheLoopMiddleware(
            interrupt_on={"risky": True}, interrupt_mode="per_call")],
        checkpointer=InMemorySaver(),
    )
    cfg = {"thread_id": "hitl-parallel-per-call"}

    app.invoke({"messages": [{"id": "h1", "role": "user", "content": "go"}]},
               config=cfg)

    # 挂起的是**单个**调用的审批请求（per_call 语义）
    payload = app.interrupted
    assert payload["type"] == "tool_approval"
    assert payload["name"] == "risky"
    assert payload["tool_call_id"] in {"r1", "r2"}
    assert executed == []

    # 批准这一个；另一个还会再挂起一次
    app.invoke(None, config={**cfg, "resume": {"type": "approve"}})
    second = app.interrupted
    assert second is not None
    assert second["type"] == "tool_approval"
    assert second["tool_call_id"] != payload["tool_call_id"]

    app.invoke(None, config={**cfg, "resume": {"type": "approve"}})
    assert app.interrupted is None
    assert sorted(executed) == ["a", "b"]


def test_hitl_per_call_reject_does_not_execute_the_tool():
    """per_call 模式 + 并行派发：被拒的那个不执行，另一个照常执行一次。"""
    from deep_agent.middleware import HumanInTheLoopMiddleware
    from minigraph.checkpoint import InMemorySaver

    executed = []

    @tool
    def risky(tag: str) -> str:
        """需要审批。"""
        executed.append(tag)
        return f"ran:{tag}"

    model = FakeModel([
        [_call("r1", "risky", {"tag": "a"}),
         _call("r2", "risky", {"tag": "b"})],
        [_text("done")],
    ])
    app = create_deep_agent(
        model,
        tools=[risky],
        middleware=[HumanInTheLoopMiddleware(
            interrupt_on={"risky": True}, interrupt_mode="per_call")],
        checkpointer=InMemorySaver(),
    )
    cfg = {"thread_id": "hitl-per-call-reject"}

    app.invoke({"messages": [{"id": "h1", "role": "user", "content": "go"}]},
               config=cfg)
    first = app.interrupted["tool_call_id"]

    # 先拒掉一个
    app.invoke(None, config={**cfg, "resume": {"type": "reject",
                                               "message": "不批"}})
    second = app.interrupted
    assert second is not None
    assert second["tool_call_id"] != first

    # 另一个批准
    out = app.invoke(None, config={**cfg, "resume": {"type": "approve"}})

    # 被拒的没执行，批准的执行了**一次**
    assert len(executed) == 1
    results = _results(out)
    rejected = [v for k, v in results.items() if k == first]
    assert rejected and "不批" in rejected[0]
    assert len(_tool_messages(out)) == 1


def test_interrupt_inside_parallel_tool_bubbles_and_resumes():
    """工具线程里的 interrupt 找回图任务上下文，也不会被异常兜底吞掉。"""
    from minigraph.checkpoint import InMemorySaver
    from minigraph.context import interrupt

    @tool
    def ask_human(question: str) -> str:
        """把问题交给人工，并返回人工答复。"""
        return interrupt({"type": "question", "question": question})

    @tool
    def echo(text: str) -> str:
        """回显文本。"""
        return text

    model = FakeModel([
        [_call("q", "ask_human", {"question": "怎么继续"}),
         _call("e", "echo", {"text": "parallel"})],
        [_text("done")],
    ])
    app = create_deep_agent(
        model, tools=[ask_human, echo], checkpointer=InMemorySaver())
    cfg = {"thread_id": "interrupt-inside-tool"}

    app.invoke({"messages": [{"id": "h1", "role": "user", "content": "go"}]},
               config=cfg)
    assert app.interrupted == {"type": "question", "question": "怎么继续"}

    out = app.invoke(None, config={**cfg, "resume": "人工答复"})
    assert app.interrupted is None
    assert _results(out) == {"q": "人工答复", "e": "parallel"}
