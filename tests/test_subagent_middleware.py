"""SubAgentMiddleware：规格编译、task 分发、提示注入与图集成。"""

import pytest

from deep_agent import create_deep_agent
from deep_agent.middleware import (
    ModelRequest,
    SubAgentMiddleware,
    middleware,
)


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


class CompiledRunnable:
    def __init__(self, messages=None):
        self.messages = messages if messages is not None else [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": [_text("complete report")]},
            {"role": "assistant", "content": []},
        ]
        self.calls = []

    def invoke(self, state, config):
        self.calls.append((state, config))
        return {"messages": self.messages}


def _compiled(runnable):
    return {"name": "compiled", "description": "Compiled test agent",
            "runnable": runnable}


def test_compiled_subagent_tool_isolates_input_and_returns_last_nonempty_ai_text():
    runnable = CompiledRunnable()
    mw = SubAgentMiddleware(subagents=[_compiled(runnable)])

    task = mw.tools[0]
    schema = task.schema()
    assert schema["name"] == "task"
    assert schema["input_schema"]["required"] == ["description"]
    assert "compiled: Compiled test agent" in schema["description"]

    result = task.invoke({"description": "full delegated context",
                          "subagent_type": "compiled"})

    assert result == "complete report"
    assert len(runnable.calls) == 1
    state, config = runnable.calls[0]
    assert [message["role"] for message in state["messages"]] == ["user"]
    assert state["messages"][0]["content"] == "full delegated context"
    assert config["thread_id"].startswith("sub-")
    # 空的尾部 assistant 不会覆盖上一条真正的最终报告。
    assert "complete report" not in state["messages"][0]["content"]

    unknown = task.invoke({"description": "x", "subagent_type": "missing"})
    assert "未知子代理" in unknown and "compiled" in unknown


def test_subagent_middleware_custom_description_and_system_prompt():
    mw = SubAgentMiddleware(
        subagents=[_compiled(CompiledRunnable())],
        system_prompt="Delegate self-contained research.",
        task_description="CUSTOM TASK\n{available_agents}",
    )
    assert mw.subagent_names == {"compiled"}
    assert mw.tools[0].description.startswith("CUSTOM TASK\n- compiled:")

    request = ModelRequest(model="m", messages=[], system="BASE")
    modified = mw.wrap_model_call(request, lambda req: req)
    assert modified.system.startswith("BASE\n\nDelegate self-contained research.")
    assert "Available subagent types:" in modified.system
    assert "- compiled: Compiled test agent" in modified.system
    assert request.system == "BASE"             # 不原地修改请求


def test_subagent_specs_validate_before_compilation():
    with pytest.raises(ValueError, match="At least one"):
        SubAgentMiddleware(subagents=[])
    with pytest.raises(ValueError, match="Duplicate subagent name"):
        SubAgentMiddleware(subagents=[
            {"name": "same", "description": "a"},
            {"name": "same", "description": "b"},
        ])
    with pytest.raises(ValueError, match="must specify 'model'"):
        SubAgentMiddleware(subagents=[
            {"name": "raw", "description": "raw", "tools": []},
        ])
    with pytest.raises(ValueError, match="must specify 'tools'"):
        SubAgentMiddleware(subagents=[
            {"name": "raw", "description": "raw", "model": FakeModel([])},
        ])
    with pytest.raises(ValueError, match="only isolated"):
        SubAgentMiddleware(subagents=[{
            **_compiled(CompiledRunnable()), "mode": "fork"}])
    with pytest.raises(ValueError, match="nested interrupt resume"):
        SubAgentMiddleware(subagents=[{
            "name": "approval", "description": "unsupported",
            "interrupt_on": {"write_file": True},
        }])


def test_compiled_subagent_requires_messages_and_handles_empty_report():
    class MissingMessages:
        def invoke(self, state, config):
            return {}

    task = SubAgentMiddleware(subagents=[
        _compiled(MissingMessages()),
    ]).tools[0]
    with pytest.raises(ValueError, match="containing a 'messages' key"):
        task.invoke({"description": "x", "subagent_type": "compiled"})

    empty = SubAgentMiddleware(subagents=[
        _compiled(CompiledRunnable(messages=[])),
    ]).tools[0]
    assert empty.invoke({
        "description": "x", "subagent_type": "compiled"}) == "（子代理没有产出文本）"


def test_declarative_subagent_uses_own_model_tools_prompt_and_middleware():
    events = []

    @middleware
    class ChildWatch:
        def before_agent(self, state):
            events.append("before")

        def after_agent(self, state):
            events.append("after")

    def parent_only() -> str:
        """Parent-only tool."""
        return "parent"

    def child_only() -> str:
        """Child-only tool."""
        return "child"

    parent = FakeModel([
        [_call("delegate", "task", {
            "description": "research with full delegated context",
            "subagent_type": "researcher",
        })],
        [_text("parent done")],
    ])
    child = FakeModel([[_text("research report")]])
    app = create_deep_agent(
        parent,
        tools=[parent_only],
        subagents=[{
            "name": "researcher",
            "description": "Research specialist",
            "model": child,
            "tools": [child_only],
            "system_prompt": "RESEARCHER PROMPT",
            "middleware": [ChildWatch],
        }],
    )

    out = app.invoke(
        {"messages": [{"id": "h", "role": "user", "content": "start"}]},
        config={"thread_id": "declarative-subagent"})

    tool_result = out["messages"][2]["content"][0]
    assert tool_result["content"] == "research report"
    assert out["messages"][-1]["content"][0]["text"] == "parent done"
    assert events == ["before", "after"]

    request = child.requests[0]
    assert request.messages == [{
        "role": "user", "content": "research with full delegated context"}]
    assert request.system.startswith("RESEARCHER PROMPT")
    names = {item["name"] for item in request.tools}
    assert "child_only" in names and "write_todos" in names
    assert "parent_only" not in names
    assert "task" not in names                 # 默认不能递归派发

    parent_task = next(item for item in parent.requests[0].tools
                       if item["name"] == "task")
    assert "- researcher: Research specialist" in parent_task["description"]


def test_custom_subagent_middleware_replaces_default_in_real_graph():
    runnable = CompiledRunnable()
    custom = SubAgentMiddleware(subagents=[_compiled(runnable)])
    parent = FakeModel([
        [_call("delegate", "task", {
            "description": "use compiled", "subagent_type": "compiled"})],
        [_text("done")],
    ])
    app = create_deep_agent(parent, middleware=[custom])

    out = app.invoke(
        {"messages": [{"id": "h", "role": "user", "content": "start"}]},
        config={"thread_id": "custom-subagent-middleware"})

    assert out["messages"][2]["content"][0]["content"] == "complete report"
    assert len(runnable.calls) == 1
    task_schema = next(item for item in parent.requests[0].tools
                       if item["name"] == "task")
    assert "- compiled: Compiled test agent" in task_schema["description"]
    assert "- general-purpose:" not in task_schema["description"]


def test_explicit_general_purpose_replaces_default_without_duplicate_listing():
    parent = FakeModel([
        [_call("delegate", "task", {
            "description": "handle this", "subagent_type": "general-purpose"})],
        [_text("done")],
    ])
    child = FakeModel([[_text("custom general report")]])
    app = create_deep_agent(parent, subagents=[{
        "name": "general-purpose",
        "description": "Custom default override",
        "model": child,
        "tools": [],
        "system_prompt": "CUSTOM GENERAL PROMPT",
    }])

    out = app.invoke(
        {"messages": [{"id": "h", "role": "user", "content": "start"}]},
        config={"thread_id": "general-purpose-override"})

    assert out["messages"][2]["content"][0]["content"] == "custom general report"
    assert child.requests[0].system.startswith("CUSTOM GENERAL PROMPT")
    task_schema = next(item for item in parent.requests[0].tools
                       if item["name"] == "task")
    assert task_schema["description"].count("- general-purpose:") == 1
    assert "Custom default override" in task_schema["description"]
