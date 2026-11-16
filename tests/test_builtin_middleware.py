"""六个内置中间件：model_call_limit / tool_call_limit / model_retry /
tool_retry / model_fallback / human_in_the_loop。"""

from typing import Annotated

import pytest

from minigraph.builder import StateGraph
from minigraph.checkpoint import InMemorySaver
from minigraph.constants import END, START
from minigraph.context import GraphInterrupt
from deep_agent.middleware import (
    HumanInTheLoopMiddleware,
    MiddlewareStack,
    ModelCallLimitExceededError,
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    ModelRequest,
    ModelRetryMiddleware,
    ToolCallLimitExceededError,
    ToolCallLimitMiddleware,
    ToolRequest,
    ToolRetryMiddleware,
    agent_state_schema,
    merge_state_schemas,
    wire_middleware_graph,
)
from minigraph.state import DeltaChannel, State, add_messages


def _req(model="m"):
    return ModelRequest(model=model, messages=[{"role": "user", "content": "hi"}])


def _tool_req(name="t", args=None, call_id="1"):
    return ToolRequest({"id": call_id, "name": name, "input": args or {}})


class _AgentState(State):
    """HITL 测试用的 agent 状态：消息 + 待执行的人工决定。"""

    messages: Annotated[list, DeltaChannel(add_messages, list)]
    hitl_decisions: dict


# ---------------------------------------------------------------------------
# model_call_limit
# ---------------------------------------------------------------------------

def test_model_call_limit_requires_a_limit():
    with pytest.raises(ValueError, match="至少要指定一个上限"):
        ModelCallLimitMiddleware()
    with pytest.raises(ValueError, match="exit_behavior"):
        ModelCallLimitMiddleware(run_limit=1, exit_behavior="nope")


def test_model_call_limit_counts_and_jumps_to_end():
    mw = ModelCallLimitMiddleware(run_limit=2, exit_behavior="end")

    # 第一次：没超限，继续
    assert mw.before_model({"run_model_call_count": 0}) is None
    # after_model 计数 +1
    assert mw.after_model({"thread_model_call_count": 0,
                           "run_model_call_count": 0}) == {
        "thread_model_call_count": 1, "run_model_call_count": 1}

    # 第二次：1 < 2，继续
    assert mw.before_model({"run_model_call_count": 1}) is None
    assert mw.after_model({"thread_model_call_count": 1,
                           "run_model_call_count": 1}) == {
        "thread_model_call_count": 2, "run_model_call_count": 2}

    # 第三次：2 >= 2，跳到 end
    out = mw.before_model({"run_model_call_count": 2})
    assert out["jump_to"] == "end"
    assert "run limit (2/2)" in out["messages"][0]["content"]


def test_model_call_limit_error_behavior_raises():
    mw = ModelCallLimitMiddleware(thread_limit=1, exit_behavior="error")
    with pytest.raises(ModelCallLimitExceededError) as ei:
        mw.before_model({"thread_model_call_count": 1})
    assert ei.value.thread_limit == 1
    assert ei.value.thread_count == 1


def test_model_call_limit_declares_its_state_fields():
    mw = ModelCallLimitMiddleware(run_limit=1)
    assert "thread_model_call_count" in mw.state_schema.reducers
    assert "run_model_call_count" in mw.state_schema.reducers


# ---------------------------------------------------------------------------
# tool_call_limit
# ---------------------------------------------------------------------------

def _assistant_with_calls(*names):
    return {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": f"c{i}", "name": n, "input": {}}
                    for i, n in enumerate(names)],
    }


def test_tool_call_limit_requires_a_limit():
    with pytest.raises(ValueError, match="至少要指定一个上限"):
        ToolCallLimitMiddleware()
    with pytest.raises(ValueError, match="exit_behavior"):
        ToolCallLimitMiddleware(run_limit=1, exit_behavior="nope")
    with pytest.raises(ValueError, match="不能大于"):
        ToolCallLimitMiddleware(thread_limit=1, run_limit=5)


def test_tool_call_limit_continue_blocks_only_the_exceeded_call():
    mw = ToolCallLimitMiddleware(run_limit=1, exit_behavior="continue")
    state = {"messages": [_assistant_with_calls("a", "b")],
             "run_tool_call_count": {}}

    out = mw.after_model(state)
    # 第一个放行，第二个超限被拦；消息保持原样，由工具节点统一补结果。
    assert out["run_tool_call_count"] == {"__all__": 2}
    assert out["tool_call_limit_blocked"] == {mw.name: ["c1"]}
    assert "messages" not in out

    ran = []
    tool_state = {**state, **out}

    def run(request):
        ran.append(request.tool_call["id"])
        return "执行成功"

    allowed = ToolRequest(state["messages"][0]["content"][0], state=tool_state)
    blocked = ToolRequest(state["messages"][0]["content"][1], state=tool_state)
    assert mw.wrap_tool_call(allowed, run) == "执行成功"
    assert "Do not make additional tool calls" in mw.wrap_tool_call(blocked, run)
    assert ran == ["c0"]


def test_tool_call_limit_continue_under_limit_just_counts():
    mw = ToolCallLimitMiddleware(run_limit=5, exit_behavior="continue")
    state = {"messages": [_assistant_with_calls("a")], "run_tool_call_count": {}}
    out = mw.after_model(state)
    assert out["run_tool_call_count"] == {"__all__": 1}
    assert "messages" not in out


def test_tool_call_limit_scoped_to_one_tool():
    mw = ToolCallLimitMiddleware(tool_name="search", run_limit=1,
                                 exit_behavior="continue")
    # 不归它管的工具不计数
    state = {"messages": [_assistant_with_calls("other")],
             "run_tool_call_count": {}}
    out = mw.after_model(state)
    assert out is None


def test_tool_call_limit_error_behavior_raises():
    mw = ToolCallLimitMiddleware(run_limit=1, exit_behavior="error")
    state = {"messages": [_assistant_with_calls("a", "b")],
             "run_tool_call_count": {}}
    with pytest.raises(ToolCallLimitExceededError):
        mw.after_model(state)


def test_tool_call_limit_end_jumps_and_answers_pending_calls():
    mw = ToolCallLimitMiddleware(run_limit=1, exit_behavior="end")
    state = {"messages": [_assistant_with_calls("a", "b")],
             "run_tool_call_count": {}}
    out = mw.after_model(state)
    assert out["jump_to"] == "end"
    blocks = out["messages"][0]["content"]
    # 超限的那个 + 同批次剩下的那个，各有一条 tool_result
    assert len(blocks) == 2
    assert out["messages"][1]["role"] == "assistant"


def test_tool_call_limit_name_includes_tool():
    mw = ToolCallLimitMiddleware(tool_name="search", run_limit=1)
    assert mw.name == "ToolCallLimitMiddleware[search]"


# ---------------------------------------------------------------------------
# model_retry / tool_retry
# ---------------------------------------------------------------------------

def test_model_retry_retries_until_success():
    mw = ModelRetryMiddleware(max_retries=2, initial_delay=0, jitter=False)
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("boom")
        return ["ok"]

    assert mw.wrap_model_call(_req(), handler) == ["ok"]
    assert len(calls) == 3


def test_model_retry_non_matching_exception_propagates():
    mw = ModelRetryMiddleware(max_retries=3, retry_on=(KeyError,),
                              initial_delay=0)

    def handler(request):
        raise ValueError("不该重试")

    with pytest.raises(ValueError):
        mw.wrap_model_call(_req(), handler)


def test_model_retry_on_failure_continue_returns_error_message():
    mw = ModelRetryMiddleware(max_retries=1, initial_delay=0, jitter=False)

    def handler(request):
        raise RuntimeError("boom")

    out = mw.wrap_model_call(_req(), handler)
    assert out[0]["type"] == "text"
    assert "boom" in out[0]["text"]


def test_model_retry_on_failure_error_reraises():
    mw = ModelRetryMiddleware(max_retries=1, on_failure="error",
                              initial_delay=0)

    def handler(request):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        mw.wrap_model_call(_req(), handler)


def test_model_retry_custom_on_failure_message():
    mw = ModelRetryMiddleware(max_retries=0, on_failure=lambda e: "自定义错误",
                              initial_delay=0)

    def handler(request):
        raise RuntimeError("boom")

    assert mw.wrap_model_call(_req(), handler) == [
        {"type": "text", "text": "自定义错误"}]


def test_model_retry_does_not_retry_interrupts():
    mw = ModelRetryMiddleware(max_retries=3, initial_delay=0)

    def handler(request):
        raise GraphInterrupt("pause")

    with pytest.raises(GraphInterrupt):
        mw.wrap_model_call(_req(), handler)


def test_model_retry_backoff_is_capped():
    from deep_agent.middleware._retry import calculate_delay
    d = calculate_delay(10, backoff_factor=2.0, initial_delay=1.0,
                        max_delay=8.0, jitter=False)
    assert d == 8.0


def test_tool_retry_retries_until_success():
    mw = ToolRetryMiddleware(max_retries=2, initial_delay=0, jitter=False)
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 2:
            raise RuntimeError("boom")
        return "ok"

    assert mw.wrap_tool_call(_tool_req(), handler) == "ok"
    assert len(calls) == 2


def test_tool_retry_exhausted_returns_error_text():
    mw = ToolRetryMiddleware(max_retries=1, initial_delay=0, jitter=False)

    def handler(request):
        raise RuntimeError("boom")

    out = mw.wrap_tool_call(_tool_req("search", call_id="c9"), handler)
    assert isinstance(out, str)
    assert "search" in out
    assert "boom" in out


def test_tool_retry_only_applies_to_selected_tools():
    mw = ToolRetryMiddleware(max_retries=3, tools=["search"],
                             initial_delay=0, jitter=False)
    calls = []

    def handler(request):
        calls.append(request.name)
        raise RuntimeError("boom")

    # 不在名单里的工具直接抛，不重试
    with pytest.raises(RuntimeError):
        mw.wrap_tool_call(_tool_req("other"), handler)
    assert calls == ["other"]


def test_tool_retry_does_not_retry_interrupts():
    mw = ToolRetryMiddleware(max_retries=3, initial_delay=0)

    def handler(request):
        raise GraphInterrupt("pause")

    with pytest.raises(GraphInterrupt):
        mw.wrap_tool_call(_tool_req(), handler)


def test_retry_validates_parameters():
    with pytest.raises(ValueError, match="max_retries"):
        ModelRetryMiddleware(max_retries=-1)
    with pytest.raises(ValueError, match="initial_delay"):
        ModelRetryMiddleware(initial_delay=-1)
    with pytest.raises(ValueError, match="backoff_factor"):
        ToolRetryMiddleware(backoff_factor=-1)


# ---------------------------------------------------------------------------
# model_fallback
# ---------------------------------------------------------------------------

def test_model_fallback_tries_next_model_on_error():
    mw = ModelFallbackMiddleware("backup-1", "backup-2")
    seen = []

    def handler(request):
        seen.append(request.model)
        if len(seen) < 3:
            raise RuntimeError("boom")
        return ["ok"]

    assert mw.wrap_model_call(_req("primary"), handler) == ["ok"]
    assert seen == ["primary", "backup-1", "backup-2"]


def test_model_fallback_reraises_when_all_fail():
    mw = ModelFallbackMiddleware("backup-1")

    def handler(request):
        raise RuntimeError("all down")

    with pytest.raises(RuntimeError, match="all down"):
        mw.wrap_model_call(_req("primary"), handler)


def test_model_fallback_does_not_swallow_interrupts():
    mw = ModelFallbackMiddleware("backup-1")

    def handler(request):
        raise GraphInterrupt("pause")

    with pytest.raises(GraphInterrupt):
        mw.wrap_model_call(_req("primary"), handler)


# ---------------------------------------------------------------------------
# human_in_the_loop
# ---------------------------------------------------------------------------

def test_hitl_rejects_bad_config():
    with pytest.raises(ValueError, match="interrupt_mode"):
        HumanInTheLoopMiddleware(interrupt_on={}, interrupt_mode="nope")
    with pytest.raises(ValueError, match="allowed_decisions"):
        HumanInTheLoopMiddleware(interrupt_on={"t": {"allowed_decisions": []}})


def _hitl_app(mw, ran, calls=None):
    """搭真图：第一轮请求工具，工具结果回来后第二轮输出最终答复。"""
    stack = MiddlewareStack([mw])
    calls = calls or [{"id": "c1", "name": "search", "input": {"q": "x"}}]

    def model(state):
        if any(m.get("role") == "assistant" for m in state["messages"]):
            return {"messages": [{
                "id": "ai-final", "role": "assistant", "content": "完成"}]}
        return {"messages": [{
            "id": "ai-1", "role": "assistant",
            "content": [{"type": "tool_use", **c} for c in calls],
        }]}

    def tools(state):
        last = state["messages"][-1]
        tool_calls = [b for b in (last.get("content") or [])
                      if isinstance(b, dict) and b.get("type") == "tool_use"]
        results = []
        for c in tool_calls:
            def run(req):
                ran.append((req.name, req.args))
                return "结果"

            out = stack.tool_call(ToolRequest(c, state=state), run)
            results.append({"type": "tool_result",
                            "tool_use_id": c["id"], "content": out})
        return {"messages": [{"role": "user", "content": results}]}

    def route(state):
        last = state["messages"][-1]
        if last.get("role") == "assistant" and any(
                isinstance(b, dict) and b.get("type") == "tool_use"
                for b in (last.get("content") or [])):
            return "tools"
        return END

    g = StateGraph(_AgentState)
    g.add_node("model", model)
    g.add_node("tools", tools)
    wire_middleware_graph(g, stack, route=route)
    return g.compile(checkpointer=InMemorySaver())


def _tool_results(messages):
    return [b for m in messages if m.get("role") == "user"
            for b in (m.get("content") or []) if isinstance(b, dict)
            and b.get("type") == "tool_result"]


def test_hitl_batched_interrupts_and_resumes_with_approve():
    """
    端到端：跑一张真图，人工批准后继续。

    这测的是完整链路 —— interrupt() 挂起、resume 值回来、
    工具真的执行了 —— 而不是单个方法。
    """
    mw = HumanInTheLoopMiddleware(interrupt_on={"search": True})
    ran = []
    app = _hitl_app(mw, ran)

    cfg = {"thread_id": "t1"}
    app.invoke({"messages": [{"role": "user", "content": "搜一下"}]}, cfg)
    # 挂起了：HITL 节点还没跑完，工具没执行
    assert ran == []

    # 人工批准
    app.invoke(None, {"thread_id": "t1",
                      "resume": {"decisions": [{"type": "approve"}]}})
    assert ran == [("search", {"q": "x"})]
    assert app.interrupted is None


def test_hitl_batched_reject_does_not_run_the_tool():
    mw = HumanInTheLoopMiddleware(interrupt_on={"search": True})
    ran = []
    app = _hitl_app(mw, ran)

    cfg = {"thread_id": "t2"}
    app.invoke({"messages": [{"role": "user", "content": "搜一下"}]}, cfg)
    app.invoke(None, {"thread_id": "t2",
                      "resume": {"decisions": [{"type": "reject",
                                                "message": "不行"}]}})
    assert ran == []
    assert app.interrupted is None


def test_hitl_interrupted_payload_lists_the_gated_call():
    """挂起时交给调用方的清单长什么样。"""
    mw = HumanInTheLoopMiddleware(interrupt_on={"search": True})
    state = {"messages": [{
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "c1", "name": "search",
                     "input": {"q": "x"}}],
    }]}

    with pytest.raises(RuntimeError, match="只能在图节点内"):
        # 不在节点里调 interrupt() -> 明确的错误，而不是静默挂起
        mw.after_model(state)


def test_hitl_edit_replaces_the_tool_call():
    """edit 决定：真跑的是人工改过的调用，模型看到的结果里带说明。"""
    mw = HumanInTheLoopMiddleware(interrupt_on={"search": True})
    ran = []
    app = _hitl_app(mw, ran)

    cfg = {"thread_id": "e1"}
    app.invoke({"messages": [{"role": "user", "content": "go"}]}, cfg)
    app.invoke(None, {"thread_id": "e1", "resume": {"decisions": [
        {"type": "edit",
         "edited_action": {"name": "search", "args": {"q": "改过的"}}}]}})

    # 真跑的是人工改过的参数，结果前面带"人工改过"的说明。
    assert ran == [("search", {"q": "改过的"})]
    results = _tool_results(app.get_state(cfg).values["messages"])
    assert "human reviewer replaced" in results[0]["content"]
    assert "改过的" in results[0]["content"]
    assert app.interrupted is None


def test_hitl_reject_preserves_protocol_and_returns_a_tool_result():
    """拒绝不改写原始 tool_use；工具不执行，结果由工具节点补回。"""
    mw = HumanInTheLoopMiddleware(interrupt_on={"search": True})
    ran = []
    app = _hitl_app(mw, ran)

    cfg = {"thread_id": "r1"}
    app.invoke({"messages": [{"role": "user", "content": "go"}]}, cfg)
    app.invoke(None, {"thread_id": "r1", "resume": {"decisions": [
        {"type": "reject", "message": "不行"}]}})

    msgs = app.get_state(cfg).values["messages"]
    assistants = [m for m in msgs if m.get("role") == "assistant"]
    assert len(assistants) == 2       # 请求工具的回复 + 最终答复
    assert assistants[0]["content"][0]["id"] == "c1"
    results = _tool_results(msgs)
    assert len(results) == 1
    assert results[0]["tool_use_id"] == "c1"
    assert "不行" in results[0]["content"]
    assert ran == []
    assert app.interrupted is None


def test_hitl_batched_reject_does_not_block_an_ungated_call():
    mw = HumanInTheLoopMiddleware(interrupt_on={"search": True})
    ran = []
    app = _hitl_app(mw, ran, calls=[
        {"id": "c1", "name": "search", "input": {"q": "x"}},
        {"id": "c2", "name": "other", "input": {}},
    ])

    cfg = {"thread_id": "mixed-hitl"}
    app.invoke({"messages": [{"role": "user", "content": "go"}]}, cfg)
    app.invoke(None, {"thread_id": "mixed-hitl", "resume": {"decisions": [
        {"type": "reject", "message": "不行"}]}})

    assert ran == [("other", {})]
    results = _tool_results(app.get_state(cfg).values["messages"])
    assert [r["tool_use_id"] for r in results] == ["c1", "c2"]
    assert "不行" in results[0]["content"]
    assert results[1]["content"] == "结果"
    assert app.interrupted is None


def test_hitl_per_call_mode_asks_once_per_tool_call():
    """per_call 模式：逐个调用问，答复是单个决定。"""
    mw = HumanInTheLoopMiddleware(interrupt_on={"search": True},
                                  interrupt_mode="per_call")
    ran = []
    app = _hitl_app(mw, ran)

    cfg = {"thread_id": "p1"}
    app.invoke({"messages": [{"role": "user", "content": "go"}]}, cfg)
    # 挂起值是单个调用的审批请求
    payload = app.interrupted
    assert payload["type"] == "tool_approval"
    assert payload["name"] == "search"
    assert payload["tool_call_id"] == "c1"

    app.invoke(None, {"thread_id": "p1", "resume": {"type": "approve"}})
    assert len(ran) == 1


def test_hitl_per_call_mode_defers_to_wrap_tool_call():
    mw = HumanInTheLoopMiddleware(interrupt_on={"search": True},
                                  interrupt_mode="per_call")
    # per_call 模式 after_model 不挂起
    assert mw.after_model({"messages": []}) is None


def test_hitl_process_decision_reject_makes_error_result():
    mw = HumanInTheLoopMiddleware(interrupt_on={"search": True})
    config = mw.interrupt_on["search"]
    call = {"id": "c1", "name": "search", "input": {}}
    _, msg = mw._process_decision({"type": "reject", "message": "不行"}, call, config)
    assert msg["is_error"] is True
    assert "不行" in msg["content"]


def test_hitl_process_decision_respond_is_not_error():
    mw = HumanInTheLoopMiddleware(interrupt_on={"search": True})
    config = mw.interrupt_on["search"]
    call = {"id": "c1", "name": "search", "input": {}}
    _, msg = mw._process_decision({"type": "respond", "message": "42"}, call, config)
    assert msg["is_error"] is False
    assert msg["content"] == "42"


def test_hitl_process_decision_rejects_unallowed_type():
    mw = HumanInTheLoopMiddleware(
        interrupt_on={"search": {"allowed_decisions": ["approve"]}})
    config = mw.interrupt_on["search"]
    call = {"id": "c1", "name": "search", "input": {}}
    with pytest.raises(ValueError, match="不在工具"):
        mw._process_decision({"type": "edit", "edited_action": {}}, call, config)


def test_hitl_tool_not_in_interrupt_on_is_auto_approved():
    mw = HumanInTheLoopMiddleware(interrupt_on={"search": True})
    # 不在名单里的工具：after_model 直接返回 None（没有待审项）
    assert mw.after_model({"messages": [{
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "c1", "name": "other", "input": {}}],
    }]}) is None


# ---------------------------------------------------------------------------
# state schema 合并
# ---------------------------------------------------------------------------

def test_merge_state_schemas_unions_fields():
    class Base(State):
        messages: Annotated[list, DeltaChannel(add_messages, list)]

    merged = merge_state_schemas(Base, [ModelCallLimitMiddleware(run_limit=1).state_schema])
    assert "messages" in merged.reducers
    assert "thread_model_call_count" in merged.reducers
    assert "run_model_call_count" in merged.reducers


def test_merge_state_schemas_base_wins_on_conflict():
    class Base(State):
        count: int

    class Extra(State):
        count: int
        other: int

    merged = merge_state_schemas(Base, [Extra])
    assert set(merged.reducers) == {"count", "other"}


def test_agent_state_schema_adds_jump_to_only_when_needed():
    class Base(State):
        messages: Annotated[list, DeltaChannel(add_messages, list)]

    limited = MiddlewareStack([ModelCallLimitMiddleware(run_limit=1)])
    plain = MiddlewareStack([ModelRetryMiddleware()])

    assert "jump_to" in agent_state_schema(Base, limited).reducers
    assert "jump_to" not in agent_state_schema(Base, plain).reducers


def test_model_call_limit_stops_a_tool_loop_at_the_run_limit():
    """完整图：前两次调用放行，第三次在 before_model 被 jump_to=end 拦下。"""
    class Base(State):
        messages: Annotated[list, DeltaChannel(add_messages, list)]

    mw = ModelCallLimitMiddleware(run_limit=2, exit_behavior="end")
    stack = MiddlewareStack([mw])
    model_calls = []

    def model(state):
        model_calls.append(1)
        n = len(model_calls)
        return {"messages": [{
            "id": f"ai-{n}", "role": "assistant",
            "content": [{"type": "tool_use", "id": f"c{n}",
                         "name": "search", "input": {}}],
        }]}

    def tools(state):
        return {"messages": [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "c1", "content": "结果"},
        ]}]}

    def route(state):
        last = state["messages"][-1]
        if last.get("role") == "assistant":
            return "tools"
        return END

    g = StateGraph(agent_state_schema(Base, stack))
    g.add_node("model", model)
    g.add_node("tools", tools)
    wire_middleware_graph(g, stack, route=route)
    app = g.compile(checkpointer=InMemorySaver())

    out = app.invoke({"messages": [{"role": "user", "content": "go"}]},
                     {"thread_id": "limit-1"})

    assert len(model_calls) == 2
    assert out["run_model_call_count"] == 2
    assert out["messages"][-1]["role"] == "assistant"
    assert "Model call limits exceeded" in out["messages"][-1]["content"]


def test_model_call_limit_resets_run_count_between_invokes():
    """checkpoint 保留 thread 计数，但每次 invoke 都有新的 run 配额。"""
    class Base(State):
        messages: Annotated[list, DeltaChannel(add_messages, list)]

    mw = ModelCallLimitMiddleware(run_limit=1, thread_limit=2)
    stack = MiddlewareStack([mw])
    model_calls = []

    def model(state):
        model_calls.append(1)
        return {"messages": [{
            "id": f"ai-{len(model_calls)}", "role": "assistant",
            "content": "完成",
        }]}

    g = StateGraph(agent_state_schema(Base, stack))
    g.add_node("model", model)
    wire_middleware_graph(g, stack, route=lambda state: END)
    app = g.compile(checkpointer=InMemorySaver())
    cfg = {"thread_id": "limit-across-runs"}

    first = app.invoke({"messages": [{"role": "user", "content": "一"}]}, cfg)
    assert first["thread_model_call_count"] == 1
    assert first["run_model_call_count"] == 1

    second = app.invoke({"messages": [{"role": "user", "content": "二"}]}, cfg)
    assert len(model_calls) == 2       # 没被上一次 invoke 的 run=1 挡住
    assert second["thread_model_call_count"] == 2
    assert second["run_model_call_count"] == 1

    third = app.invoke({"messages": [{"role": "user", "content": "三"}]}, cfg)
    assert len(model_calls) == 2       # thread=2 仍跨 invoke 生效
    assert third["run_model_call_count"] == 0
    assert "thread limit (2/2)" in third["messages"][-1]["content"]


def _tool_limit_app(mw, model, executed):
    """工具节点真实经过 MiddlewareStack，便于验证 handler 有没有执行。"""
    class Base(State):
        messages: Annotated[list, DeltaChannel(add_messages, list)]

    stack = MiddlewareStack([mw])

    def tools(state):
        last = state["messages"][-1]
        calls = [b for b in (last.get("content") or [])
                 if isinstance(b, dict) and b.get("type") == "tool_use"]
        results = []
        for call in calls:
            def run(request):
                executed.append(request.tool_call["id"])
                return "执行成功"

            content = stack.tool_call(ToolRequest(call, state=state), run)
            results.append({"type": "tool_result",
                            "tool_use_id": call["id"], "content": content})
        return {"messages": [{"role": "user", "content": results}]}

    def route(state):
        last = state["messages"][-1]
        if last.get("role") == "assistant" and any(
                isinstance(b, dict) and b.get("type") == "tool_use"
                for b in (last.get("content") or [])):
            return "tools"
        return END

    g = StateGraph(agent_state_schema(Base, stack))
    g.add_node("model", model)
    g.add_node("tools", tools)
    wire_middleware_graph(g, stack, route=route)
    return g.compile(checkpointer=InMemorySaver())


def test_tool_call_limit_continue_really_blocks_only_the_blocked_tool():
    """同批次 allowed/blocked：前者执行，后者只得到错误 tool_result。"""
    mw = ToolCallLimitMiddleware(run_limit=1, exit_behavior="continue")
    executed = []
    model_calls = []

    def model(state):
        model_calls.append(1)
        if len(model_calls) == 1:
            return {"messages": [{
                "id": "ai-calls", "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "allowed", "name": "a", "input": {}},
                    {"type": "tool_use", "id": "blocked", "name": "b", "input": {}},
                ],
            }]}
        return {"messages": [{
            "id": "ai-final", "role": "assistant", "content": "完成"}]}

    app = _tool_limit_app(mw, model, executed)
    out = app.invoke({"messages": [{"role": "user", "content": "go"}]},
                     {"thread_id": "mixed-tool-limit"})

    assert executed == ["allowed"]
    assert out["thread_tool_call_count"] == {"__all__": 1}
    assert out["run_tool_call_count"] == {"__all__": 2}
    results = _tool_results(out["messages"])
    assert [r["tool_use_id"] for r in results] == ["allowed", "blocked"]
    assert results[0]["content"] == "执行成功"
    assert "Tool call limit exceeded" in results[1]["content"]


def test_tool_call_limit_resets_run_count_between_invokes():
    """第二次 invoke 应重新获得一次 run 配额，同时保留 thread 累计。"""
    mw = ToolCallLimitMiddleware(run_limit=1, thread_limit=2)
    executed = []
    model_calls = []

    def model(state):
        model_calls.append(1)
        n = len(model_calls)
        if n % 2:                    # 每次 invoke 的第一轮：请求一个工具
            return {"messages": [{
                "id": f"ai-{n}", "role": "assistant",
                "content": [{"type": "tool_use", "id": f"c{n}",
                             "name": "search", "input": {}}],
            }]}
        return {"messages": [{       # 每次 invoke 的第二轮：收尾
            "id": f"ai-{n}", "role": "assistant", "content": "完成"}]}

    app = _tool_limit_app(mw, model, executed)
    cfg = {"thread_id": "tool-limit-across-runs"}

    first = app.invoke({"messages": [{"role": "user", "content": "一"}]}, cfg)
    assert first["thread_tool_call_count"] == {"__all__": 1}
    assert first["run_tool_call_count"] == {"__all__": 1}

    second = app.invoke({"messages": [{"role": "user", "content": "二"}]}, cfg)
    assert executed == ["c1", "c3"]
    assert second["thread_tool_call_count"] == {"__all__": 2}
    assert second["run_tool_call_count"] == {"__all__": 1}
