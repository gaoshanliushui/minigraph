"""中间件：@middleware / @wrap_model_call / @wrap_tool_call + MiddlewareStack。"""

import pytest

from deep_agent.middleware import (
    MiddlewareStack,
    ModelRequest,
    ToolRequest,
    add_hook_nodes,
    after_agent,
    after_model,
    before_agent,
    before_model,
    middleware,
    wrap_model_call,
    wrap_tool_call,
)


def _req():
    return ModelRequest(model="m", messages=[{"role": "user", "content": "hi"}])


def test_wrap_model_call_can_modify_request_and_response():
    seen = {}

    @wrap_model_call
    def tweak(request, handler):
        request.model = "m2"                    # 改请求
        result = handler(request)
        return result + ["extra"]               # 改响应

    def handler(request):
        seen["model"] = request.model
        return ["text"]

    out = MiddlewareStack([tweak]).model_call(_req(), handler)
    assert seen["model"] == "m2"
    assert out == ["text", "extra"]


def test_onion_order_first_is_outermost():
    events = []

    @middleware
    class A:
        def wrap_model_call(self, request, handler):
            events.append("A in")
            out = handler(request)
            events.append("A out")
            return out

    @wrap_model_call
    def b(request, handler):
        events.append("B in")
        out = handler(request)
        events.append("B out")
        return out

    def handler(request):
        events.append("handler")
        return "ok"

    assert MiddlewareStack([A, b]).model_call(_req(), handler) == "ok"
    assert events == ["A in", "B in", "handler", "B out", "A out"]


def test_model_middleware_does_not_wrap_tool_calls():
    calls = []

    @wrap_model_call
    def only_model(request, handler):
        calls.append("model")
        return handler(request)

    tc = ToolRequest({"id": "1", "name": "t", "input": {}})
    out = MiddlewareStack([only_model]).tool_call(tc, lambda req: "tool-ok")
    assert out == "tool-ok"
    assert calls == []


def test_wrap_tool_call_short_circuit_and_guard():
    @wrap_tool_call
    def deny(request, handler):
        return "被拒绝"          # 不调用 handler：缓存/审批拒绝场景

    tc = ToolRequest({"id": "1", "name": "t", "input": {"x": 1}})
    assert MiddlewareStack([deny]).tool_call(tc, lambda r: 1 / 0) == "被拒绝"

    @wrap_tool_call
    def guard(request, handler):
        try:
            return handler(request)
        except Exception as e:
            return f"工具执行失败: {e}"

    out = MiddlewareStack([guard]).tool_call(tc, lambda r: 1 / 0)
    assert out.startswith("工具执行失败")


def test_tool_request_convenience_properties():
    req = ToolRequest({"id": "1", "name": "get_weather", "input": {"city": "北京"}})
    assert req.name == "get_weather"
    assert req.args == {"city": "北京"}
    assert ToolRequest({"name": "x"}).args == {}


def test_middleware_class_returns_instance_and_validates():
    @middleware
    class Log:
        def wrap_tool_call(self, request, handler):
            return handler(request)

    assert not isinstance(Log, type)         # 装饰后名字就是实例，直接进 stack
    tc = ToolRequest({"name": "t"})
    assert MiddlewareStack([Log]).tool_call(tc, lambda r: "ok") == "ok"

    with pytest.raises(TypeError, match="不认识的钩子"):
        @middleware
        class Typo:
            def wrap_model_cal(self, request, handler):  # 少个 l，应被拦下
                ...

    with pytest.raises(TypeError, match="没有定义任何"):
        @middleware
        class Empty:
            pass


def test_middleware_on_function_wraps_both():
    seen = []

    @middleware
    def both(request, handler):
        seen.append(type(request).__name__)
        return handler(request)

    stack = MiddlewareStack([both])
    stack.model_call(_req(), lambda r: "m")
    stack.tool_call(ToolRequest({"name": "t"}), lambda r: "t")
    assert seen == ["ModelRequest", "ToolRequest"]


def test_empty_stack_passthrough():
    stack = MiddlewareStack()
    assert stack.model_call(_req(), lambda r: "raw") == "raw"


# ---------------------------------------------------------------------------
# 状态钩子：before_agent / after_agent、before_model / after_model
# ---------------------------------------------------------------------------

def test_state_hooks_order_before_forward_after_reverse():
    events = []

    @middleware
    class A:
        def before_model(self, state):
            events.append("A before")

        def after_model(self, state):
            events.append("A after")

    @middleware
    class B:
        def before_model(self, state):
            events.append("B before")

        def after_model(self, state):
            events.append("B after")

    stack = MiddlewareStack([A, B])
    stack.before_model({})
    assert events == ["A before", "B before"]       # before 正序
    events.clear()
    stack.after_model({})
    assert events == ["B after", "A after"]         # after 逆序（洋葱出口）


def test_state_hook_updates_are_merged():
    @before_model
    def add_flag(state):
        return {"flag": True}

    @before_model
    def add_count(state):
        return {"count": 1}

    updates = MiddlewareStack([add_flag, add_count]).before_model({})
    assert updates == {"flag": True, "count": 1}
    assert MiddlewareStack().before_agent({}) == {}  # 没有更新时是空 dict


def test_function_decorators_only_fire_their_hook():
    seen = []

    @before_agent
    def ba(state):
        seen.append("before_agent")

    @after_agent
    def aa(state):
        seen.append("after_agent")

    @before_model
    def bm(state):
        seen.append("before_model")

    @after_model
    def am(state):
        seen.append("after_model")

    stack = MiddlewareStack([ba, aa, bm, am])
    stack.before_agent({})
    stack.after_agent({})
    stack.before_model({})
    stack.after_model({})
    assert sorted(seen) == ["after_agent", "after_model",
                            "before_agent", "before_model"]
    # 纯状态钩子中间件不影响 wrap 链
    assert stack.model_call(_req(), lambda r: "ok") == "ok"


def test_state_only_class_middleware_does_not_break_wrap_chain():
    @middleware
    class OnlyState:
        def before_model(self, state):
            return None

    stack = MiddlewareStack([OnlyState])
    assert stack.model_call(_req(), lambda r: "ok") == "ok"
    assert stack.tool_call(ToolRequest({"name": "t"}), lambda r: "t") == "t"


def test_agent_hooks_and_state_hook_typo_detection():
    events = []

    @middleware
    class Hooks:
        def before_agent(self, state):
            events.append("ba")

        def after_agent(self, state):
            events.append("aa")

    stack = MiddlewareStack([Hooks])
    stack.before_agent({})
    stack.after_agent({})
    assert events == ["ba", "aa"]

    with pytest.raises(TypeError, match="不认识的钩子"):
        @middleware
        class Typo:
            def before_modell(self, state):          # 多个 l，应被拦下
                ...


# ---------------------------------------------------------------------------
# 状态钩子 -> 图节点（add_hook_nodes，对齐 create_agent）
# ---------------------------------------------------------------------------

class _FakeGraph:
    def __init__(self):
        self.nodes = {}

    def add_node(self, name, fn):
        self.nodes[name] = fn


def test_add_hook_nodes_names_and_execution_order():
    @middleware
    class A:
        def before_model(self, state):
            return {"a": 1}

        def after_model(self, state):
            return None

    @middleware
    class B:
        def before_model(self, state):
            return None

        def after_model(self, state):
            return None

    g = _FakeGraph()
    hooks = add_hook_nodes(g, MiddlewareStack([A, B]))
    assert hooks["before_model"] == ["A.before_model", "B.before_model"]   # 正序
    assert hooks["after_model"] == ["B.after_model", "A.after_model"]      # 逆序
    assert hooks["before_agent"] == [] and hooks["after_agent"] == []
    # 节点就是可执行函数：返回钩子给的更新；None -> {}
    assert g.nodes["A.before_model"]({"x": 1}) == {"a": 1}
    assert g.nodes["B.before_model"]({}) == {}


def test_add_hook_nodes_skips_base_and_around_only():
    @before_model
    def bm(state):
        return {"seen": True}

    @wrap_model_call
    def around(request, handler):
        return handler(request)

    g = _FakeGraph()
    hooks = add_hook_nodes(g, MiddlewareStack([bm, around]))
    # @before_model 只生成自己那个钩子节点；around-only 不生成状态节点
    assert hooks["before_model"] == ["bm.before_model"]
    assert all(not names for hook, names in hooks.items() if hook != "before_model")
    assert len(g.nodes) == 1
