"""
§8 中间件 —— 模型/工具调用的横切层（对齐 LangChain 1.0 agent middleware）

图内核只管"节点什么时候跑、写进哪个通道"；**节点里面**调模型、调工具
这两件事，是横切关注点天然聚集的地方：日志、重试、限流、缓存、异常兜底、
人工审批……中间件把它们从业务代码里剥出来。

## 六个钩子

around 型（洋葱模型，拿到 request 和 handler）：

    wrap_model_call(request, handler)   # request: ModelRequest，可改；handler(request) 发起真实调用
    wrap_tool_call(request, handler)    # request: ToolRequest，可改；handler(request) 执行真实工具

状态型（拿到 agent state，返回 **state 更新 dict** 或 None）：

    before_agent(state)   # 整个 agent 调用开始前，一次
    after_agent(state)    # 整个 agent 调用结束后，一次
    before_model(state)   # 每次模型调用前（ReAct 每轮都触发）
    after_model(state)    # 每次模型调用后

执行顺序：before_* 按中间件列表**正序**，after_* **逆序** ——
与 wrap_* 的洋葱出口一致（A in -> B in -> handler -> B out -> A out）。
多个中间件返回的更新按执行顺序 dict.update 合并（后执行的覆盖先执行的）。

## 写法

    # 1) 类式：@middleware 装饰带钩子方法的类（六个钩子任意组合），
    #    装饰后名字就是实例；钩子名拼错会在装饰时直接报错
    @middleware
    class LogCalls:
        def before_model(self, state):
            print("历史消息:", len(state["messages"]))
        def wrap_model_call(self, request, handler):
            return handler(request)

    # 2) 函数式 around：只包模型 / 工具调用
    @wrap_model_call
    def retry(request, handler):
        for attempt in range(3):
            try:
                return handler(request)
            except Exception:
                if attempt == 2:
                    raise

    @wrap_tool_call
    def guard(request, handler):
        try:
            return handler(request)
        except Exception as e:
            return f"工具执行失败: {e}"     # 短路：异常变成结果，不往下抛

    # 3) 函数式状态钩子：fn(state) -> 更新 dict 或 None
    @before_model
    def audit(state):
        return {"audit_count": 1}

    # @middleware 装饰单个函数 = 同时包模型和工具调用（统一日志这类场景）

## 执行

    stack = MiddlewareStack([LogCalls, retry, guard])   # wrap 链：第一个在最外层

    stack.before_agent(input_state)         # agent 入口/出口各一次（调用方安排）
    stack.after_agent(final_state)

    # 模型节点里：
    pre  = stack.before_model(state)
    out  = stack.model_call(ModelRequest(model=..., messages=..., tools=...), real_chat)
    post = stack.after_model(state)
    # pre/post 是 state 更新，由节点合并进自己的返回值（走各字段的 reducer）

    # 工具节点里：
    result = stack.tool_call(ToolRequest(tool_use_block), real_tool)

handler 约定：`handler(request) -> result`。around 中间件可以：改 request
再传给 handler、多次调用 handler（重试）、不调用 handler 直接返回
（缓存/审批拒绝）、改写 handler 的返回值。只定义状态钩子的中间件会被
wrap 链自动跳过，互不干扰。
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

Handler = Callable[[Any], Any]

_WRAP_HOOKS = ("wrap_model_call", "wrap_tool_call")
_STATE_HOOKS = ("before_agent", "after_agent", "before_model", "after_model")
_HOOKS = _WRAP_HOOKS + _STATE_HOOKS


# ---------------------------------------------------------------------------
# 请求对象：around 中间件看到和修改的就是它们
# ---------------------------------------------------------------------------

@dataclass
class ModelRequest:
    """一次模型调用的全部入参。中间件可整体替换 model/messages/tools。"""

    model: str
    messages: list[dict]
    tools: list[dict] = field(default_factory=list)
    max_tokens: int = 1024


@dataclass
class ToolRequest:
    """一次工具调用。tool_call 是 Anthropic 的 tool_use block：{id, name, input}。"""

    tool_call: dict

    @property
    def name(self) -> str:
        return self.tool_call.get("name", "")

    @property
    def args(self) -> dict:
        return self.tool_call.get("input", {}) or {}


# ---------------------------------------------------------------------------
# 中间件本体
# ---------------------------------------------------------------------------

class Middleware:
    """可选基类：六个钩子默认直通/无更新。不继承它、只定义钩子方法也可以。"""

    def wrap_model_call(self, request, handler: Handler):
        return handler(request)

    def wrap_tool_call(self, request, handler: Handler):
        return handler(request)

    def before_agent(self, state) -> dict | None:
        return None

    def after_agent(self, state) -> dict | None:
        return None

    def before_model(self, state) -> dict | None:
        return None

    def after_model(self, state) -> dict | None:
        return None


class _FuncMiddleware(Middleware):
    """把一个 fn(request, handler) 变成 around 中间件；kinds 决定包哪类调用。"""

    def __init__(self, fn: Callable, kinds: frozenset[str]) -> None:
        self.fn = fn
        self.kinds = kinds
        self.name = getattr(fn, "__name__", "middleware")

    def wrap_model_call(self, request, handler: Handler):
        if "model" in self.kinds:
            return self.fn(request, handler)
        return handler(request)

    def wrap_tool_call(self, request, handler: Handler):
        if "tool" in self.kinds:
            return self.fn(request, handler)
        return handler(request)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<middleware {self.name} ({'/'.join(sorted(self.kinds))})>"


class _StateFuncMiddleware(Middleware):
    """把一个 fn(state) 绑到某个状态钩子上（@before_model 这类装饰器的产物）。"""

    def __init__(self, fn: Callable, hook: str) -> None:
        self.fn = fn
        self.hook = hook
        self.name = getattr(fn, "__name__", "middleware")

    def _run(self, hook: str, state):
        return self.fn(state) if self.hook == hook else None

    def before_agent(self, state):
        return self._run("before_agent", state)

    def after_agent(self, state):
        return self._run("after_agent", state)

    def before_model(self, state):
        return self._run("before_model", state)

    def after_model(self, state):
        return self._run("after_model", state)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<middleware {self.name} ({self.hook})>"


def wrap_model_call(fn: Callable) -> _FuncMiddleware:
    """@wrap_model_call：fn(request, handler) 只包**模型**调用。"""
    return _FuncMiddleware(fn, frozenset({"model"}))


def wrap_tool_call(fn: Callable) -> _FuncMiddleware:
    """@wrap_tool_call：fn(request, handler) 只包**工具**调用。"""
    return _FuncMiddleware(fn, frozenset({"tool"}))


def _state_hook(hook: str):
    def deco(fn: Callable) -> _StateFuncMiddleware:
        return _StateFuncMiddleware(fn, hook)
    deco.__name__ = hook
    deco.__doc__ = f"@{hook}：fn(state) -> state 更新 dict 或 None。"
    return deco


before_agent = _state_hook("before_agent")
after_agent = _state_hook("after_agent")
before_model = _state_hook("before_model")
after_model = _state_hook("after_model")


def middleware(target):
    """
    @middleware：
      - 装饰**类**：六个钩子任意组合；校验钩子名（防拼写错误），
        返回**实例**（无参构造）。至少要定义一个钩子。
      - 装饰**函数**：fn(request, handler) 同时包模型和工具调用。
    """
    if inspect.isclass(target):
        defined = {n for n in _HOOKS if n in target.__dict__}
        typo = [n for n in vars(target)
                if n.startswith(("wrap_", "before_", "after_"))
                and n not in _HOOKS]
        if typo:
            raise TypeError(
                f"{target.__name__} 里有不认识的钩子 {typo}；"
                f"可用钩子：{', '.join(_HOOKS)}"
            )
        if not defined:
            raise TypeError(
                f"{target.__name__} 没有定义任何中间件钩子（{_HOOKS}）"
            )
        return target()
    if callable(target):
        return _FuncMiddleware(target, frozenset({"model", "tool"}))
    raise TypeError("@middleware 只能装饰类或函数")


# ---------------------------------------------------------------------------
# 执行：把中间件链套到真实调用 / agent 节点上
# ---------------------------------------------------------------------------

def _bind(fn: Callable, nxt: Handler) -> Handler:
    # fn / nxt 是本函数的局部变量，每次调用独立 —— 闭包不会串链。
    def wrapped(request):
        return fn(request, nxt)
    return wrapped


class MiddlewareStack:
    """
    中间件链。

      - wrap_*：洋葱序，列表第一个在最外层（先看到请求、后看到响应）；
        没有该钩子的中间件自动跳过（纯状态钩子中间件不影响 wrap 链）。
      - before_* / after_*：状态钩子，before 正序、after 逆序；
        返回合并后的 state 更新 dict（没有更新时是 {}）。
    """

    def __init__(self, middlewares=()) -> None:
        self.middlewares = list(middlewares)

    # -- around 型 ----------------------------------------------------------
    def _chain(self, hook: str, request, handler: Handler):
        for mw in reversed(self.middlewares):
            fn = getattr(mw, hook, None)
            if fn is None:                 # 纯状态钩子中间件不进这条链
                continue
            handler = _bind(fn, handler)
        return handler(request)

    def model_call(self, request: ModelRequest, handler: Handler):
        return self._chain("wrap_model_call", request, handler)

    def tool_call(self, request: ToolRequest, handler: Handler):
        return self._chain("wrap_tool_call", request, handler)

    # -- 状态型 -------------------------------------------------------------
    def _run_state_hooks(self, hook: str, state, *, reverse: bool) -> dict:
        updates: dict = {}
        mws = reversed(self.middlewares) if reverse else self.middlewares
        for mw in mws:
            fn = getattr(mw, hook, None)
            if fn is None:
                continue
            result = fn(state)
            if result:
                updates.update(result)
        return updates

    def before_agent(self, state) -> dict:
        return self._run_state_hooks("before_agent", state, reverse=False)

    def after_agent(self, state) -> dict:
        return self._run_state_hooks("after_agent", state, reverse=True)

    def before_model(self, state) -> dict:
        return self._run_state_hooks("before_model", state, reverse=False)

    def after_model(self, state) -> dict:
        return self._run_state_hooks("after_model", state, reverse=True)
