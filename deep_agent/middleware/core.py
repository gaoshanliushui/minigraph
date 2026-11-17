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
from typing import Annotated, Any, Callable, Literal

from minigraph.channels import EphemeralValue
from minigraph.constants import END, LAST_VALUE, START

Handler = Callable[[Any], Any]

_WRAP_HOOKS = ("wrap_model_call", "wrap_tool_call")
_STATE_HOOKS = ("before_agent", "after_agent", "before_model", "after_model")
_HOOKS = _WRAP_HOOKS + _STATE_HOOKS

# jump_to 的三个目的地（对齐 LangChain v1 的 JumpTo）。
#   "tools" 跳到工具节点
#   "model" 跳回模型节点（再问一次）
#   "end"   直接结束
JumpTo = Literal["tools", "model", "end"]

# jump_to 存在 state 里的字段名。它必须是**临时通道**（EphemeralValue）：
# 跳一次、读一次，下个超步就没了 —— 否则一次跳转会被后续每一轮重复执行。
JUMP_TO_FIELD = "jump_to"

#: agent state 里 jump_to 字段的声明。直接用：
#:
#:     class AgentState(State):
#:         messages: Annotated[list, DeltaChannel(add_messages, list)]
#:         jump_to: Annotated[str | None, jump_to_channel()]
#:
#: 它是**临时通道**：本超步写进去的值，下个超步就没了。
#: 这样"跳一次"不会变成"每轮都跳"。
JUMP_TO_ANNOTATION = Annotated[str | None, EphemeralValue(JUMP_TO_FIELD)]


# ---------------------------------------------------------------------------
# 请求对象：around 中间件看到和修改的就是它们
# ---------------------------------------------------------------------------

@dataclass
class ModelRequest:
    """一次模型调用的全部入参。中间件可整体替换 model/messages/tools/system。"""

    model: str
    messages: list[dict]
    tools: list[dict] = field(default_factory=list)
    max_tokens: int = 1024
    system: str | None = None          # Messages API 的顶层 system 参数
    state: dict | None = None          # 当时的 agent state（可选，供状态感知中间件读取）


@dataclass
class ModelResponse:
    """
    一次模型调用的结果（对齐 LangChain v1 的 ModelResponse）。

    只有一个字段 `result`：content blocks 列表（Anthropic 风格的
    text / tool_use block）。LangChain 里是 BaseMessage 列表，
    minigraph 的消息模型是 dict，所以这里直接是 block 列表。
    """

    result: list[dict]
    structured_response: Any = None


@dataclass
class ToolRequest:
    """
    一次工具调用。tool_call 是 Anthropic 的 tool_use block：{id, name, input}。

    state  是当时的 agent 状态（可选）——
    有些中间件要读状态才能决定怎么处理这次调用
    （比如 human_in_the_loop 要读"人工改过的调用"记在哪）。
    """

    tool_call: dict
    state: dict | None = None

    @property
    def name(self) -> str:
        return self.tool_call.get("name", "")

    @property
    def args(self) -> dict:
        return self.tool_call.get("input", {}) or {}


# ---------------------------------------------------------------------------
# 中间件本体
# ---------------------------------------------------------------------------

def hook_config(*, can_jump_to: list[JumpTo] | None = None):
    """
    标注一个状态钩子**能跳到哪些地方**（对齐 LangChain v1 的 hook_config）。

        @middleware
        class RateLimit:
            @hook_config(can_jump_to=["end"])
            def before_model(self, state):
                if too_many_calls(state):
                    return {"jump_to": "end"}
                return None

    声明了 can_jump_to 的钩子，其节点会变成**条件边**而不是静态边：
    钩子返回 {"jump_to": "end"} 就跳到出口，返回 None（或其他状态更新）
    就走默认的下一个节点。

    为什么必须显式声明：跳转会绕过图里的正常拓扑，是"逃生舱"。
    声明出来，图的接线阶段才知道要给这个节点建哪几条边 ——
    没声明的钩子想跳也没地方跳（jump_to 会被忽略）。

    函数式写法等价：

        @before_model(can_jump_to=["end"])
        def bail_out(state):
            return {"jump_to": "end"}
    """
    def deco(fn: Callable) -> Callable:
        if can_jump_to is not None:
            fn.__can_jump_to__ = list(can_jump_to)
        return fn
    return deco


class Middleware:
    """
    可选基类：六个钩子默认直通/无更新。不继承它、只定义钩子方法也可以。

    对齐 LangChain v1 的 AgentMiddleware，额外支持：

      - ``name``      节点名（"LogCalls.before_model" 里的前半段）。
                      默认取类名；_FuncMiddleware 用函数名。
      - ``tools``     中间件自己注册的工具（会并进 agent 的工具表）。
      - ``state_schema`` 该中间件**额外需要的 state 字段**。
                      见 merge_state_schemas()。
    """

    #: 中间件自己注册的工具。create_agent 会把它们并进工具表，
    #: 于是中间件可以"顺带"提供工具，不必让调用者单独传。
    tools: tuple = ()

    #: 该中间件额外需要的 state 字段（State 子类或 {字段: reducer} 裸 dict）。
    #: agent 组装时会把这些字段并进 agent 的 state schema ——
    #: 中间件要计数、要记状态，就不用让每个使用者手动往 schema 里加字段。
    state_schema: Any = None

    @property
    def name(self) -> str:
        """节点名前缀。默认类名；可以覆盖。"""
        return type(self).__name__

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
        # 基类的 name 是**只读 property**，不能直接 self.name = ... 赋值
        # （property 没有 setter -> AttributeError）。所以存到 _name，
        # 再用下面的 property 把它露出来。
        self._name = getattr(fn, "__name__", "middleware")

    @property
    def name(self) -> str:
        return self._name

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

    def __init__(self, fn: Callable, hook: str,
                 can_jump_to: list[JumpTo] | None = None) -> None:
        self.fn = fn
        self.hook = hook
        # 声明了跳转能力 -> 该钩子节点要建成条件边
        self.can_jump_to = list(can_jump_to) if can_jump_to else None
        self._name = getattr(fn, "__name__", "middleware")

    @property
    def name(self) -> str:
        return self._name

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
    """
    造一个状态钩子装饰器。

    两种用法：
        @before_model                       —— fn(state) -> 更新 dict 或 None
        @before_model(can_jump_to=["end"])  —— 声明这个钩子能跳到哪些地方
    """
    def deco(fn: Callable | None = None, *, can_jump_to: list[JumpTo] | None = None):
        # 兼容 @before_model 和 @before_model(can_jump_to=[...]) 两种写法
        if fn is None:
            def inner(f: Callable) -> _StateFuncMiddleware:
                return _StateFuncMiddleware(f, hook, can_jump_to)
            inner.__name__ = hook
            inner.__doc__ = deco.__doc__
            return inner
        return _StateFuncMiddleware(fn, hook, can_jump_to)
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
# 状态钩子 -> 图节点（对齐 LangChain create_agent 的 factory）
# ---------------------------------------------------------------------------
# LangChain（langchain/agents/factory.py）不是"在模型节点里手动调钩子"，
# 而是给每个实现了状态钩子的中间件**生成一个图节点**：
#   节点名   f"{middleware.name}.{hook}"（如 "LogCalls.before_model"）
#   入口     START -> before_agent*(正序) -> before_model*(正序) -> model
#   回环     tools -> before_model* -> model        （loop_entry）
#   出口     model -> after_model*(逆序) -> 路由     （loop_exit 上挂条件边）
#   结束     路由的 END 分支 -> after_agent*(逆序) -> END
# before_agent/after_agent 每次 agent 调用只跑一次，before_model/after_model
# 每轮 ReAct 都过 —— 这是它们和 around 钩子的本质区别，也是"为什么应该是
# 节点"：执行时机由**图拓扑**表达，而不是藏在某个节点的函数体里。

def _mw_name(mw) -> str:
    return getattr(mw, "name", None) or type(mw).__name__


def _implements(mw, hook: str) -> bool:
    """该中间件是否**真的实现了**这个状态钩子（而不是基类默认直通）。"""
    if isinstance(mw, _StateFuncMiddleware):
        return mw.hook == hook          # @before_model 这类只实现自己那一个
    fn = getattr(type(mw), hook, None)
    return fn is not None and fn is not getattr(Middleware, hook)


def _can_jump_to(mw, hook: str) -> list[JumpTo] | None:
    """
    这个中间件的该钩子声明了哪些跳转目的地（没有声明就是 None）。

    两条来源：
      - 函数式中间件（@before_model(can_jump_to=[...])）直接带在对象上
      - 类式中间件在方法上用 @hook_config(can_jump_to=[...]) 标注
    """
    if isinstance(mw, _StateFuncMiddleware):
        return mw.can_jump_to
    fn = getattr(type(mw), hook, None)
    if fn is None:
        return None
    return getattr(fn, "__can_jump_to__", None)


def add_hook_nodes(g, stack: "MiddlewareStack") -> dict[str, list[str]]:
    """
    把 stack 里实现了状态钩子的中间件各注册成一个图节点。

    返回 {hook: [节点名, ...]}，列表按**执行顺序**排列：
    before_* 正序、after_* 逆序（after 与 wrap 链的洋葱出口一致）。
    接线（入口/回环/出口/END 分支）由调用方按 create_agent 的拓扑完成
    （见 examples/03 的 build()），本函数只负责"哪些节点、什么顺序"。

    声明了 can_jump_to 的钩子节点，它的返回值里可能有 {"jump_to": ...}；
    这个返回值照常写进 state（jump_to 是个临时通道，读完即清），
    路由函数负责读它。接线见 wire_middleware_graph()。
    """
    result: dict[str, list[str]] = {}
    for hook in _STATE_HOOKS:
        mws = [mw for mw in stack.middlewares if _implements(mw, hook)]
        if hook.startswith("after_"):
            mws.reverse()
        names = []
        for mw in mws:
            name = f"{_mw_name(mw)}.{hook}"
            fn = getattr(mw, hook)

            def node(state, _fn=fn):
                return _fn(state) or {}   # None = 无状态更新

            node.__name__ = name
            g.add_node(name, node)
            names.append(name)
        result[hook] = names
    return result


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

    # -- 跳转能力 -----------------------------------------------------------
    def jump_targets(self, hook: str) -> dict[str, list[JumpTo]]:
        """
        {节点名: 该节点声明的跳转目的地}，只含**声明过**的节点。

        声明过跳转的钩子节点会建成条件边；没声明的就是静态边。
        """
        out: dict[str, list[JumpTo]] = {}
        for mw in self.middlewares:
            if not _implements(mw, hook):
                continue
            targets = _can_jump_to(mw, hook)
            if targets:
                out[f"{_mw_name(mw)}.{hook}"] = list(targets)
        return out

    def has_jumps(self) -> bool:
        """栈里有没有中间件声明过跳转（决定 state 要不要 jump_to 字段）。"""
        return any(self.jump_targets(hook)
                   for hook in _STATE_HOOKS)


# ---------------------------------------------------------------------------
# 接线：把中间件的钩子节点按 create_agent 的拓扑接进图
# ---------------------------------------------------------------------------
# 这一段对应 LangChain 的 factory.py::_add_middleware_edges / _make_*_edge。
# 抽出来是因为 example 03 和 deep_agent 都要同一套拓扑，写两遍必然漂移。
#
# 拓扑（无跳转时）：
#
#     START -> before_agent*(正序) -> before_model*(正序) -> model
#     tools -> before_model*(正序) -> model          （回环也过 before_model）
#     model -> after_model*(逆序) -> 路由
#     路由的 END 分支 -> after_agent*(逆序) -> END
#
# 拓扑（有跳转时）：声明了 can_jump_to 的钩子节点，它的**出边**变成条件边：
#     钩子返回 {"jump_to": X} -> 去 X
#     否则                     -> 去默认的下一个节点

def _resolve_jump(jump_to: str | None, *, model_dest: str, end_dest: str,
                   tools_dest: str = "tools") -> str | None:
    """jump_to 的值 -> 实际节点名（对齐 LangChain 的 _resolve_jump）。"""
    if jump_to == "model":
        return model_dest
    if jump_to == "end":
        return end_dest
    if jump_to == "tools":
        return tools_dest
    return None


def _chain(g, src: str, names: list[str]) -> str:
    """src -> n1 -> n2 -> ...，返回链尾（names 为空时返回 src）。"""
    prev = src
    for n in names:
        g.add_edge(prev, n)
        prev = n
    return prev


def wire_middleware_graph(g, stack: "MiddlewareStack", *,
                          model_node: str = "model",
                          tools_node: str = "tools",
                          join_node: str | None = None,
                          route: Callable[[dict], str] | None = None,
                          state_schema=None):
    """
    把中间件的钩子节点接进一张已经建好 model/tools 节点的图。

    参数：
      model_node / tools_node  两个主节点的名字
      join_node                可选的"结果汇总"节点。给了它，工具回环就变成
                               tools -> join_node -> before_model* -> model：
                               并行工具任务各写各的结果，汇总节点再拼成
                               一条消息。不给就是 tools -> before_model* -> model。
      route                    路由函数：state -> "tools" | END，
                               也可以返回 list[Send] 做并行扇出。
                               决定模型之后是去工具还是收尾。
      state_schema             agent 的 state schema（要含 jump_to 字段）

    返回 {hook: [节点名, ...]}，和各钩子的节点顺序。

    接线规则（对齐 LangChain factory.py）：
      - 入口：START -> before_agent* -> before_model* -> model
      - 回环：tools -> join_node? -> before_model* -> model
      - 出口：model -> after_model* -> 路由
      - 收尾：路由的 END 分支 -> after_agent* -> END
      - 声明了 can_jump_to 的钩子节点：出边是条件边，跳转优先于默认走向
    """
    hooks = add_hook_nodes(g, stack)

    # 跳转目的地：
    #   "model" -> loop_entry（before_model 链首）。对齐 LangChain：
    #     _add_middleware_edge 的 model_destination 一律是 loop_entry_node，
    #     不是 model 节点本身 —— 跳回模型要**重新过一遍 before_model 钩子**。
    #   "end"   -> after_agent 链首（没有就直接 END）
    #   "tools" -> 工具节点
    after_agent_nodes = hooks["after_agent"]
    end_dest = after_agent_nodes[0] if after_agent_nodes else END
    loop_entry = hooks["before_model"][0] if hooks["before_model"] else model_node

    # 需要建成条件边的钩子节点（声明过 can_jump_to 的）
    jump_nodes: dict[str, list[JumpTo]] = {}
    for hook in _STATE_HOOKS:
        jump_nodes.update(stack.jump_targets(hook))

    def _add_jump_edge(node: str, default_dest: str):
        """给声明了跳转的钩子节点接条件边；没声明的接静态边。"""
        targets = jump_nodes.get(node)
        if not targets:
            g.add_edge(node, default_dest)
            return

        def _jump_edge(state):
            # 跳转优先：钩子写了 {"jump_to": X} 就走 X
            resolved = _resolve_jump(
                state.get(JUMP_TO_FIELD),
                model_dest=loop_entry, end_dest=end_dest,
                tools_dest=tools_node,
            )
            return resolved or default_dest

        destinations = [default_dest]
        if "end" in targets and end_dest not in destinations:
            destinations.append(end_dest)
        if "tools" in targets and tools_node in g.nodes \
                and tools_node not in destinations:
            destinations.append(tools_node)
        if "model" in targets and loop_entry not in destinations:
            destinations.append(loop_entry)
        g.add_conditional_edges(node, _jump_edge, destinations)

    # ---- 逐节点接线（对齐 LangChain 的 _add_middleware_edge 逐个节点接）----
    #
    # 每个钩子节点都有自己的**出边**：
    #   - 链中间的节点 -> 链的下一个节点
    #   - 链尾         -> 该链的终点（model / 路由 / END）
    # 声明了 can_jump_to 的节点，出边升级成条件边。
    #
    # 为什么必须逐节点接：链中间某个钩子声明了跳转时，它要能直接跳走，
    # 不能被"链 -> 下一个"这条静态边锁死。
    wired: set[str] = set()

    def _wire(node: str, default_dest: str):
        """给一个钩子节点接出边（同一个节点只接一次）。"""
        if node in wired:
            return
        wired.add(node)
        _add_jump_edge(node, default_dest)

    def _wire_chain(nodes: list[str], tail: str):
        """nodes[0] -> nodes[1] -> ... -> nodes[-1] -> tail，逐节点接。"""
        for i, node in enumerate(nodes):
            nxt = nodes[i + 1] if i + 1 < len(nodes) else tail
            _wire(node, nxt)

    exits = hooks["after_agent"]

    # ---- 入口：START -> before_agent* -> before_model* -> model ----
    _wire_chain(hooks["before_agent"], hooks["before_model"][0]
                if hooks["before_model"] else model_node)
    _wire_chain(hooks["before_model"], model_node)
    if not hooks["before_agent"] and not hooks["before_model"]:
        g.add_edge(START, model_node)
    elif not hooks["before_agent"]:
        g.add_edge(START, hooks["before_model"][0])
    else:
        g.add_edge(START, hooks["before_agent"][0])

    # ---- 回环：tools -> join_node? -> before_model* -> model ----
    # 入口链和回环链的尾节点常常是同一个（before_model 链尾）。
    # 它的出边只能接一次 —— 接两次会变成两条条件边，
    # 节点每轮被触发两次，图就转起来了。_wire 的 wired 集合负责去重。
    _wire_chain(hooks["before_model"], model_node)
    # 没有工具节点的图（纯问答 agent）就不接这条回环
    if tools_node in g.nodes:
        # 工具节点可能收到多个并行任务（Send 扇出），但**汇总**只该发生一次，
        # 所以回环的静态边挂在 join_node 上：所有并行任务跑完后，
        # join_node 被触发一次，把结果拼成一条消息再回模型。
        after_tools = join_node if (join_node and join_node in g.nodes) \
            else tools_node
        if hooks["before_model"]:
            g.add_edge(after_tools, hooks["before_model"][0])
        else:
            g.add_edge(after_tools, model_node)
        if after_tools != tools_node:
            g.add_edge(tools_node, after_tools)

    # ---- 出口：model -> after_model* -> 路由 ----
    # 链尾的出边在"路由边"那一段统一处理（它是路由源，不是静态边）。
    if hooks["after_model"]:
        g.add_edge(model_node, hooks["after_model"][0])
        for i in range(len(hooks["after_model"]) - 1):
            _wire(hooks["after_model"][i], hooks["after_model"][i + 1])

    # ---- 收尾：after_agent 链 -> END ----
    # after_agent 链是**逆序**的（add_hook_nodes 已经排好）：
    #   路由的 END 分支 -> after_agent[0] -> ... -> after_agent[-1] -> END
    _wire_chain(hooks["after_agent"], END)

    # ---- 路由边 ----
    # 路由挂在 after_model 链尾（没有 after_model 钩子就是 model 自己）。
    # 如果这个节点同时声明了跳转，跳转**优先**于路由 —— 所以路由函数
    # 要先看 jump_to。
    route_from = hooks["after_model"][-1] if hooks["after_model"] else model_node
    if route is not None:
        default_end = exits[0] if exits else END
        targets = jump_nodes.get(route_from)

        def _route_edge(state, _route=route):
            resolved = _resolve_jump(
                state.get(JUMP_TO_FIELD),
                model_dest=loop_entry, end_dest=end_dest,
                tools_dest=tools_node,
            )
            if resolved is not None:
                return resolved
            destination = _route(state)
            # 业务路由只知道 END；图存在 after_agent 钩子时，真正的结束
            # 分支必须先进入收尾链，不能直接绕过它跳到图终点。
            return default_end if destination == END else destination

        destinations = [default_end]
        # 没有工具节点的图不把 "tools" 列为合法目的地
        if tools_node in g.nodes:
            destinations.insert(0, "tools")
        if targets:
            if "model" in targets and loop_entry not in destinations:
                destinations.append(loop_entry)
            if "end" in targets and default_end not in destinations:
                destinations.append(default_end)
            if "tools" in targets and tools_node in g.nodes \
                    and tools_node not in destinations:
                destinations.append(tools_node)
        # after_model 链尾自己就是路由源；它前面那个节点要接到它
        if len(hooks["after_model"]) > 1:
            _wire(hooks["after_model"][-2], route_from)
        g.add_conditional_edges(route_from, _route_edge, destinations)
        wired.add(route_from)

    return hooks


# ---------------------------------------------------------------------------
# state schema 合并：中间件要的字段并进 agent 的 schema
# ---------------------------------------------------------------------------
# 中间件要记状态（调用次数、待审批的动作……），就得有 state 字段。
# 让每个使用者手动往 agent 的 State 里加字段很啰嗦，而且加错了
# （忘了声明）运行时才报"未声明的字段"。
#
# 所以中间件在 state_schema 里声明自己要的字段，组装时统一合并：
#
#     class ModelCallLimitState(State):
#         thread_model_call_count: int
#         run_model_call_count: int
#
#     class LimitMW(Middleware):
#         state_schema = ModelCallLimitState
#
#     app = create_agent(model, middleware=[LimitMW(run_limit=5)])
#
# 合并是"字段取并集"：agent 自己的字段优先（同名字段以 agent 为准，
# 避免中间件悄悄改掉 agent 的语义）。

def merge_state_schemas(base, extra_schemas, *, need_jump_to: bool = False):
    """
    把若干个 schema 的字段合并成一个 State 子类。

    base           agent 自己的 state schema（State 子类或 {字段: reducer}）
    extra_schemas  各中间件声明的 state_schema（列表）
    need_jump_to   是否需要 jump_to 字段。

    同名字段以 base 为准 —— agent 自己的字段优先，
    避免中间件悄悄改掉 agent 的语义。

    need_jump_to=True 时自动补上 jump_to 字段（临时通道）：
    声明了 can_jump_to 的中间件要写 {"jump_to": X}，
    而运行时只认**已声明**的字段 —— 忘了声明会在运行时炸
    "节点写了未声明的字段 jump_to"。让组装阶段自动补上，
    使用者就不用记这件事了。
    """
    from minigraph.state import _schema, _schema_channels, _schema_types

    merged: dict[str, Any] = dict(_schema(base))
    merged_types = dict(_schema_types(base))
    merged_channels = dict(_schema_channels(base))

    for extra in extra_schemas:
        if extra is None:
            continue
        for field in _schema(extra):
            merged.setdefault(field, _schema(extra)[field])
        for field, typ in _schema_types(extra).items():
            merged_types.setdefault(field, typ)
        for field, chan in _schema_channels(extra).items():
            merged_channels.setdefault(field, chan)

    if need_jump_to and JUMP_TO_FIELD not in merged:
        # 临时通道：本超步写进去的值，下个超步就没了
        merged[JUMP_TO_FIELD] = LAST_VALUE
        merged_types[JUMP_TO_FIELD] = str | None
        merged_channels[JUMP_TO_FIELD] = EphemeralValue(JUMP_TO_FIELD)

    # reducer 和类型信息要一起交给 StateGraph —— 它按 schema 形态建通道。
    # 裸 dict 形态带不走类型/通道信息，所以这里返回一个 State 子类。
    return _make_merged_state(merged, merged_types, merged_channels)


def _make_merged_state(reducers: dict[str, Any], types: dict[str, Any],
                       channels: dict[str, Any]):
    """造一个 State 子类，装着合并后的字段。"""
    from minigraph.state import State

    ann: dict[str, Any] = {}
    for field, reducer in reducers.items():
        typ = types.get(field)
        if channels.get(field) is not None:
            ann[field] = Annotated[typ or Any, channels[field]]
        elif typ is not None:
            ann[field] = Annotated[typ, reducer]
        else:
            ann[field] = reducer
    return type("MergedAgentState", (State,), {"__annotations__": ann})


def agent_state_schema(base, stack: "MiddlewareStack"):
    """
    按中间件栈算出 agent 真正要用的 state schema。

        schema = agent_state_schema(AgentState, MiddlewareStack([...]))
        g = StateGraph(schema)

    做两件事：
      1. 把各中间件声明的 state_schema 并进来
      2. 栈里有中间件声明过跳转时，补上 jump_to 字段

    比手动 merge_state_schemas 省事 —— 跳转能力是栈的性质，
    让使用者自己判断"要不要 jump_to 字段"太容易漏。
    """
    extras = [getattr(mw, "state_schema", None) for mw in stack.middlewares]
    return merge_state_schemas(base, extras, need_jump_to=stack.has_jumps())
