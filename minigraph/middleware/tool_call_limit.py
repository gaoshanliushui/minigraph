"""
工具调用次数上限（对齐 LangChain v1 的 middleware/tool_call_limit.py）。

和 model_call_limit 的区别：
  - 计数是 **dict**（按工具名分开记），不是单个整数
  - 可以只限某一个工具（tool_name="search"），也可以限所有工具
  - 超限行为多一种："continue" —— 只拦下超限的那个工具，
    同批次里的其他工具照常执行，agent 继续跑

因为要按工具名分桶计，state 字段是 dict[str, int]；
没有指定 tool_name 时用 "__all__" 这个桶。
"""

from __future__ import annotations

from typing import Any, Literal

from .core import Middleware, ToolRequest, hook_config
from ..state import State

__all__ = [
    "ToolCallLimitState",
    "ToolCallLimitMiddleware",
    "ToolCallLimitExceededError",
    "ExitBehavior",
]

#: 超限后的行为
ExitBehavior = Literal["continue", "error", "end"]

#: 不指定具体工具时，计数存在这个桶里
ALL_TOOLS = "__all__"

#: 各 limiter 在本轮拦下的 tool_call id；wrap_tool_call 据此拒绝执行
_BLOCKED_CALLS_KEY = "tool_call_limit_blocked"


class ToolCallLimitState(State):
    """工具调用计数和被拦截的调用 id。"""

    #: 线程级计数（跨 invoke 累计）
    thread_tool_call_count: dict[str, int]
    #: 运行级计数（单次 invoke 内）
    run_tool_call_count: dict[str, int]
    #: {limiter 节点名: 本轮拦下的 tool_call id 列表}
    tool_call_limit_blocked: dict[str, list[str]]


def _build_tool_message_content(tool_name: str | None) -> str:
    """给模型看的拦截说明 —— 不提 thread/run，模型不认识这些概念。"""
    if tool_name:
        return f"Tool call limit exceeded. Do not call '{tool_name}' again."
    return "Tool call limit exceeded. Do not make additional tool calls."


def _build_final_ai_message_content(thread_count: int, run_count: int,
                                    thread_limit: int | None,
                                    run_limit: int | None,
                                    tool_name: str | None) -> str:
    """给用户看的收尾说明 —— 这里可以提 thread/run。"""
    tool_desc = f"'{tool_name}' tool" if tool_name else "Tool"
    exceeded = []
    if thread_limit is not None and thread_count > thread_limit:
        exceeded.append(f"thread limit exceeded ({thread_count}/{thread_limit} calls)")
    if run_limit is not None and run_count > run_limit:
        exceeded.append(f"run limit exceeded ({run_count}/{run_limit} calls)")
    limits_text = " and ".join(exceeded)
    return f"{tool_desc} call limit reached: {limits_text}."


def _tool_result_message(blocks: list[dict]) -> dict:
    """
    把 tool_result block 们包成一条 user 消息。

    Anthropic Messages 协议要求：assistant 的 tool_use 之后必须紧跟
    一条 user 消息，content 是 tool_result block 列表 ——
    直接把 block 当消息塞进 messages 会让 API 报错。
    """
    return {"role": "user", "content": blocks}


class ToolCallLimitExceededError(Exception):
    """超限且 exit_behavior="error" 时抛出。"""

    def __init__(self, thread_count: int, run_count: int,
                 thread_limit: int | None, run_limit: int | None,
                 tool_name: str | None = None) -> None:
        self.thread_count = thread_count
        self.run_count = run_count
        self.thread_limit = thread_limit
        self.run_limit = run_limit
        self.tool_name = tool_name
        super().__init__(_build_final_ai_message_content(
            thread_count, run_count, thread_limit, run_limit, tool_name))


class ToolCallLimitMiddleware(Middleware):
    """
    限制工具被调用的次数。

        limiter = ToolCallLimitMiddleware(tool_name="search", run_limit=3)
        agent = create_agent(model, tools=[search], middleware=[limiter])

    超限后的行为：
      - "continue"（默认）：拦下超限的那次调用，给它一条错误结果；
        同批次里其他工具照常执行，agent 继续跑
      - "error"：抛 ToolCallLimitExceededError
      - "end"：立刻收尾 —— 超限的调用和同批次其余待执行调用都补一条
        说明性的 tool_result（不执行），再加一条 assistant 说明
    """

    state_schema = ToolCallLimitState

    def __init__(self, *, tool_name: str | None = None,
                 thread_limit: int | None = None,
                 run_limit: int | None = None,
                 exit_behavior: ExitBehavior = "continue") -> None:
        if thread_limit is None and run_limit is None:
            raise ValueError("至少要指定一个上限（thread_limit 或 run_limit）")
        if exit_behavior not in ("continue", "error", "end"):
            raise ValueError(
                f"exit_behavior 只能是 'continue' / 'error' / 'end'，"
                f"收到 {exit_behavior!r}")
        if (thread_limit is not None and run_limit is not None
                and run_limit > thread_limit):
            raise ValueError(
                f"run_limit ({run_limit}) 不能大于 thread_limit ({thread_limit})："
                "单次运行的上限不该比线程累计上限还高")

        self.tool_name = tool_name
        self.thread_limit = thread_limit
        self.run_limit = run_limit
        self.exit_behavior = exit_behavior

    @property
    def name(self) -> str:
        """带工具名的节点名 —— 多个实例（不同工具）不会撞名。"""
        base = type(self).__name__
        return f"{base}[{self.tool_name}]" if self.tool_name else base

    def _would_exceed(self, thread_count: int, run_count: int) -> bool:
        """再调一次会不会越界。"""
        return ((self.thread_limit is not None and thread_count + 1 > self.thread_limit)
                or (self.run_limit is not None and run_count + 1 > self.run_limit))

    def _matches(self, name: str) -> bool:
        """这次调用归不归这个中间件管。"""
        return self.tool_name is None or name == self.tool_name

    def before_agent(self, state) -> dict:
        """每次 invoke 都重新计算 run 级配额；thread 级计数保留。"""
        return {"run_tool_call_count": {}, _BLOCKED_CALLS_KEY: {}}

    def _clear_blocked(self, state) -> dict | None:
        """本轮不再有这个 limiter 的拦截记录时，清掉自己的那一桶。"""
        blocked_by_limiter = state.get(_BLOCKED_CALLS_KEY) or {}
        if self.name not in blocked_by_limiter:
            return None
        updated = dict(blocked_by_limiter)
        updated.pop(self.name, None)
        return {_BLOCKED_CALLS_KEY: updated}

    @hook_config(can_jump_to=["end"])
    def after_model(self, state) -> dict | None:
        """
        模型返回后计算配额，并记下哪些 tool_call id 必须拦截。

        continue 模式不改写消息：路由器仍正常进入工具节点，
        wrap_tool_call 对被记下的 id 返回错误文本，绝不调用工具本身。
        """
        messages = state.get("messages") or []
        if not messages:
            return self._clear_blocked(state)

        last = messages[-1]
        if last.get("role") != "assistant":
            return self._clear_blocked(state)
        calls = [b for b in (last.get("content") or [])
                 if isinstance(b, dict) and b.get("type") == "tool_use"]
        if not calls:
            return self._clear_blocked(state)

        count_key = self.tool_name or ALL_TOOLS
        thread_counts = dict(state.get("thread_tool_call_count") or {})
        run_counts = dict(state.get("run_tool_call_count") or {})
        thread_count = thread_counts.get(count_key, 0)
        run_count = run_counts.get(count_key, 0)

        # 逐个看：超限之前的都放行，从超限的那个开始都拦下
        allowed, blocked = [], []
        for call in calls:
            if not self._matches(call.get("name", "")):
                continue                      # 不归我管，原样放过
            if self._would_exceed(thread_count, run_count):
                blocked.append(call)
            else:
                allowed.append(call)
                thread_count += 1
                run_count += 1

        if not blocked:
            if not allowed:
                return self._clear_blocked(state)
            thread_counts[count_key] = thread_count
            run_counts[count_key] = run_count
            result = {"thread_tool_call_count": thread_counts,
                      "run_tool_call_count": run_counts}
            cleared = self._clear_blocked(state)
            if cleared:
                result.update(cleared)
            return result

        if any(not call.get("id") for call in blocked):
            raise ValueError("被限流的工具调用必须有 id，否则无法可靠阻止执行")

        if self.exit_behavior == "error":
            raise ToolCallLimitExceededError(
                thread_count=thread_count + len(blocked),
                run_count=run_count,
                thread_limit=self.thread_limit, run_limit=self.run_limit,
                tool_name=self.tool_name)

        blocked_by_limiter = dict(state.get(_BLOCKED_CALLS_KEY) or {})
        blocked_by_limiter[self.name] = [call["id"] for call in blocked]
        thread_counts[count_key] = thread_count
        # 运行级计数把**尝试过**的调用都算上（包括被拦下的）。
        run_counts[count_key] = run_count + len(blocked)

        if self.exit_behavior == "end":
            # 跳到出口 = 工具节点不执行了，在这里直接补齐所有 tool_result。
            content = _build_tool_message_content(self.tool_name)
            result_blocks = [{
                "type": "tool_result",
                "tool_use_id": call["id"],
                "content": content,
                "is_error": True,
            } for call in blocked]
            blocked_ids = {call["id"] for call in blocked}
            pending = [c for c in calls if c.get("id") not in blocked_ids]
            result_blocks.extend({
                "type": "tool_result",
                "tool_use_id": call.get("id"),
                "content": ("Execution stopped before this tool call could run "
                            "because another tool call in the same batch "
                            "exceeded the limit."),
                "is_error": True,
            } for call in pending)

            # 这一批没真执行，不该占线程级配额。
            thread_counts[count_key] = thread_count - len(allowed)
            return {"thread_tool_call_count": thread_counts,
                    "run_tool_call_count": run_counts,
                    _BLOCKED_CALLS_KEY: blocked_by_limiter,
                    "jump_to": "end",
                    "messages": [
                        _tool_result_message(result_blocks),
                        {"role": "assistant",
                         "content": _build_final_ai_message_content(
                             thread_count + len(blocked), run_count,
                             self.thread_limit, self.run_limit, self.tool_name)},
                    ]}

        return {"thread_tool_call_count": thread_counts,
                "run_tool_call_count": run_counts,
                _BLOCKED_CALLS_KEY: blocked_by_limiter}

    def wrap_tool_call(self, request: ToolRequest, handler):
        """continue 模式的真正拦截点：被限流的调用不进入 handler。"""
        blocked_by_limiter = (request.state or {}).get(_BLOCKED_CALLS_KEY) or {}
        if request.tool_call.get("id") in blocked_by_limiter.get(self.name, []):
            return _build_tool_message_content(self.tool_name)
        return handler(request)
