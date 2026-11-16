"""
工具调用重试（对齐 LangChain v1 的 middleware/tool_retry.py）。

和 model_retry 是同一套重试语义，只是包的是**工具**调用，
而且可以只对某些工具生效（tools=["search"]）。

    retry = ToolRetryMiddleware(max_retries=2, tools=["search"])
    agent = create_agent(model, tools=[search, write], middleware=[retry])

重试耗尽后的产物是工具结果文本，不是抛异常 —— 模型下一轮能看到
"这个工具失败了"，有机会改个参数再试。
"""

from __future__ import annotations

from ..context import GraphInterrupt
from .core import Middleware, ToolRequest
from ._retry import (
    OnFailure,
    RetryOn,
    default_retry_on,
    should_retry_exception,
    sleep_before_retry,
    validate_retry_params,
)

__all__ = ["ToolRetryMiddleware"]


class ToolRetryMiddleware(Middleware):
    """
    工具调用失败自动重试。

        ToolRetryMiddleware(
            max_retries=2,
            tools=["search"],        # 只对这些工具有效（None = 全部）
            retry_on=(ConnectionError,),
            on_failure="continue",   # continue / error / 自定义
            backoff_factor=2.0,
            initial_delay=1.0,
            max_delay=60.0,
            jitter=True,
        )
    """

    def __init__(self, *, max_retries: int = 2,
                 tools: list | None = None,
                 retry_on: RetryOn = default_retry_on,
                 on_failure: OnFailure = "continue",
                 backoff_factor: float = 2.0,
                 initial_delay: float = 1.0,
                 max_delay: float = 60.0,
                 jitter: bool = True) -> None:
        validate_retry_params(max_retries, initial_delay, max_delay, backoff_factor)

        self.max_retries = max_retries
        # 工具可以是名字，也可以是 Tool 对象（取它的 .name）
        self._tool_filter: list[str] | None = (
            None if tools is None
            else [t if isinstance(t, str) else getattr(t, "name", str(t))
                  for t in tools]
        )
        self.retry_on = retry_on
        self.on_failure = on_failure
        self.backoff_factor = backoff_factor
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.jitter = jitter

    def _should_retry_tool(self, tool_name: str) -> bool:
        """这次工具调用归不归这个中间件管。"""
        if self._tool_filter is None:
            return True
        return tool_name in self._tool_filter

    @staticmethod
    def _format_failure_message(tool_name: str, exc: Exception,
                                attempts_made: int) -> str:
        """重试耗尽后的错误文本。"""
        word = "attempt" if attempts_made == 1 else "attempts"
        return (f"Tool '{tool_name}' failed after {attempts_made} {word} "
                f"with {type(exc).__name__}: {exc}. Please try again.")

    def wrap_tool_call(self, request: ToolRequest, handler):
        """包住工具调用：失败就退避重试。"""
        tool_name = request.name
        if not self._should_retry_tool(tool_name):
            return handler(request)

        for attempt in range(self.max_retries + 1):
            try:
                return handler(request)
            except GraphInterrupt:
                # 中断是控制流信号，不是失败 —— 原样传出去
                raise
            except Exception as exc:
                if not should_retry_exception(exc, self.retry_on):
                    raise
                if attempt < self.max_retries:
                    sleep_before_retry(attempt, backoff_factor=self.backoff_factor,
                                      initial_delay=self.initial_delay,
                                      max_delay=self.max_delay, jitter=self.jitter)
                else:
                    if self.on_failure == "error":
                        raise exc
                    if callable(self.on_failure):
                        content = self.on_failure(exc)
                    else:
                        content = self._format_failure_message(
                            tool_name, exc, attempt + 1)
                    # 工具节点会把返回值放进 tool_result.content；这里返回
                    # 文本，避免产出一个嵌套的、协议不认识的 tool_result dict。
                    return content

        raise RuntimeError("重试循环没有返回（不该发生）")
