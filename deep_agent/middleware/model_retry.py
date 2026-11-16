"""
模型调用重试（对齐 LangChain v1 的 middleware/model_retry.py）。

包住模型调用：失败了按指数退避再试，重试耗尽后按 on_failure 决定
是抛出去还是变成一条错误消息让 agent 继续。

    retry = ModelRetryMiddleware(max_retries=3, retry_on=(TimeoutError,))
    agent = create_agent(model, middleware=[retry])

关键语义：
  - **不匹配 retry_on 的异常直接抛** —— 不重试，也不走 on_failure。
    这是刻意的：4xx（key 错了、模型名错了）重试多少次都一样，
    把它变成"模型返回了一条错误消息"只会掩盖真正的配置问题。
  - **GraphInterrupt 不重试** —— 中断是控制流信号，不是失败。
"""

from __future__ import annotations

from typing import Any

from minigraph.context import GraphInterrupt
from .core import Middleware, ModelRequest
from ._retry import (
    OnFailure,
    RetryOn,
    calculate_delay,
    default_retry_on,
    should_retry_exception,
    sleep_before_retry,
    validate_retry_params,
)

__all__ = ["ModelRetryMiddleware"]


class ModelRetryMiddleware(Middleware):
    """
    模型调用失败自动重试。

        ModelRetryMiddleware(
            max_retries=2,           # 首次调用之外再试几次
            retry_on=(TimeoutError,),  # 只重试这些异常
            on_failure="continue",   # 耗尽后：continue / error / 自定义
            backoff_factor=2.0,      # 退避倍数
            initial_delay=1.0,       # 首次重试前等多久（秒）
            max_delay=60.0,          # 退避封顶
            jitter=True,             # 加 ±25% 抖动，避免惊群
        )
    """

    def __init__(self, *, max_retries: int = 2,
                 retry_on: RetryOn = default_retry_on,
                 on_failure: OnFailure = "continue",
                 backoff_factor: float = 2.0,
                 initial_delay: float = 1.0,
                 max_delay: float = 60.0,
                 jitter: bool = True) -> None:
        validate_retry_params(max_retries, initial_delay, max_delay, backoff_factor)

        self.max_retries = max_retries
        self.retry_on = retry_on
        self.on_failure = on_failure
        self.backoff_factor = backoff_factor
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.jitter = jitter

    @staticmethod
    def _format_failure_message(exc: Exception, attempts_made: int) -> str:
        """重试耗尽后的错误文本。"""
        word = "attempt" if attempts_made == 1 else "attempts"
        return (f"Model call failed after {attempts_made} {word} "
                f"with {type(exc).__name__}: {exc}")

    def _handle_failure(self, exc: Exception,
                        attempts_made: int) -> list[dict[str, str]]:
        """重试耗尽：抛出去，或返回与模型 handler 相同的 content blocks。"""
        if self.on_failure == "error":
            raise exc

        if callable(self.on_failure):
            content = self.on_failure(exc)
        else:
            content = self._format_failure_message(exc, attempts_made)

        return [{"type": "text", "text": content}]

    def wrap_model_call(self, request: ModelRequest, handler):
        """包住模型调用：失败就退避重试。"""
        for attempt in range(self.max_retries + 1):
            try:
                return handler(request)
            except GraphInterrupt:
                # 中断是控制流信号，不是失败 —— 原样传出去
                raise
            except Exception as exc:
                if not should_retry_exception(exc, self.retry_on):
                    # 不该重试的异常：直接抛，不走 on_failure
                    raise
                if attempt < self.max_retries:
                    sleep_before_retry(attempt, backoff_factor=self.backoff_factor,
                                      initial_delay=self.initial_delay,
                                      max_delay=self.max_delay, jitter=self.jitter)
                else:
                    return self._handle_failure(exc, attempt + 1)

        raise RuntimeError("重试循环没有返回（不该发生）")
