"""
模型降级（对齐 LangChain v1 的 middleware/model_fallback.py）。

主模型失败时依次尝试备用模型。

    fallback = ModelFallbackMiddleware("claude-haiku-4-5", "claude-sonnet-4-5")
    agent = create_agent("claude-opus-4-5", middleware=[fallback])

顺序是：主模型 -> fallback[0] -> fallback[1] -> ...
全部失败就抛最后一次的异常（不吞掉）。

和 LangChain 的差别：那边备用模型是 BaseChatModel 对象，
这边是**模型名字符串**（minigraph 的 ModelRequest.model 就是字符串）。
"""

from __future__ import annotations

from dataclasses import replace

from ..context import GraphInterrupt
from .core import Middleware, ModelRequest

__all__ = ["ModelFallbackMiddleware"]


class ModelFallbackMiddleware(Middleware):
    """
    模型调用失败时自动降级到备用模型。

        ModelFallbackMiddleware(
            "claude-haiku-4-5",      # 第一个备用
            "claude-sonnet-4-5",     # 第二个备用
        )
    """

    def __init__(self, first_model: str, *additional_models: str) -> None:
        self.models = [first_model, *additional_models]

    def wrap_model_call(self, request: ModelRequest, handler):
        """先试主模型，失败就依次换备用模型。"""
        last_exception: Exception | None = None

        try:
            return handler(request)
        except GraphInterrupt:
            raise
        except Exception as exc:
            last_exception = exc

        for model in self.models:
            try:
                return handler(replace(request, model=model))
            except GraphInterrupt:
                raise
            except Exception as exc:
                last_exception = exc
                continue

        raise last_exception
