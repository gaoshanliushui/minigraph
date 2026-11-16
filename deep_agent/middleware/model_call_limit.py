"""
模型调用次数上限（对齐 LangChain v1 的 middleware/model_call_limit.py）。

两种计数：
    thread_limit  跨 invoke 累计（存在 checkpoint 里，线程级）
    run_limit     单次 invoke 内累计（运行级）

超限后的行为：
    "end"   跳到出口，并补一条说明超限的 assistant 消息
    "error" 抛 ModelCallLimitExceededError

计数发生在 **after_model**（模型已经返回了才算一次调用），
检查发生在 **before_model**（要不要再问一次模型）。
"""

from __future__ import annotations

from typing import Any, Literal

from .core import Middleware, hook_config
from ..state import State

__all__ = [
    "ModelCallLimitState",
    "ModelCallLimitMiddleware",
    "ModelCallLimitExceededError",
]


class ModelCallLimitState(State):
    """模型调用计数。字段名与 LangChain 一致，方便两边对照。"""

    #: 线程级计数（跨 invoke 累计）
    thread_model_call_count: int
    #: 运行级计数（单次 invoke 内）
    run_model_call_count: int


def _build_limit_exceeded_message(thread_count: int, run_count: int,
                                  thread_limit: int | None,
                                  run_limit: int | None) -> str:
    """超限说明文本。"""
    exceeded = []
    if thread_limit is not None and thread_count >= thread_limit:
        exceeded.append(f"thread limit ({thread_count}/{thread_limit})")
    if run_limit is not None and run_count >= run_limit:
        exceeded.append(f"run limit ({run_count}/{run_limit})")
    return f"Model call limits exceeded: {', '.join(exceeded)}"


class ModelCallLimitExceededError(Exception):
    """超限且 exit_behavior="error" 时抛出。"""

    def __init__(self, thread_count: int, run_count: int,
                 thread_limit: int | None, run_limit: int | None) -> None:
        self.thread_count = thread_count
        self.run_count = run_count
        self.thread_limit = thread_limit
        self.run_limit = run_limit
        super().__init__(_build_limit_exceeded_message(
            thread_count, run_count, thread_limit, run_limit))


class ModelCallLimitMiddleware(Middleware):
    """
    限制模型被调用的次数。

        limiter = ModelCallLimitMiddleware(run_limit=5, exit_behavior="end")
        agent = create_agent(model, middleware=[limiter])

    超过上限后：
      - "end"：跳到出口（after_agent 照常跑），并补一条说明超限的消息
      - "error"：抛 ModelCallLimitExceededError
    """

    state_schema = ModelCallLimitState

    def __init__(self, *, thread_limit: int | None = None,
                 run_limit: int | None = None,
                 exit_behavior: Literal["end", "error"] = "end") -> None:
        if thread_limit is None and run_limit is None:
            raise ValueError("至少要指定一个上限（thread_limit 或 run_limit）")
        if exit_behavior not in ("end", "error"):
            raise ValueError(
                f"exit_behavior 只能是 'end' 或 'error'，收到 {exit_behavior!r}")

        self.thread_limit = thread_limit
        self.run_limit = run_limit
        self.exit_behavior = exit_behavior

    def before_agent(self, state) -> dict:
        """每次 invoke 重置运行级计数；线程级计数从 checkpoint 保留。"""
        return {"run_model_call_count": 0}

    @hook_config(can_jump_to=["end"])
    def before_model(self, state) -> dict | None:
        """
        还要不要再问一次模型？

        判断条件是"当前计数已经 >= 上限"：再调一次就越界了，
        所以在**调用前**拦下来。
        """
        thread_count = state.get("thread_model_call_count", 0)
        run_count = state.get("run_model_call_count", 0)

        thread_exceeded = (self.thread_limit is not None
                           and thread_count >= self.thread_limit)
        run_exceeded = (self.run_limit is not None
                        and run_count >= self.run_limit)

        if not (thread_exceeded or run_exceeded):
            return None

        if self.exit_behavior == "error":
            raise ModelCallLimitExceededError(
                thread_count=thread_count, run_count=run_count,
                thread_limit=self.thread_limit, run_limit=self.run_limit)

        message = _build_limit_exceeded_message(
            thread_count, run_count, self.thread_limit, self.run_limit)
        return {
            "jump_to": "end",
            "messages": [{"role": "assistant", "content": message}],
        }

    def after_model(self, state) -> dict | None:
        """模型返回了，计数 +1。"""
        return {
            "thread_model_call_count": state.get("thread_model_call_count", 0) + 1,
            "run_model_call_count": state.get("run_model_call_count", 0) + 1,
        }
