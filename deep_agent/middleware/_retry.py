"""
重试中间件的公共部件（对齐 LangChain v1 的 middleware/_retry.py）。

model_retry 和 tool_retry 用的是**同一套**重试语义：
    retry_on     哪些异常该重试
    on_failure   重试耗尽后怎么办
    backoff      指数退避 + 抖动

所以公共逻辑放这里，两个中间件只各自负责"包模型"还是"包工具"。
"""

from __future__ import annotations

import random
import time
from typing import Callable, Literal

#: retry_on 的两种形态：
#:   - 异常类型元组（按 isinstance 判断）
#:   - 可调用对象（收到异常返回 True 才重试）
RetryOn = tuple[type[Exception], ...] | Callable[[Exception], bool]

#: on_failure 的三种形态：
#:   "error"    抛出原异常，agent 停住
#:   "continue" 把错误变成一条消息，agent 继续跑
#:   可调用对象  收到异常返回一段错误文本（自定义措辞）
OnFailure = Literal["error", "continue"] | Callable[[Exception], str]


def validate_retry_params(max_retries: int, initial_delay: float,
                          max_delay: float, backoff_factor: float) -> None:
    """参数校验。负数一律拒绝 —— 负的退避会变成"越失败越快重试"。"""
    if max_retries < 0:
        raise ValueError("max_retries 必须 >= 0")
    if initial_delay < 0:
        raise ValueError("initial_delay 必须 >= 0")
    if max_delay < 0:
        raise ValueError("max_delay 必须 >= 0")
    if backoff_factor < 0:
        raise ValueError("backoff_factor 必须 >= 0")


def default_retry_on(exc: BaseException) -> bool:
    """
    默认可重试判定。

    BaseException 的子类（KeyboardInterrupt、SystemExit）**不重试** ——
    它们是控制流信号，不是"这次调用失败了"。
    """
    if not isinstance(exc, Exception):
        return False
    return True


def should_retry_exception(exc: Exception, retry_on: RetryOn) -> bool:
    """这个异常该不该重试。"""
    if callable(retry_on):
        return retry_on(exc)
    return isinstance(exc, retry_on)


def calculate_delay(retry_number: int, *, backoff_factor: float,
                    initial_delay: float, max_delay: float,
                    jitter: bool) -> float:
    """
    第 n 次重试前该等多久（秒）。

        delay = initial_delay * backoff_factor ** n     （backoff_factor=0 则恒定）
        delay = min(delay, max_delay)                    （封顶）
        delay += 随机(-25%, +25%)                          （抖动，避免惊群）

    jitter 是必要的：多个 worker 同时失败时，没有抖动它们会同时醒来、
    同时失败、同时再失败 —— 退避也就白设了。
    """
    if backoff_factor == 0.0:
        delay = initial_delay
    else:
        delay = initial_delay * (backoff_factor ** retry_number)

    delay = min(delay, max_delay)

    if jitter and delay > 0:
        amount = delay * 0.25
        delay += random.uniform(-amount, amount)
        delay = max(0, delay)

    return delay


def sleep_before_retry(retry_number: int, *, backoff_factor: float,
                       initial_delay: float, max_delay: float,
                       jitter: bool) -> None:
    """算出退避时间并真的睡一会儿。"""
    delay = calculate_delay(retry_number, backoff_factor=backoff_factor,
                            initial_delay=initial_delay, max_delay=max_delay,
                            jitter=jitter)
    if delay > 0:
        time.sleep(delay)
