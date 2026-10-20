"""
§0 续 + §2 interrupt：thread-local 上下文 + 中断机制

每个 invoke / resume 共享一份上下文，供 interrupt() 读取 resume 值。
"""

from __future__ import annotations

import threading
from typing import Any

_CTX = threading.local()


def _ctx() -> dict[str, Any]:
    """每次运行（一次 invoke / resume）一份上下文，供 interrupt() 读取。"""
    if not hasattr(_CTX, "data"):
        _CTX.data = {"resume": {}, "slot": None}
    return _CTX.data


class GraphInterrupt(Exception):
    def __init__(self, value: Any):
        self.value = value
        super().__init__(f"interrupt: {value!r}")


def interrupt(value: Any) -> Any:
    """
    节点内调用即暂停整张图，value 作为"请求"暴露给调用方。
    恢复时本函数返回 Command(resume=...) 传入的值；
    注意节点会从头重跑，所以 interrupt 之前的副作用必须幂等。
    """
    slot = _ctx().get("slot")
    if slot is None:
        raise RuntimeError("interrupt() 只能在图节点内调用")
    store = _ctx()["resume"]
    if slot in store:
        return store.pop(slot)
    raise GraphInterrupt(value)