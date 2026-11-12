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
    恢复时本函数返回 resume 值。

    注意节点会从头重跑，所以 interrupt 之前的副作用必须幂等。

    resume 值按**槽位**存放，一个槽位是一个队列：
      - 多个节点并行挂起：每个节点一个槽位，各拿各的值
      - 一个节点里多次 interrupt()：同一个槽位多个值，按调用顺序出队
    队列空了说明这个挂起点还没被答复 -> 挂起。
    """
    slot = _ctx().get("slot")
    if slot is None:
        raise RuntimeError("interrupt() 只能在图节点内调用")
    queue = _ctx()["resume"].get(slot)
    if queue:
        return queue.pop(0)        # 队首出队
    raise GraphInterrupt(value)