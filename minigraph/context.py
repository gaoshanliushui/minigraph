"""
§0 续 + §2 interrupt：thread-local 上下文 + 中断机制 + 流式写出

每个 invoke / resume 共享一份上下文，供 interrupt() 读取 resume 值、
供 get_stream_writer() 写出自定义数据。
"""

from __future__ import annotations

import itertools
import threading
from typing import Any, Callable

_CTX = threading.local()


def _ctx() -> dict[str, Any]:
    """每次运行（一次 invoke / resume）一份上下文，供 interrupt() 读取。"""
    if not hasattr(_CTX, "data"):
        _CTX.data = {
            "resume": {},
            "slot": None,
            "interrupt_counter": None,
            "stream_writer": None,
        }
    return _CTX.data


def _noop_writer(_: Any) -> None:
    """没开 stream_mode="custom" 时的占位 writer：写出去没人收，直接丢。"""


def get_stream_writer() -> Callable[[Any], None]:
    """节点内调用：把任意数据写到 stream(stream_mode="custom") 的消费者。

    对齐 LangGraph 的 get_stream_writer()。模型适配器靠它把每个 token
    实时送出去；节点跑在别的线程（线程池 / 后台线程）里也能用 ——
    writer 由运行时注入到当前线程的上下文。

    没在 custom 模式下调用时是 no-op（和 LangGraph 一致）：节点不用
    关心调用方开没开流式，写就是了。
    """
    writer = _ctx().get("stream_writer")
    return writer if writer is not None else _noop_writer


class GraphInterrupt(Exception):
    def __init__(self, value: Any):
        self.value = value
        super().__init__(f"interrupt: {value!r}")


def interrupt(value: Any) -> Any:
    """
    节点内调用即暂停整张图，value 作为"请求"暴露给调用方。
    恢复时本函数返回 resume 值。

    注意节点会从头重跑，所以 interrupt 之前的副作用必须幂等。

    但**兄弟任务**不用为这件事操心：同一超步里别人挂起时，
    已经跑成功的那些任务不会重跑 —— 它们的写入被存进 checkpoint，
    恢复时直接用掉（对齐 LangGraph 的 `_reapply_writes_to_succeeded_nodes`）。
    也就是说，只有**调用 interrupt() 的那个节点**需要幂等。

    resume 值按**槽位**存放，一个槽位对应一个列表：
      - 多个节点并行挂起：每个节点一个槽位，各拿各的值
      - 一个任务里多次 interrupt()：按任务内的调用序号读取 resume[idx]

    列表只读、不按索引消费。节点挂起后会从头重跑，计数器也从零开始，
    于是前几个 interrupt() 会依次重放已经收到的答复；走到第一个还没有
    答复的序号才再次挂起。对齐 LangGraph 的 scratchpad.interrupt_counter()
    + scratchpad.resume[idx]。工具线程池里的子调用共享父图任务的计数器和
    答复列表，因此作用域仍是整个图任务，而不是各自独立的小任务。
    """
    ctx = _ctx()
    slot = ctx.get("slot")
    if slot is None:
        raise RuntimeError("interrupt() 只能在图节点内调用")
    counter = ctx.get("interrupt_counter")
    if counter is None:
        # 兼容直接把 slot/resume 注入上下文的调用方；正常图任务由运行时
        # 在 _run_task 里创建计数器。
        counter = itertools.count()
        ctx["interrupt_counter"] = counter
    idx = next(counter)
    resumes = ctx["resume"].get(slot) or []
    if idx < len(resumes):
        return resumes[idx]
    raise GraphInterrupt(value)