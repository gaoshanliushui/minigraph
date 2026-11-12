"""
逐个通道类型实测 is_available() 的生命周期。

对应文档：docs/channel通道模型详解.md

跑法：
    .venv/bin/python scripts/channel_availability.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minigraph.channels import (
    BaseChannel,
    BinaryOperatorAggregate,
    EmptyChannelError,
    EphemeralValue,
    LastValue,
    NamedBarrierValue,
    Topic,
)
from minigraph.constants import MISSING


def show(tag: str, ch) -> None:
    try:
        val = ch.get()
    except EmptyChannelError:
        val = "<空>"
    print(f"    {tag:34s} is_available={str(ch.is_available()):5s}  值={val!r}")


class Plain(BaseChannel):
    """演示 BaseChannel 的默认 is_available()：get() 不抛异常就算可用。"""

    def __init__(self, key: str = "k"):
        super().__init__(key)
        self.value = MISSING

    def update(self, values):
        if not values:
            return False
        self.value = values[-1]
        return True

    def get(self):
        if self.value is MISSING:
            raise EmptyChannelError()
        return self.value


def main() -> None:
    print("=== LastValue（无 reducer 的状态字段）===")
    c = LastValue("k")
    show("初始", c)
    c.update(["x"])
    show("update(['x'])", c)
    c.update([])
    show("update([])  —— 没人写它", c)
    c.update(["y"])
    show("update(['y'])", c)

    print("\n=== BinaryOperatorAggregate（有 reducer 的状态字段）===")
    c = BinaryOperatorAggregate("k", lambda old, new: old + new)
    show("初始", c)
    c.update([1])
    show("update([1])  —— 首次直接落盘", c)
    c.update([2])
    show("update([2])  —— reducer 合并", c)
    c.update([])
    show("update([])  —— 没人写它", c)

    print("\n=== EphemeralValue（分支触发通道 branch:to:X）===")
    c = EphemeralValue("k", guard=False)
    show("初始", c)
    c.update([None])           # 分支通道的值通常是 None，意义在"被写过"
    show("update([None])  —— 值是 None", c)
    c.update([])
    show("update([])  —— 自清", c)
    c.update([None])
    show("再次 update([None])", c)

    print("\n=== Topic（__pregel_tasks 的 Send 队列）===")
    c = Topic("k", accumulate=False)
    show("初始", c)
    c.update([1, 2])
    show("update([1,2])", c)
    c.update([])
    show("update([])  —— 自清（只留本步的值）", c)
    c2 = Topic("k2", accumulate=True)
    c2.update([1])
    c2.update([2])
    show("accumulate=True 写两次", c2)

    print("\n=== NamedBarrierValue（多源汇聚 join:a+b:c）===")
    c = NamedBarrierValue("k", names={"a", "b"})
    show("初始（没人到）", c)
    c.update(["a"])
    show("a 到了（还差 b）", c)
    c.update(["b"])
    show("b 到了（齐了）", c)
    print(f"    consume() -> {c.consume()}")
    show("consume() 之后（重新武装）", c)

    print("\n=== BaseChannel 默认实现（is_available 走 get()）===")
    c = Plain()
    show("初始", c)
    c.update(["v"])
    show("update(['v'])", c)


if __name__ == "__main__":
    main()
