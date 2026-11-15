"""
实测：通道的可用性在什么时候被改。

跑法：
    .venv/bin/python scripts/exp_availability_lifecycle.py

对应 docs/channel通道模型详解.md 的"可用性什么时候被更改"一节。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minigraph.channels import (
    BinaryOperatorAggregate,
    EphemeralValue,
    LastValue,
    NamedBarrierValue,
    Topic,
)
from minigraph.constants import MISSING


def show(tag: str, ch) -> None:
    print(f"    {tag:<42} is_available={ch.is_available()!s:<5}")


def section(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def main() -> None:
    section("① update() 返回 True 不等于 可用性从 False 变 True")
    print("  Topic(accumulate=False)：先清空再收 —— 队列非空才有值")
    t = Topic("x", accumulate=False)
    show("初始", t)
    t.update([1, 2])
    show("update([1,2])  —— 收下", t)
    t.update([3])
    show("update([3])    —— 再次写入（仍是 True）", t)
    print("  ↑ 第二次 update 返回 True，但可用性一直是 True ——")
    print("    它推的是**版本号**，不是可用性翻转。")

    print()
    print("  BinaryOperatorAggregate：带 reducer 的字段，初值就有")
    b = BinaryOperatorAggregate("n", operator_add := (lambda old, new: old + new), int)
    show("初始（typ() = 0）", b)
    b.update([5])
    show("update([5])", b)
    b.update([])
    show("update([]) —— 没人写它", b)
    print("  ↑ update([]) 返回 False，数据通道**永远可用**。")

    section("② EphemeralValue：写 None 也算可用")
    e = EphemeralValue("branch:to:x", guard=False)
    show("初始", e)
    e.update([None])
    show("update([None])  —— 值是 None", e)
    print(f"  ↑ is_available()={e.is_available()}  判的是 value is not MISSING")
    print(f"    MISSING={MISSING!r}  None={None!r}")
    e.update([])
    show("update([])  —— 自清", e)
    e.update([])
    show("再次 update([]) —— 已经是空的", e)
    print("  ↑ 第二次自清返回 False —— 不推版本号。")

    section("③ NamedBarrierValue：可用 = 集齐")
    nb = NamedBarrierValue("join:a+b:c", names={"a", "b"})
    show("初始（没人到）", nb)
    nb.update(["a"])
    show("a 到了（还差 b）", nb)
    nb.update(["b"])
    show("b 到了（齐了）", nb)
    nb.update([])
    show("update([]) —— 半满不会被误伤/满的也不清", nb)
    consumed = nb.consume()
    show(f"consume() -> {consumed}（重新武装）", nb)
    print("  ↑ consume() 是**唯一**把满屏障变回不可用的入口。")
    print(f"    再次 consume() -> {nb.consume()}（没满，返回 False）")

    section("④ 谁能改变可用性？—— 三个入口")
    print("""
  update(values)     写一批值（可能是空写 = 自清）
  consume()          订阅者跑完后（只有 NamedBarrierValue 有实质行为）
  finish()           整图收尾（minigraph 里从没被调用）
""")

    section("⑤ 写同一个值，可用性不变但版本会推进")
    lv = LastValue("plain")
    show("初始", lv)
    lv.update(["x"])
    show("update(['x'])", lv)
    lv.update(["x"])
    show("再次 update(['x']) —— 同值", lv)
    print("  ↑ LastValue.update 永远返回 True ——")
    print("    「内容变了」和「可用性变了」在 LastValue 上无法区分。")


if __name__ == "__main__":
    main()
