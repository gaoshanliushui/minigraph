"""
dump 一张代表性图的完整通道清单（编译产物）。

对应文档：docs/channel通道模型详解.md §2

跑法：
    .venv/bin/python scripts/channel_inventory.py
"""
from __future__ import annotations

import operator
import os
import sys
from typing import Annotated

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minigraph.builder import StateGraph
from minigraph.constants import END, START
from minigraph.state import State, add_messages

ROLE = {
    "__start__": "触发通道（入口伪节点占位）",
    "__pregel_tasks": "触发通道（PUSH 轨 / Send 队列）",
}


def role_of(key: str) -> str:
    if key in ROLE:
        return ROLE[key]
    if key.startswith("branch:to:"):
        return "触发通道（分支信号）"
    if key.startswith("join:"):
        return "触发通道（多源汇聚屏障）"
    return "数据通道（状态字段）"


class S(State):
    n: Annotated[int, operator.add]       # 有 reducer -> BinaryOperatorAggregate
    msgs: Annotated[list, add_messages]   # 有 reducer -> BinaryOperatorAggregate
    plain: str                            # 无 reducer -> LastValue


def build() -> StateGraph:
    g = StateGraph(S)
    g.add_node("a", lambda s: {"n": 1})
    g.add_node("b", lambda s: {"msgs": [{"id": 1}]})
    g.add_node("c", lambda s: {"plain": "c"})
    g.add_node("d", lambda s: {"n": 1})
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge(["b", "c"], "d")            # 屏障边 -> join:b+c:d
    g.add_conditional_edges("d", lambda s: "a" if s["n"] < 3 else END,
                            {"a": "a", END: END})
    g.add_edge("d", END)
    return g


def main() -> None:
    app = build().compile()

    print("通道表")
    print("-" * 96)
    print(f"{'通道名':28s} {'类型':26s} {'角色'}")
    for key, ch in app.channels.items():
        print(f"{key:28s} {type(ch).__name__:26s} {role_of(key)}")

    print("\n节点订阅 / 写入关系")
    print("-" * 96)
    for name, spec in app.nodes.items():
        print(f"{name:10s} triggers={spec.triggers}  writers={spec.writers}"
              f"  branches={len(spec.branches)}  is_start={spec.is_start}")

    print("\n反向索引 trigger_to_nodes")
    print("-" * 96)
    for chan, names in app.trigger_to_nodes.items():
        print(f"  {chan:28s} -> {names}")

    print("\n开局时每个通道的状态")
    print("-" * 96)
    for key, ch in app.channels.items():
        print(f"{key:28s} available={str(ch.is_available()):5s} checkpoint={ch.checkpoint()!r}")


if __name__ == "__main__":
    main()
