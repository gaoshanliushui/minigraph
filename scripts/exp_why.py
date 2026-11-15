"""
为什么"只看可用性"和"版本+水位线"结果一样？

假设：因为 EphemeralValue 每步自清，"可用"⟺"本步被写过"，
      而"本步被写过"⟺"版本比水位线新"。两者恒等。

这个脚本直接检验这个等价关系：跑一批图，
在**每一次算触发时**逐节点比对两个判据的结论。

跑法：
    .venv/bin/python scripts/exp_why.py
"""
from __future__ import annotations

import operator
import os
import sys
from typing import Annotated

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minigraph import runtime as rt
from minigraph.builder import StateGraph
from minigraph.constants import END, START
from minigraph.control import Command, Send
from minigraph.state import State

MISMATCHES = []


class S(State):
    x: Annotated[list, operator.add]


def probes(channels, versions, seen, trig) -> bool:
    """原判据，同时记录它和"只看可用性"是否一致。"""
    # 只看可用性的结论
    avail_only = False
    for chan in trig:
        ch = channels.get(chan)
        if ch is not None and ch.is_available():
            avail_only = True
            break

    # 原判据的结论
    if seen is None:
        real = False
        for chan in trig:
            ch = channels.get(chan)
            if ch is not None and ch.is_available():
                real = True
                break
    else:
        real = False
        for chan in trig:
            ch = channels.get(chan)
            if ch is None or not ch.is_available():
                continue
            if versions.get(chan, 0) > seen.get(chan, 0):
                real = True
                break

    if real != avail_only:
        MISMATCHES.append((dict(versions), seen, trig, real, avail_only))

    return real


def build(kind: str):
    g = StateGraph(S)
    if kind == "chain":
        g.add_node("a", lambda s: {"x": ["a"]})
        g.add_node("b", lambda s: {"x": ["b"]})
        g.add_edge(START, "a")
        g.add_edge("a", "b")
        g.add_edge("b", END)
    elif kind == "barrier":
        g.add_node("a", lambda s: {"x": ["a"]})
        g.add_node("b", lambda s: {"x": ["b"]})
        g.add_node("c", lambda s: {"x": ["c"]})
        g.add_edge(START, "a")
        g.add_edge(START, "b")
        g.add_edge(["a", "b"], "c")
        g.add_edge("c", END)
    elif kind == "send":
        g.add_node("fan", lambda s: [Send("w", {"x": "1"}), Send("w", {"x": "2"})])
        g.add_node("w", lambda s: {"x": [s.get("x", "?")]})
        g.add_edge(START, "fan")
        g.add_edge("w", END)
    elif kind == "loop":
        g.add_node("agent", lambda s: {"x": ["a"]})
        g.add_node("tools", lambda s: {"x": ["t"]})
        g.add_edge(START, "agent")
        g.add_edge("agent", "tools")
        g.add_edge("tools", "agent")
    elif kind == "fanin":
        g.add_node("a", lambda s: Command(goto="c", update={"x": ["a"]}))
        g.add_node("b", lambda s: Command(goto="c", update={"x": ["b"]}))
        g.add_node("c", lambda s: {"x": ["c"]})
        g.add_edge(START, "a")
        g.add_edge(START, "b")
        g.add_edge("c", END)
    return g.compile()


def main() -> None:
    rt._triggers = probes
    kinds = ["chain", "barrier", "send", "loop", "fanin"]
    for kind in kinds:
        app = build(kind)
        try:
            for _ in app.stream({}, {"recursion_limit": 8}):
                pass
        except RuntimeError:
            pass

    print(f"共检查 {len(kinds)} 种图")
    if MISMATCHES:
        print(f"★ 发现 {len(MISMATCHES)} 处判据不一致：")
        for versions, seen, trig, real, avail in MISMATCHES[:10]:
            print(f"  trig={trig} seen={seen}")
            print(f"    versions={versions}")
            print(f"    原判据={real}  只看可用性={avail}")
    else:
        print("✓ 没有任何不一致 —— 两种判据在所有图上完全等价")


if __name__ == "__main__":
    main()
