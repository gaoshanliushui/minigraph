"""
对比两种触发判据在各种图上的行为差异。

  正常：  通道可用 且  versions[chan] > seen[node][chan]
  实验：  只看通道可用性

用途：为 docs/versions与seen水位线详解.md 的「为什么不能只看可用性」一节
提供**实测证据** —— 差异到底出现在哪种图上。

不改项目源码 —— 运行时 monkeypatch `runtime._triggers`。

跑法：
    .venv/bin/python scripts/exp_compare_triggers.py
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
from minigraph.control import Command
from minigraph.state import State


class S(State):
    x: Annotated[list, operator.add]


def build(kind: str):
    g = StateGraph(S)
    if kind == "chain":
        # a -> b -> c
        g.add_node("a", lambda s: {"x": ["a"]})
        g.add_node("b", lambda s: {"x": ["b"]})
        g.add_node("c", lambda s: {"x": ["c"]})
        g.add_edge(START, "a")
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.add_edge("c", END)

    elif kind == "barrier":
        # 屏障：a+b -> c，c 只能被触发一次
        g.add_node("a", lambda s: {"x": ["a"]})
        g.add_node("b", lambda s: {"x": ["b"]})
        g.add_node("c", lambda s: {"x": ["c"]})
        g.add_edge(START, "a")
        g.add_edge(START, "b")
        g.add_edge(["a", "b"], "c")
        g.add_edge("c", END)

    elif kind == "parallel_fanin":
        # a、b 并行，都 goto c
        g.add_node("a", lambda s: Command(goto="c", update={"x": ["a"]}))
        g.add_node("b", lambda s: Command(goto="c", update={"x": ["b"]}))
        g.add_node("c", lambda s: {"x": ["c"]})
        g.add_edge(START, "a")
        g.add_edge(START, "b")
        g.add_edge("c", END)

    elif kind == "self_loop":
        g.add_node("c", lambda s: {"x": ["c"]})
        g.add_edge(START, "c")
        g.add_edge("c", "c")

    return g.compile()


def run(kind: str):
    app = build(kind)
    steps = []
    try:
        for name, step, _ in app.stream({}, {"recursion_limit": 10}):
            steps.append((step, name))
    except RuntimeError as exc:
        steps.append(("ERR", str(exc)[:30]))
    return steps


KINDS = ["chain", "barrier", "parallel_fanin", "self_loop"]


def triggers_avail_only(channels, versions, seen, trig) -> bool:
    """只判可用性 —— 不看版本，不管水位线。"""
    for chan in trig:
        ch = channels.get(chan)
        if ch is not None and ch.is_available():
            return True
    return False


def main() -> None:
    original = rt._triggers

    normal = {k: run(k) for k in KINDS}
    rt._triggers = triggers_avail_only
    avail = {k: run(k) for k in KINDS}
    rt._triggers = original

    print("图类型            正常（版本+水位线）        只看可用性              差异")
    print("-" * 78)
    for k in KINDS:
        same = "同" if normal[k] == avail[k] else "★ 不同 ★"
        print(f"{k:<17} {str(normal[k]):<25} {str(avail[k]):<24} {same}")


if __name__ == "__main__":
    main()
