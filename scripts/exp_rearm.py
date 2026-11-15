"""
找"只看可用性"真正会出错的图。

思路：要让两个判据分歧，必须构造一个**通道可用、但节点不该再醒**的状态。
关键在于 `EphemeralValue` 每步自清，所以分支通道天然"一步性"。
突破口是 **屏障**（NamedBarrierValue）：它触发后不立刻消失，
而 `consume()` 只是重新武装（清空 seen 集合）。

另外 `update_state` 会人工推版本 —— 这是版本号存在的一个独立理由。

跑法：
    .venv/bin/python scripts/exp_rearm.py
"""
from __future__ import annotations

import operator
import os
import sys
from typing import Annotated

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minigraph import runtime as rt
from minigraph.builder import StateGraph
from minigraph.checkpoint import InMemorySaver
from minigraph.constants import END, START
from minigraph.state import State


class S(State):
    x: Annotated[list, operator.add]


def triggers_avail_only(channels, versions, seen, trig) -> bool:
    for chan in trig:
        ch = channels.get(chan)
        if ch is not None and ch.is_available():
            return True
    return False


# ---------------------------------------------------------------------------
# 场景 1：屏障要能"重新武装"，第二轮汇聚还要能触发 c
# ---------------------------------------------------------------------------

def scenario_barrier_rearm() -> None:
    print("=" * 72)
    print("场景 1：屏障第二轮汇聚")
    print("  a、b 各跑两轮 -> join:a+b:c 凑齐两次 -> c 应该被触发两次")
    print("=" * 72)

    g = StateGraph(S)
    g.add_node("a", lambda s: {"x": ["a"]})
    g.add_node("b", lambda s: {"x": ["b"]})
    g.add_node("c", lambda s: {"x": ["c"]})
    g.add_edge(START, "a")
    g.add_edge(START, "b")
    g.add_edge(["a", "b"], "c")
    g.add_edge("c", END)
    app = g.compile()

    print("c 的触发通道:", app.nodes["c"].triggers)
    for name, step, _ in app.stream({}, {"recursion_limit": 10}):
        print(f"  step {step}: {name}")


# ---------------------------------------------------------------------------
# 场景 2：update_state 之后重跑
# ---------------------------------------------------------------------------

def scenario_update_state() -> None:
    print()
    print("=" * 72)
    print("场景 2：update_state(as_node) 之后重跑")
    print("=" * 72)

    g = StateGraph(S)
    g.add_node("a", lambda s: {"x": ["a"]})
    g.add_node("b", lambda s: {"x": ["b"]})
    g.add_node("c", lambda s: {"x": ["c"]})
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", END)
    app = g.compile(checkpointer=InMemorySaver())
    cfg = {"thread_id": "t"}

    for name, step, _ in app.stream({}, cfg):
        print(f"  step {step}: {name}")

    snap = app.get_state(cfg)
    print("  checkpoint versions_seen:", snap.versions_seen)

    print("  -> update_state(as_node='a')")
    app.update_state(cfg, {"x": ["injected"]}, as_node="a")
    snap = app.get_state(cfg)
    print("  注入后 versions_seen :", snap.versions_seen)
    print("  注入后 channel_versions:", snap.channel_versions)
    print("  -> 重跑：")
    for name, step, _ in app.stream(None, cfg):
        print(f"     step {step}: {name}")


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    orig = rt._triggers

    if which in ("both", "normal"):
        print("### 正常（版本+水位线）")
        scenario_barrier_rearm()
        scenario_update_state()

    if which in ("both", "avail"):
        rt._triggers = triggers_avail_only
        print()
        print("### 只看可用性")
        scenario_barrier_rearm()
        scenario_update_state()
        rt._triggers = orig


if __name__ == "__main__":
    main()
