"""Trace minigraph's superstep loop internals for explanation purposes."""
import operator
import sys
from typing import Annotated

sys.path.insert(0, "/Users/zhouquan/project/ai/framework/minigraph")

from minigraph import runtime as rt
from minigraph.builder import StateGraph
from minigraph.constants import END, START
from minigraph.state import State

_orig = rt._prepare_next_tasks


def traced(channels, versions, seen, nodes, trigger_to_nodes=None, updated_channels=None):
    tasks = _orig(channels, versions, seen, nodes, trigger_to_nodes, updated_channels)
    print("  [state before trigger]")
    print("    updated_channels:", sorted(updated_channels) if updated_channels is not None else None)
    print("    versions:", {k: v for k, v in sorted(versions.items())})
    print("    seen     :", {k: dict(sorted(v.items())) for k, v in sorted(seen.items())})
    print("    avail    :", {c: (ch.get() if ch.is_available() else None)
                            for c, ch in sorted(channels.items()) if ch.is_available()})
    print("    -> tasks:", [(t.name, "push" if t.is_push else "pull") for t in tasks])
    return tasks


rt._prepare_next_tasks = traced


class S(State):
    n: Annotated[int, operator.add]
    trace: Annotated[list, operator.add]


g = StateGraph(S)
g.add_node("inc", lambda s: {"n": 1, "trace": ["inc"]})
g.add_node("check", lambda s: {"trace": ["check"]})
g.add_edge(START, "inc")
g.add_edge("inc", "check")
g.add_conditional_edges(
    "check", lambda s: "again" if s["n"] < 3 else "stop", {"again": "inc", "stop": END}
)

print("=== graph ===")
c = g.compile()
print("channels:", sorted(c.channels))
print("nodes   :", {k: (v.triggers, v.writers) for k, v in c.nodes.items()})
print()
out = c.invoke({"n": 0, "trace": []})
print()
print("final:", out)
