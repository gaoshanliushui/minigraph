"""
实测 `versions` 与 `seen` 两张表在运行中怎么变。

对应文档：docs/versions与seen水位线详解.md

跑法：
    .venv/bin/python scripts/versions_seen_trace.py

这个脚本跑四个场景，每个场景都打印每超步的
versions / seen / avail 快照：

    1. 线性链 a -> b            —— 通道自清 + 节点水位线
    2. 回边 agent -> tools -> agent —— 水位线落后一格，循环才转得起来
    3. 屏障 a+b -> c            —— 屏障凑齐才触发，触发后重新武装
    4. Send 扇出                 —— PUSH 不推进目标节点的水位线
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import operator
from typing import Annotated

from minigraph.builder import StateGraph
from minigraph.control import Send
from minigraph.state import State


# ---------------------------------------------------------------------------
# 探针：把每个超步的 versions / seen / avail 打出来
# ---------------------------------------------------------------------------

def instrument() -> None:
    """给 _RuntimeState 打上探针，把完整的 seen 也打出来。"""
    from minigraph.runtime import _RuntimeState

    original = _RuntimeState.dump

    def dump(self, tag: str) -> None:
        original(self, tag)
        print(f"      [全部 seen] {self.seen}")

    _RuntimeState.dump = dump


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------------
# 场景 1：线性链
# ---------------------------------------------------------------------------

class S1(State):
    x: Annotated[list, operator.add]


def scenario_linear() -> None:
    section("场景 1：线性链 a -> b")
    print("预期：a 跑完写 branch:to:b（版本推进），branch:to:a 没人写 -> 自清")
    print("      b 从没跑过（seen 里没有 b）-> 只要通道可用就触发")

    g = StateGraph(S1)
    g.add_node("a", lambda s: {"x": ["a"]})
    g.add_node("b", lambda s: {"x": ["b"]})
    g.add_edge("__start__", "a")
    g.add_edge("a", "b")
    app = g.compile()
    for _ in app.stream({}, {"debug": True}):
        pass


# ---------------------------------------------------------------------------
# 场景 2：回边循环
# ---------------------------------------------------------------------------

class S2(State):
    x: Annotated[list, operator.add]


def scenario_loop() -> None:
    section("场景 2：回边 agent -> tools -> agent")
    print("预期：seen[agent] 的水位线总是比 versions[branch:to:agent] 落后一格")
    print("      —— 这个'落后一格'就是循环的燃料")

    def agent(s):
        return {"x": ["agent"]}

    def tools(s):
        return {"x": ["tools"]}

    g = StateGraph(S2)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.add_edge("__start__", "agent")
    g.add_edge("agent", "tools")
    g.add_edge("tools", "agent")
    app = g.compile()
    # 这个图会一直转下去 —— 正好用来看水位线怎么"落后一格"。
    # recursion_limit 撞顶会抛异常，那是预期的退出方式。
    try:
        for _ in app.stream({}, {"debug": True, "recursion_limit": 6}):
            pass
    except RuntimeError as exc:
        print(f"  (按预期撞上 recursion_limit: {exc})")


# ---------------------------------------------------------------------------
# 场景 3：屏障
# ---------------------------------------------------------------------------

class S3(State):
    x: Annotated[list, operator.add]


def scenario_barrier() -> None:
    section("场景 3：屏障 add_edge(['a','b'], 'c')")
    print("预期：join:a+b:c 要两个源都到齐才可用；c 触发后屏障重新武装")

    g = StateGraph(S3)
    g.add_node("a", lambda s: {"x": ["a"]})
    g.add_node("b", lambda s: {"x": ["b"]})
    g.add_node("c", lambda s: {"x": ["c"]})
    g.add_edge("__start__", "a")
    g.add_edge("__start__", "b")
    g.add_edge(["a", "b"], "c")
    app = g.compile()
    print("c 的触发通道:", app.nodes["c"].triggers)
    for _ in app.stream({}, {"debug": True}):
        pass


# ---------------------------------------------------------------------------
# 场景 4：Send 扇出
# ---------------------------------------------------------------------------

class S4(State):
    x: Annotated[list, operator.add]


def scenario_send() -> None:
    section("场景 4：Send 扇出（PUSH 轨道）")
    print("预期：seen['worker'] 永远是空字典 {} —— PUSH 不推进目标节点的水位线")
    print("      branch:to:worker 的版本始终是 0 —— PUSH 绕开分支通道")

    def fan(s):
        return [Send("worker", {"x": "w1"}), Send("worker", {"x": "w2"})]

    def worker(s):
        return {"x": [s.get("x", "")]}

    g = StateGraph(S4)
    g.add_node("fan", fan)
    g.add_node("worker", worker)
    g.add_edge("__start__", "fan")
    g.add_edge("fan", "worker")
    app = g.compile()
    for _ in app.stream({}, {"debug": True}):
        pass


# ---------------------------------------------------------------------------
# 不变式断言
# ---------------------------------------------------------------------------

def check_invariants() -> None:
    """跑完各场景后检查文档里声称的不变式。"""
    from minigraph.runtime import _RuntimeState

    section("不变式检查")

    # 不变式 1：versions 单调递增
    # 不变式 2：seen[node][chan] <= versions[chan]
    # 不变式 3：循环里水位线落后一格（燃料存在）
    seen_hist: list[dict] = []
    versions_hist: list[dict] = []

    original = _RuntimeState.dump

    def dump(self, tag: str) -> None:
        original(self, tag)
        seen_hist.append({k: dict(v) for k, v in self.seen.items()})
        versions_hist.append(dict(self.versions))

    _RuntimeState.dump = dump

    g = StateGraph(S2)
    g.add_node("agent", lambda s: {"x": ["a"]})
    g.add_node("tools", lambda s: {"x": ["t"]})
    g.add_edge("__start__", "agent")
    g.add_edge("agent", "tools")
    g.add_edge("tools", "agent")
    app = g.compile()
    try:
        for _ in app.stream({}, {"recursion_limit": 6, "debug": True}):
            pass
    except RuntimeError:
        pass   # 故意让它撞上限：我们要的就是循环里水位线落后一格的样本

    _RuntimeState.dump = original

    # 检查 1：versions 单调（逐通道）
    ok = True
    for chan in versions_hist[0]:
        seq = [v.get(chan, 0) for v in versions_hist]
        if seq != sorted(seq):
            ok = False
            print(f"  ✗ 通道 {chan} 的版本不是单调递增: {seq}")
    print(f"  {'✓' if ok else '✗'} versions 每个通道单调递增")

    # 检查 2：seen <= versions
    ok = True
    for snap_v, snap_s in zip(versions_hist, seen_hist):
        for node, node_seen in snap_s.items():
            for chan, val in node_seen.items():
                if val > snap_v.get(chan, 0):
                    ok = False
                    print(f"  ✗ {node}.{chan} 水位线 {val} > 版本 {snap_v.get(chan)}")
    print(f"  {'✓' if ok else '✗'} seen[node][chan] <= versions[chan]")

    # 检查 3：循环里水位线落后一格
    ok = True
    for snap_v, snap_s in zip(versions_hist, seen_hist):
        node_seen = snap_s.get("agent", {})
        chan = "branch:to:agent"
        if chan in node_seen and node_seen[chan] >= snap_v.get(chan, 0):
            ok = False
    print(f"  {'✓' if ok else '✗'} 循环里 agent 的水位线落后于版本（燃料存在）")


def main() -> None:
    instrument()
    scenario_linear()
    scenario_loop()
    scenario_barrier()
    scenario_send()
    check_invariants()
    print()


if __name__ == "__main__":
    main()
