"""
差分测试的执行与比对逻辑。

比对三个维度（从粗到细）：

  1. **超步序列**   每个超步跑了哪些节点（并行节点按集合比，
                   因为完成顺序不确定；超步的**先后**严格比）
  2. **超步总数**   必须完全一致 —— 这是触发模型是否对齐的最强信号
  3. **最终状态**   完整状态字典必须一致

出错时两边都该怎么报错的情况，比对的是"是否都抛异常"，
而不是异常消息（两边的措辞本来就不同）。
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from .backends import HAS_LANGGRAPH, LANGGRAPH, MINIGRAPH
from .scenarios import SCENARIOS, Scenario


class _NodeRecorder:
    """直接记录节点函数调用，避免把流事件的产出时机误当成执行时机。"""

    def __init__(self):
        self.counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def wrap(self, name: str, fn):
        def wrapped(state):
            with self._lock:
                self.counts[name] = self.counts.get(name, 0) + 1
            return fn(state)

        return wrapped

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self.counts)


class _RecordingGraph:
    """StateGraph 的透明代理，只在 add_node 时包一层计数器。"""

    def __init__(self, graph, recorder: _NodeRecorder):
        self._graph = graph
        self._recorder = recorder

    def add_node(self, name, action, **options):
        if isinstance(action, Mapping):
            spec = dict(action)
            spec["fn"] = self._recorder.wrap(name, spec["fn"])
            self._graph.add_node(name, spec, **options)
        else:
            self._graph.add_node(name, self._recorder.wrap(name, action), **options)
        return self

    def __getattr__(self, name):
        return getattr(self._graph, name)


@dataclass
class RunResult:
    """一次运行的观测结果。"""

    steps: list[frozenset[str]] = field(default_factory=list)   # 每个超步的节点集合
    task_counts: dict[str, int] = field(default_factory=dict)   # 报错前的执行轨迹
    final: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    error_type: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _run_minigraph(scenario: Scenario) -> RunResult:
    recorder = _NodeRecorder()
    app = _rebuild(scenario, MINIGRAPH, recorder).compile()
    res = RunResult()
    try:
        for node, _step, _upd in app.stream(scenario.input, scenario.config):
            res.steps.append(frozenset(n for n in node.split(",") if n))
        res.task_counts = recorder.snapshot()
        res.final = dict(app.invoke(scenario.input, scenario.config))
    except Exception as exc:  # noqa: BLE001
        res.task_counts = recorder.snapshot()
        res.error = str(exc)
        res.error_type = type(exc).__name__
    return res


def _run_langgraph(scenario: Scenario) -> RunResult:
    recorder = _NodeRecorder()
    app = _rebuild(scenario, LANGGRAPH, recorder).compile()
    res = RunResult()
    try:
        # updates 模式是**每个任务完成一次就 yield 一次**，不是每个超步一次。
        # debug 模式的 task 事件带有 langgraph 分配的超步号；按它分组，
        # 才能把同一超步里并行完成的任务还原成一个批次。
        by_step: dict[int, set[str]] = {}
        for event in app.stream(scenario.input, config=scenario.config, stream_mode="debug"):
            if event.get("type") == "task":
                by_step.setdefault(event["step"], set()).add(event["payload"]["name"])
        res.steps = [frozenset(by_step[step]) for step in sorted(by_step)]
        res.task_counts = recorder.snapshot()
        res.final = dict(app.invoke(scenario.input, config=scenario.config))
    except Exception as exc:  # noqa: BLE001
        res.task_counts = recorder.snapshot()
        res.error = str(exc)
        res.error_type = type(exc).__name__
    return res


def _rebuild(scenario: Scenario, backend, recorder: _NodeRecorder | None = None) -> Any:
    """按给定后端搭一张全新的图，可选地给所有节点包上调用计数器。"""
    if recorder is None:
        return scenario.build(backend)
    graph_cls = backend.state_graph_cls

    def recording_graph_cls(schema):
        return _RecordingGraph(graph_cls(schema), recorder)

    return scenario.build(replace(backend, state_graph_cls=recording_graph_cls))


@dataclass
class DiffReport:
    scenario: str
    mini: RunResult
    lg: RunResult
    problems: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.problems

    def format(self) -> str:
        lines = [f"场景 {self.scenario}: {'PASS' if self.passed else 'FAIL'}"]
        if self.problems:
            lines.append(f"  minigraph : steps={self.mini.steps} final={self.mini.final} "
                         f"error={self.mini.error}")
            lines.append(f"  langgraph: steps={self.lg.steps} final={self.lg.final} "
                         f"error={self.lg.error}")
            for p in self.problems:
                lines.append(f"  - {p}")
        return "\n".join(lines)


def compare(scenario: Scenario) -> DiffReport:
    mini = _run_minigraph(scenario)
    lg = _run_langgraph(scenario)
    report = DiffReport(scenario.name, mini, lg)

    if mini.error is not None and lg.error is not None:
        # 两边都失败时也要比较失败前的执行轨迹，不能只凭"都抛异常"判一致。
        # debug 流在失败时可能不完整，所以比较节点执行次数而非超步集合。
        if mini.task_counts != lg.task_counts:
            report.problems.append(
                f"报错前的节点执行次数不同：minigraph {mini.task_counts}，"
                f"langgraph {lg.task_counts}"
            )
        return report

    if mini.error is not None and lg.error is None:
        report.problems.append(f"minigraph 报错了但 langgraph 没有：{mini.error_type}: {mini.error}")
        return report
    if lg.error is not None and mini.error is None:
        report.problems.append(f"langgraph 报错了但 minigraph 没有：{lg.error_type}: {lg.error}")
        return report

    # 超步总数
    if len(mini.steps) != len(lg.steps):
        report.problems.append(
            f"超步数不同：minigraph {len(mini.steps)} 步，langgraph {len(lg.steps)} 步"
        )

    # 逐超步比节点集合
    for i, (m, l) in enumerate(zip(mini.steps, lg.steps)):
        if m != l:
            report.problems.append(
                f"第 {i + 1} 超步节点不同：minigraph {sorted(m)}，langgraph {sorted(l)}"
            )

    # 最终状态只在两边都成功时比较；两边都报错时，上面的超步比对仍然有效，
    # 因而不会把"在不同时间、因不同原因报错"误判为一致。
    if mini.ok and lg.ok and mini.final != lg.final:
        report.problems.append(
            f"最终状态不同：\n    minigraph = {mini.final}\n    langgraph= {lg.final}"
        )

    return report


def run_all(scenarios=None) -> list[DiffReport]:
    if not HAS_LANGGRAPH:
        raise RuntimeError("langgraph 没装，无法做差分测试")
    return [compare(s) for s in (scenarios or SCENARIOS)]
