"""
检查点层面的差分测试：minigraph vs 真实 LangGraph。

超步循环的差分见 tests/test_diff_langgraph.py；这里比的是**落盘之后**的
行为 —— update_state 的 as_node 语义、interrupt 的挂起与恢复、时间旅行。

跑法（需要同一个解释器里既有 langgraph 又有 minigraph）：

    /tmp/lgvenv/bin/python -m pytest tests/test_diff_checkpoint.py -v
"""

from __future__ import annotations

import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

pytest.importorskip("langgraph", reason="没装 langgraph，跳过差分测试")

from tests.difflang.backends import LANGGRAPH, MINIGRAPH  # noqa: E402
from tests.difflang.checkpoint_scenarios import SCENARIOS  # noqa: E402


def _run(backend, scenario):
    app = scenario.build(backend)
    try:
        return scenario.run(backend, app, f"thread-{backend.name}")
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_checkpoint_semantics_match_real_langgraph(scenario):
    """检查点语义：minigraph 与真实 LangGraph 应当一致。"""
    mini = _run(MINIGRAPH, scenario)
    lg = _run(LANGGRAPH, scenario)
    assert mini == lg, (
        f"场景 {scenario.name} 不一致：\n"
        f"    minigraph = {mini}\n"
        f"    langgraph = {lg}"
    )
