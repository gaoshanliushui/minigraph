"""
差分测试：minigraph vs 真实 LangGraph。

这个文件**只有在装了 langgraph 的解释器下才有意义**。没装就整体 skip，
不影响普通的单元测试。

跑法（需要同一个解释器里既有 langgraph 又有 minigraph）：

    /tmp/lgvenv/bin/python -m pytest tests/test_diff_langgraph.py -v

minigraph 是纯标准库实现，所以把仓库根目录加进 sys.path 就能被同一个
解释器 import 到。
"""

from __future__ import annotations

import os
import sys

import pytest

# 让这个解释器能 import 到仓库里的 minigraph
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

pytest.importorskip("langgraph", reason="没装 langgraph，跳过差分测试")

from tests.difflang.harness import compare  # noqa: E402
from tests.difflang.scenarios import SCENARIOS  # noqa: E402


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_matches_real_langgraph(scenario):
    """同一张图，minigraph 与真实 LangGraph 的执行过程应当一致。"""
    report = compare(scenario)
    assert report.passed, report.format()
