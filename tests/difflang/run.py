"""
差分测试的独立运行器（不依赖 pytest）。

    /tmp/lgvenv/bin/python tests/difflang/run.py

输出一张逐场景的对照表：minigraph 与真实 LangGraph 的超步序列、
最终状态、以及差异说明。
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests.difflang.harness import compare  # noqa: E402
from tests.difflang.scenarios import SCENARIOS  # noqa: E402


def main() -> int:
    reports = [compare(s) for s in SCENARIOS]

    width = max(len(r.scenario) for r in reports)
    print(f"{'场景':<{width}}  {'结果':<6} 说明")
    print("-" * (width + 30))
    for r in reports:
        status = "PASS" if r.passed else "FAIL"
        note = "" if r.passed else f"{len(r.problems)} 处差异"
        print(f"{r.scenario:<{width}}  {status:<6} {note}")

    failed = [r for r in reports if not r.passed]
    if failed:
        print("\n" + "=" * 70)
        for r in failed:
            print(r.format())
            print()

    print(f"\n{len(reports) - len(failed)}/{len(reports)} 个场景与真实 LangGraph 一致")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
