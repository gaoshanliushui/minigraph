"""
校验文档里 `runtime.py:NNN` 这类行号引用是否还对得上。

代码一改，行号就漂。这个脚本把每处引用拿去源码里查一下，
看它指向的那一行是不是"预期的那个" —— 预期用关键词描述。

用法：
    .venv/bin/python scripts/check_doc_refs.py          # 报告对不上的引用
    .venv/bin/python scripts/check_doc_refs.py --fix    # 自动把行号修对
"""
from __future__ import annotations

import os
import re
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
RUNTIME = os.path.join(ROOT, "minigraph", "runtime.py")
DOC = os.path.join(ROOT, "docs", "versions与seen水位线详解.md")


def load(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        return fh.read().splitlines()


# 文档里每个行号引用应该指向的代码。
# 格式：(行号, 该行必须包含的关键词)
# 关键词写具体一点，避免"同一个词在别处也出现"导致误判。
EXPECT: dict[int, str] = {
    135: "def advance_watermarks",
    160: "def consume_triggers",
    176: "self.versions[chan] = next_version",
    181: "def expire_unwritten",
    191: "self.versions[chan] = next_version",
    196: "def dump",
    309: "self.trigger_to_nodes",
    332: "def invoke",
    424: "rt = _RuntimeState(",
    429: "rt.expire_unwritten",
    463: "for name, node_seen in seen.items()",
    474: "def _restore_channels",
    490: "versions = {c: 0 for c in channels}",
    549: "versions = {c: 0 for c in channels}",
    562: "seen={}",
    570: "start_spec = self.nodes.get(START)",
    577: "rt.seen[START]",
    650: '_ctx()["slot"] = None',
    671: "会漂移",
    692: "多个节点并行挂起时",
    727: "rt.advance_watermarks(tasks)",
    852: "def _triggers",
    873: "if versions.get(chan, 0) > seen.get(chan, 0):",
    878: "def _prepare_next_tasks",
    928: "def _next_version",
    933: "def _apply_writes",
    966: "versions[chan] = version",
    1059: "ThreadPoolExecutor",
}


def check(runtime: list[str], doc: str) -> list[tuple[int, int]]:
    """返回 [(旧行号, 新行号)]，表示需要修的引用。"""
    # 文档里实际用到的行号
    used = sorted({int(m) for m in re.findall(r"runtime\.py:(\d+)", doc)})

    bad = []
    for old in used:
        want = EXPECT.get(old)
        if want is None:
            print(f"  ? runtime.py:{old} 不在预期表里（新增的引用？加进 EXPECT）")
            continue
        # 在源码里找"包含关键词、且离 old 最近"的行
        candidates = [i for i, line in enumerate(runtime, start=1) if want in line]
        if not candidates:
            print(f"  ? 源码里找不到 {want!r}")
            continue
        new = min(candidates, key=lambda i: abs(i - old))
        if new != old:
            bad.append((old, new))
    return bad


def main() -> int:
    runtime = load(RUNTIME)
    with open(DOC, encoding="utf-8") as fh:
        doc = fh.read()

    bad = check(runtime, doc)
    for old, new in bad:
        print(f"  runtime.py:{old}  →  runtime.py:{new}")

    if not bad:
        print(f"✓ 全部 {len(EXPECT)} 处引用都对得上")
        return 0

    if "--fix" not in sys.argv:
        print(f"\n共 {len(bad)} 处需要改。跑 --fix 自动改。")
        return 0

    for old, new in sorted(bad, key=lambda x: -x[0]):
        doc = re.sub(rf"runtime\.py:{old}(?!\d)", f"runtime.py:{new}", doc)
    with open(DOC, "w", encoding="utf-8") as fh:
        fh.write(doc)
    print(f"\n已改 {len(bad)} 处")
    return 0


if __name__ == "__main__":
    sys.exit(main())
