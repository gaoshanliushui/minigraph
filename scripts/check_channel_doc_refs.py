"""
校验 docs/channel通道模型详解.md 里的行号引用是否还对得上。

代码一改，行号就漂。这个脚本把每处引用拿去源码里查一下，
看它指向的那一行是不是"预期的那个" —— 预期用关键词描述。

用法：
    .venv/bin/python scripts/check_channel_doc_refs.py          # 报告对不上的引用
    .venv/bin/python scripts/check_channel_doc_refs.py --fix    # 自动把行号修对
"""
from __future__ import annotations

import os
import re
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
CHANNELS = os.path.join(ROOT, "minigraph", "channels.py")
RUNTIME = os.path.join(ROOT, "minigraph", "runtime.py")
DOC = os.path.join(ROOT, "docs", "channel通道模型详解.md")


def load(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        return fh.read().splitlines()


# 文件 -> {行号: 该行必须包含的关键词}
# 关键词写具体一点，避免"同一个词在别处也出现"导致误判。
EXPECT: dict[str, dict[int, str]] = {
    "channels.py": {
        82: "def update(self, values: Sequence[Any]) -> bool:",
        94: "def is_available(self) -> bool:",
        103: "def consume(self) -> bool:",
        107: "def finish(self) -> bool:",
        146: "self.value: Any = MISSING",
        149: "if len(values) == 0:",
        151: "if len(values) != 1:",
        158: "self.value = values[-1]",
        166: "def is_available(self) -> bool:",
        209: "# 初值 = typ()",
        212: "try:",
        214: "except Exception:",
        217: "def update(self, values: Sequence[Any]) -> bool:",
        218: "if len(values) == 0:",
        219: "return False",
        221: "# 首次写入：直接落盘",
        226: "return True",
        264: "def __init__(self, key: str = \"\", guard: bool = True):",
        267: "self.value: Any = MISSING",
        269: "def update(self, values: Sequence[Any]) -> bool:",
        270: "if len(values) == 0:",
        274: "return True",
        276: "if len(values) != 1 and self.guard:",
        277: "raise ConcurrentWriteError(",
        279: "分支通道应当用 guard=False",
        281: "self.value = values[-1]",
        307: "节点把 Send 写进来",
        317: "self.values: list[Any] = []",
        319: "def update(self, values: Sequence[Any]) -> bool:",
        320: "updated = False",
        323: "updated = True",
        324: "self.values = []",
        331: "flat.append(v)",
        375: "self.seen: set[str] = set()",
        377: "def update(self, values: Sequence[Any]) -> bool:",
        378: "updated = False",
        380: "if v not in self.names:",
        382: "f\"通道 {self.key!r} 收到意外的值",
        385: "self.seen.add(v)",
        387: "return updated",
        394: "def is_available(self) -> bool:",
        397: "def consume(self) -> bool:",
        402: "return False",
        406: "def channel_for_field",
        418: "def channel_for_field",
    },
    "runtime.py": {
        135: "def advance_watermarks",
        160: "def consume_triggers",
        173: "for chan in spec.triggers:",
        175: "ch.consume()",
        190: "ch.is_available() and ch.update([])",
        196: "def dump",
        224: "self.trigger_to_nodes",
        309: "self.trigger_to_nodes: dict[str, list[str]] = {}",
        574: "rt.channels[chan].update([None])",
        715: "def _advance",
        734: "rt.expire_unwritten",
        852: "def _triggers",
        933: "def _apply_writes",
        956: "changed = ch.update(vals)",
        976: "def _read_state",
        987: "def _state_projection",
        1021: "ch.update([val])",
    },
}


def main() -> int:
    sources = {"channels.py": load(CHANNELS), "runtime.py": load(RUNTIME)}
    with open(DOC, encoding="utf-8") as fh:
        doc = fh.read()

    fix = "--fix" in sys.argv
    bad = []

    for fname, lines in sources.items():
        expect = EXPECT[fname]
        used = sorted({int(m) for m in re.findall(rf"{re.escape(fname)}:(\d+)", doc)})
        for old in used:
            want = expect.get(old)
            if want is None:
                print(f"  ? {fname}:{old} 不在预期表里（新增的引用？加进 EXPECT）")
                continue
            cands = [i for i, line in enumerate(lines, start=1) if want in line]
            if not cands:
                print(f"  ? {fname} 里找不到 {want!r}")
                continue
            new = min(cands, key=lambda i: abs(i - old))
            if new != old:
                bad.append((fname, old, new))

    for fname, old, new in bad:
        print(f"  {fname}:{old}  →  {fname}:{new}")

    if not bad:
        total = sum(len(v) for v in EXPECT.values())
        print(f"✓ 全部引用都对得上（预期表 {total} 条）")
        return 0

    if not fix:
        print(f"\n共 {len(bad)} 处需要改。跑 --fix 自动改。")
        return 0

    for fname, old, new in sorted(bad, key=lambda x: -x[1]):
        doc = re.sub(rf"{re.escape(fname)}:{old}(?!\d)", f"{fname}:{new}", doc)
    with open(DOC, "w", encoding="utf-8") as fh:
        fh.write(doc)
    print(f"\n已改 {len(bad)} 处")
    return 0


if __name__ == "__main__":
    sys.exit(main())
