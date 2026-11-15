"""
pytest 插件：把触发判据换成「只看通道可用性」，然后跑测试。

用途：为 docs/versions与seen水位线详解.md 的「为什么不能只看可用性」一节
提供**实测证据** —— 哪些测试会因为去掉版本判定而失败。

不改项目源码：这个文件只 monkeypatch `runtime._triggers`，
项目里的 runtime.py 一行都不动。

跑法：
    .venv/bin/python -m pytest tests/ -q -p scripts.exp_plugin_avail_only
"""
from __future__ import annotations

from minigraph import runtime as rt


def _triggers_avail_only(channels, versions, seen, trig) -> bool:
    """只判可用性 —— 不看版本，不管水位线。"""
    for chan in trig:
        ch = channels.get(chan)
        if ch is not None and ch.is_available():
            return True
    return False


rt._triggers = _triggers_avail_only
