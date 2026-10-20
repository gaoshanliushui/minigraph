"""
§0 哨兵与类型别名

包内最底层的符号；其他模块都从这里 import。
"""

from __future__ import annotations

from typing import Any, Callable

START = "__start__"
END = "__end__"

Reducer = Callable[[Any, Any], Any]  # (旧值, 新值) -> 合并后的值


class _Missing:
    def __repr__(self) -> str:  # pragma: no cover
        return "<missing>"


MISSING = _Missing()