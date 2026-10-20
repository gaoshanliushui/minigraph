"""
§2 控制流原语
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class Command:
    """节点直接决定下一跳，可顺带提交 update（LangGraph 的 Command）。"""

    goto: str | Sequence[str] | None = None
    update: dict[str, Any] | None = None
    resume: Any = None


@dataclass
class Send:
    """动态扇出：把一份独立输入派给某节点，构造并行任务。"""

    node: str
    arg: Any