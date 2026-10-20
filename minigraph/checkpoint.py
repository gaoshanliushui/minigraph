"""
§3 检查点
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Checkpoint:
    id: str
    step: int
    values: dict[str, Any]
    next: tuple[str, ...]
    # 待执行任务，保留 Send 以便恢复后能重建并行分支
    tasks: list[Any] = field(default_factory=list)
    interrupt: dict[str, Any] | None = None

    def copy(self) -> "Checkpoint":
        import copy as _c

        return Checkpoint(
            id=self.id, step=self.step, values=_c.deepcopy(self.values),
            next=tuple(self.next), tasks=_c.deepcopy(self.tasks),
            interrupt=_c.deepcopy(self.interrupt),
        )


class InMemorySaver:
    """按 thread_id 追加保存 checkpoint；就是 LangGraph 的 InMemorySaver。"""

    def __init__(self) -> None:
        self.storage: dict[str, list[Checkpoint]] = {}
        self._n = 0

    def new_id(self) -> str:
        self._n += 1
        return f"ckpt-{self._n:04d}-{random.randint(0, 0xFFFF):04x}"

    def put(self, thread_id: str, ckpt: Checkpoint) -> str:
        self.storage.setdefault(thread_id, []).append(ckpt)
        return ckpt.id

    def list(self, thread_id: str) -> list[Checkpoint]:
        return self.storage.get(thread_id, [])

    def get(self, thread_id: str, checkpoint_id: str | None = None) -> Checkpoint | None:
        history = self.list(thread_id)
        if not history:
            return None
        if checkpoint_id is None:
            return history[-1]
        for ckpt in history:  # 支持回到任意历史点重新分支
            if ckpt.id == checkpoint_id:
                return ckpt
        raise KeyError(f"thread {thread_id} 下找不到 checkpoint {checkpoint_id}")