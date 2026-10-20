"""
§3 检查点
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Checkpoint:
    # checkpoint 的唯一标识，由 InMemorySaver.new_id() 生成
    # （形如 "ckpt-0001-abcd"），主要给 get_state(checkpoint_id=...) / 时间
    # 旅行 / update_state 用来精确定位某一份历史快照。
    id: str

    # 这一份快照对应的超步编号。第 1 步 = 入口节点刚跑完，
    # 第 2 步 = 第二批节点刚跑完…… 与 runtime._run 里 step 自增严格对齐。
    # 拿到 checkpoint 一眼就能看出"跑到第几步了"。
    step: int

    # 这一步结束时的**完整状态**——不是增量，而是把截至目前所有 reducer
    # 合并后的整份 values。从 checkpoint 恢复时直接把 values 拷进 values
    # 变量即可，不必再 replay 之前的写入历史（state 是 BSP 模型，幂等合并
    # 后等价于"重放整段历史"）。
    values: dict[str, Any]

    # 下一步准备执行的节点名列表（**只含名字，不含 Send**）。
    # 主要给 get_state_history 等"看一眼接下来会发生什么"的场景用，
    # 真正执行靠下面那个 tasks。
    next: tuple[str, ...]

    # 待执行任务，**保留 Send** 以便恢复后能重建并行分支。
    # next 只是这一份 tasks 的"投影"，丢失了 Send.arg 等动态信息；
    # 恢复时必须从 tasks 还原——尤其 Send 里的 arg 是上游节点在运行期
    # 才决定的，不存进 checkpoint 就丢了。约定 list 里元素要么是 str
    # （节点名）要么是 Send 实例。
    tasks: list[Any] = field(default_factory=list)

    # 如果这一步是因 interrupt() 挂起的，记录挂起点信息：
    #   {"slot": 槽位, "node": 节点名, "idx": 批内序号,
    #    "step": 挂起时所在步号, "value": 传给 interrupt() 的值}
    # 恢复时 _run 用它定位挂起的任务、注入 resume 值；非 None 即
    # "图被冻住了，必须传 config={'resume': ...} 才能继续"。
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