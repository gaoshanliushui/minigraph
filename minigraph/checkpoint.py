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

    # 同一超步里**所有**挂起点。多个节点并行挂起时会有多个 ——
    # interrupt 字段只是第一个（兼容旧写法），这里才是完整列表。
    # 恢复时按节点名或槽位给值，可以一次把几个都答复掉。
    interrupts: list[dict[str, Any]] = field(default_factory=list)

    # 已收集到的 resume 值：槽位 -> 值队列。
    #   - 多个节点同时挂起：每个槽位各有一个值
    #   - 一个节点里多次 interrupt()：同一个槽位有多个值，按调用顺序出队
    # 整步回滚意味着挂起的节点会重跑，所以这些值必须**跨 resume 累积保存**，
    # 否则重跑时取不到已经答复过的那个。
    resumes: dict[str, list[Any]] = field(default_factory=dict)

    # 挂起那一批任务的槽位分配：{name, idx, push, slot}。
    # 恢复时整批重跑，靠它把每个任务还原成当初的槽位 —— 否则槽位里的
    # step 号会漂移，之前答复过的值就配不上槽位了。
    slots: list[dict[str, Any]] = field(default_factory=list)

    # ---- 通道模型的持久化状态 ----------------------------------------
    #
    # values 是**扁平的状态投影**（只含状态字段，给 get_state 用）；
    # 下面三样才是恢复执行所需的完整内核状态。

    # 每个通道的当前值快照（数据通道 + 触发通道 + 屏障通道都有）。
    channel_values: dict[str, Any] = field(default_factory=dict)

    # 每个通道的版本号：channel_versions[chan]。
    # 通道被写入一次，版本就推进一步；下游节点据此判断"我该醒了"。
    channel_versions: dict[str, int] = field(default_factory=dict)

    # 每个节点的水位线：versions_seen[node][chan] = 该节点上次看到的版本。
    # 触发条件：channel_versions[chan] > versions_seen[node][chan]。
    versions_seen: dict[str, dict[str, int]] = field(default_factory=dict)

    def copy(self) -> "Checkpoint":
        import copy as _c

        return Checkpoint(
            id=self.id, step=self.step, values=_c.deepcopy(self.values),
            next=tuple(self.next), tasks=_c.deepcopy(self.tasks),
            interrupt=_c.deepcopy(self.interrupt),
            interrupts=_c.deepcopy(self.interrupts),
            resumes=_c.deepcopy(self.resumes),
            slots=_c.deepcopy(self.slots),
            channel_values=_c.deepcopy(self.channel_values),
            channel_versions=_c.deepcopy(self.channel_versions),
            versions_seen=_c.deepcopy(self.versions_seen),
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