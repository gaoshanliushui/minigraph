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

    # ---- DeltaChannel 的写入日志 -------------------------------------
    #
    # 这一步里各个通道收到的**增量写入**，按 (通道, 值) 记。
    # 只给 DeltaChannel 用：它的值不进 checkpoint，靠重放这些增量重建。
    #
    # 形状是 list[(channel, value)]，同一超步内多个节点写同一通道就
    # 有多条（顺序 = 节点执行顺序）。
    #
    # 存储上它是**独立的写入日志**，不是 checkpoint blob 的一部分
    # （对齐 LangGraph 的 writes 表）：SqliteSaver 把它存进 writes 表，
    # 读的时候按 idx 保序拼回这里。所以对使用者来说 Checkpoint 仍是
    # 自包含的 —— pending_writes 是个查询视图。
    pending_writes: list[tuple[str, Any]] = field(default_factory=list)

    # DeltaChannel 的快照节奏计数器：通道名 -> (写入次数, 距上次快照的超步数)。
    # 满足任一阈值就落一份完整快照（见 channels.DeltaChannel 的说明）。
    delta_counters: dict[str, tuple[int, int]] = field(default_factory=dict)

    # 父 checkpoint 的 id（None = 这条线的起点）。
    #
    # 为什么需要显式父链（对齐 LangGraph 的 parent_checkpoint_id 列）：
    # 历史**不是一条线**。update_state / invoke 带 checkpoint_id 从历史点
    # 分叉后，存储里的写入顺序（seq）不再等于祖先后代关系 —— 按 seq 往回
    # 扫会把**另一条分叉线**的写入也算进 delta 重放，状态直接算错。
    # 有了父链，get_delta_channel_history 只走 on-path 祖先。
    parent_id: str | None = None

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
            pending_writes=_c.deepcopy(self.pending_writes),
            delta_counters=_c.deepcopy(self.delta_counters),
            parent_id=self.parent_id,
        )


class InMemorySaver:
    """按 thread_id 追加保存 checkpoint；就是 LangGraph 的 InMemorySaver。"""

    def __init__(self) -> None:
        self.storage: dict[str, list[Checkpoint]] = {}
        # id -> checkpoint 索引：沿 parent_id 走祖先链时按 id 直接定位，
        # 不用每次扫全表（分叉后"前一个"根本不是存储里的前一条）。
        self._by_id: dict[str, Checkpoint] = {}
        self._n = 0

    def new_id(self) -> str:
        self._n += 1
        return f"ckpt-{self._n:04d}-{random.randint(0, 0xFFFF):04x}"

    def put(self, thread_id: str, ckpt: Checkpoint) -> str:
        self.storage.setdefault(thread_id, []).append(ckpt)
        self._by_id[ckpt.id] = ckpt
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

    def get_delta_channel_history(self, thread_id: str, channels: list[str],
                                  checkpoint_id: str | None = None
                                  ) -> dict[str, dict[str, Any]]:
        """
        沿 thread 的 **parent 链**往回走，取出每个 delta 通道的
        「种子 + 增量写入」。

        返回：{通道名: {"seed": 种子值(可缺省), "writes": [增量, ...]}}

        走法（对齐 LangGraph 的 get_delta_channel_history）：
          从目标 checkpoint 沿 parent_id 往前找，每个通道在
          **最近一个存过值的祖先**处停下 —— 那份值就是 seed；
          seed 之后积累的写入就是 writes。走到底还没找到 seed ->
          这个通道从空开始（seed 缺省）。

        为什么走父链而不是存储顺序（LangGraph: "walks the parent chain,
        not list(before=...): for forked threads, only on-path ancestors
        contribute"）：
          从历史点分叉后，存储里两条线的 checkpoint 是交错追加的，
          按写入顺序往回扫会把**另一条线**的写入也重放进来。

        为什么这样能省空间：
          有了 seed，之前的祖先就不用再看了；而 seed 每隔一段时间
          就重新落一份，重放深度因此有上限。
        """
        if not channels:
            return {}
        target = self.get(thread_id, checkpoint_id)

        # 从新到旧扫。seed_by_ch 存找到的种子，writes_by_ch 累积增量。
        # 找到 seed 之后就不再往更早的祖先走（但要继续为**别的**通道走）。
        seed_by_ch: dict[str, Any] = {}
        writes_by_ch: dict[str, list[Any]] = {c: [] for c in channels}
        remaining = set(channels)

        # 先把"要重放的 checkpoint"从新到旧收集起来。
        #
        # 为什么先收集再展开：一个 checkpoint 里同一个通道可能有**多条**
        # 写入（并行分支各写一条），它们之间的先后顺序有意义（reducer
        # 可能对顺序敏感）。边走边 insert(0) 会把这一步内部的顺序打乱，
        # 所以这里按 checkpoint 收集，最后再统一按旧->新展开。
        #
        # 关键：**找到种子的那份 checkpoint 不参与重放**。
        # 种子值 = 那个时刻的完整状态，已经包含了它自己的写入；
        # 再重放一遍就重复了（计数器这种不幂等的 reducer 会直接算错）。
        collected: list[Checkpoint] = []
        cursor = target
        walked: set[str] = set()          # 防御：parent 链成环时止步
        while cursor is not None and remaining and cursor.id not in walked:
            walked.add(cursor.id)
            found_seed_here = False
            for ch in list(remaining):
                if ch in cursor.channel_values:
                    seed_by_ch[ch] = cursor.channel_values[ch]
                    remaining.discard(ch)
                    found_seed_here = True
            if found_seed_here:
                # 这一份是种子（或含某个通道的种子）——
                # 它自己的写入已经在种子值里了，不参与重放。
                # 更早的祖先同理：种子值已经包含了它们。
                cursor = self._by_id.get(cursor.parent_id)
                continue
            collected.append(cursor)
            cursor = self._by_id.get(cursor.parent_id)

        # 从旧到新展开写入（collected 是新->旧，所以倒着遍历）
        for ckpt in reversed(collected):
            for chan, val in ckpt.pending_writes:
                if chan in writes_by_ch:
                    writes_by_ch[chan].append(val)

        result: dict[str, dict[str, Any]] = {}
        for ch in channels:
            entry: dict[str, Any] = {"writes": writes_by_ch[ch]}
            if ch in seed_by_ch:
                entry["seed"] = seed_by_ch[ch]
            result[ch] = entry
        return result