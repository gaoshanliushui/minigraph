"""
§6 通道（channel）：LangGraph 真正的"边"

这一节回答一个反直觉的问题：**LangGraph 里的"边"到底是什么？**

答案不是图结构，而是一次 **channel 写入**：

    add_edge(A, B)   编译成   A 写通道 "branch:to:B"，B 订阅这个通道

于是"什么时候该跑 B"这个问题，和"图长什么样"**完全无关**，
只取决于两件事：

  1. 通道 **branch:to:B** 的版本号有没有比 B 上次看到的版本新；
  2. 通道当前是不是**可用**（有值）。

    channel_versions["branch:to:B"] > versions_seen["B"]["branch:to:B"]

这就是本节要实现的东西。`runtime.py` 里的超步循环因此变成：

    算出「哪些节点的输入 channel 变了」 -> 并发跑它们 -> 收写入、推版本 -> 重复

对照真实 LangGraph 源码：
  - langgraph/pregel/_algo.py  → apply_writes / prepare_next_tasks / _triggers
  - langgraph/channels/        → 各通道类型
  - langgraph/graph/state.py:1444-1625 → attach_node/attach_edge/attach_branch
    （边在这里被编译成通道写入与触发订阅）

通道分两类，务必分清：

  **数据通道**  状态字段本身。存值、被节点读取，但不参与触发。
                （last_value / binary_op）

  **触发通道**  纯粹的"信号灯"。只关心有没有被写过（版本推进），
                存的值本身通常没意义。
                （ephemeral 分支通道 / named_barrier 汇聚屏障）
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

from .constants import MISSING, Reducer
from .state import LAST_VALUE


class EmptyChannelError(Exception):
    """通道为空（从未被写入，或已被消费清空）。"""


class ConcurrentWriteError(RuntimeError):
    """
    同一超步里多个节点写了同一个**没有 reducer** 的字段。

    这是 LangGraph 的 `InvalidUpdateError` 对应物。之所以要报错而不是
    静默取"最后一个"：并行下"最后一个"是不确定的，last-write-wins
    在并发场景里等于随机覆盖。

    继承 RuntimeError 是为了兼容既有调用方
    （pytest.raises(RuntimeError, match="同一超步")）。
    """


class BaseChannel:
    """
    通道基类。语义与 LangGraph 的 `langgraph.channels.base.BaseChannel` 对齐。

    子类必须实现 update() / get()；其余方法有合理默认。

    生命周期（三段式，对应 Pregel 的三段）：

        update(values)   超步末尾被调用，把这一批写入合并进通道
        consume()        订阅者跑完后被调用（少数通道借此自清）
        finish()         整张图收尾时被调用（少数通道借此"封口"）
    """

    __slots__ = ("key",)

    def __init__(self, key: str = ""):
        self.key = key

    # -- 写 ---------------------------------------------------------------
    def update(self, values: Sequence[Any]) -> bool:
        """
        合并一批写入。返回 True 表示**通道的可用性发生了变化**
        （也就是"这个通道被更新了，值得推版本号"）。
        """
        raise NotImplementedError

    # -- 读 ---------------------------------------------------------------
    def get(self) -> Any:
        """取当前值；空则抛 EmptyChannelError。"""
        raise NotImplementedError

    def is_available(self) -> bool:
        """通道是否有可读的值。"""
        try:
            self.get()
        except EmptyChannelError:
            return False
        return True

    # -- 生命周期 ---------------------------------------------------------
    def consume(self) -> bool:
        """订阅者已消费；返回 True 表示通道状态变了。"""
        return False

    def finish(self) -> bool:
        """整图收尾；返回 True 表示通道状态变了。"""
        return False

    # -- 持久化 -----------------------------------------------------------
    def checkpoint(self) -> Any:
        """可序列化快照；空通道返回 MISSING。"""
        try:
            return self.get()
        except EmptyChannelError:
            return MISSING

    def from_checkpoint(self, checkpoint: Any) -> "BaseChannel":
        """从快照恢复出一个等价通道。"""
        raise NotImplementedError

    def copy(self) -> "BaseChannel":
        """浅拷贝一个独立实例（改副本不影响原通道）。"""
        return self.from_checkpoint(self.checkpoint())

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} {self.key!r}>"


class LastValue(BaseChannel):
    """
    存最后一次收到的值；**每步最多只能收一个值**。

    对应没有 reducer 的状态字段（缺省 last-write-wins）。

    注意：这里"每步最多一个"不是性能优化，而是**正确性约束**。
    同一超步两个并行节点都写这个字段时，"最后一个"是不确定的，
    所以直接报错，让作者表态（对齐 LangGraph 的 LastValue）。
    """

    __slots__ = ("value",)

    def __init__(self, key: str = ""):
        super().__init__(key)
        self.value: Any = MISSING

    def update(self, values: Sequence[Any]) -> bool:
        if len(values) == 0:
            return False
        if len(values) != 1:
            raise ConcurrentWriteError(
                f"同一超步有多个写入争抢字段 {self.key!r}；"
                f"并发写必须给它配 reducer（如 operator.add / merge_dict）"
                f"（LangGraph 对应错误：At key {self.key!r}: "
                f"Can receive only one value per step.）"
            )
        self.value = values[-1]
        return True

    def get(self) -> Any:
        if self.value is MISSING:
            raise EmptyChannelError()
        return self.value

    def is_available(self) -> bool:
        return self.value is not MISSING

    def checkpoint(self) -> Any:
        return self.value

    def from_checkpoint(self, checkpoint: Any) -> "LastValue":
        empty = LastValue(self.key)
        if checkpoint is not MISSING:
            empty.value = checkpoint
        return empty


class BinaryOperatorAggregate(BaseChannel):
    """
    用二元算子把新值合并进旧值：value = reducer(value, new)。

    对应带 reducer 的状态字段。LangGraph 里叫 BinaryOperatorAggregate，
    判定规则是「reducer 是恰好两个位置参数的 callable」——所以
    `operator.add` / `add_messages` / `merge_dict` 全都落进这一类。

    关于初值：LangGraph 用 `typ()`（list->[]、int->0），这里用 MISSING。
    差别在于 minigraph 的 reducer 约定 old 可能是 MISSING
    （见 state.py 里 add_messages / merge_dict 对 MISSING 的处理）。
    为了两种约定都成立，首次写入走「直接落盘」的捷径：

        value = values[0]        # 等价于 reducer(MISSING, values[0])

    对 add_messages / merge_dict / operator.add 三者，这个捷径与
    reducer(MISSING, v) 的结果一致。
    """

    __slots__ = ("value", "operator")

    def __init__(self, key: str = "", operator: Reducer | None = None):
        super().__init__(key)
        self.value: Any = MISSING
        self.operator = operator

    def update(self, values: Sequence[Any]) -> bool:
        if len(values) == 0:
            return False
        if self.value is MISSING:
            # 首次写入：直接落盘（等价 reducer(MISSING, v)）
            self.value = values[0]
            values = values[1:]
        for v in values:
            self.value = self.operator(self.value, v)
        return True

    def get(self) -> Any:
        if self.value is MISSING:
            raise EmptyChannelError()
        return self.value

    def is_available(self) -> bool:
        return self.value is not MISSING

    def checkpoint(self) -> Any:
        return self.value

    def from_checkpoint(self, checkpoint: Any) -> "BinaryOperatorAggregate":
        empty = BinaryOperatorAggregate(self.key, self.operator)
        if checkpoint is not MISSING:
            empty.value = checkpoint
        return empty


class EphemeralValue(BaseChannel):
    """
    只保留"上一超步收到的值"，读一次就清空。

    这是**分支通道**（branch:to:X）的类型——它承载的是"信号"不是"数据"：
    值本身通常是 None，有意义的只是"被写过"这件事（版本号推进）。

    清空时机很关键：超步末尾，**没被本步写入**但仍然可用的通道会被
    调用 update([])，从而把自己抹掉（对齐 LangGraph apply_writes 里的
    "Channels that weren't updated in this step are notified of a new step"）。
    这正是"同一个超步不会重复触发同一节点"的由来。

    guard=False 时允许一步收到多个值（取最后一个）——两个并行节点同时
    Command(goto="x") 是合法的，所以分支通道必须 guard=False。
    """

    __slots__ = ("value", "guard")

    def __init__(self, key: str = "", guard: bool = True):
        super().__init__(key)
        self.guard = guard
        self.value: Any = MISSING

    def update(self, values: Sequence[Any]) -> bool:
        if len(values) == 0:
            # 空写入 = "本步没人写我" -> 自清
            if self.value is not MISSING:
                self.value = MISSING
                return True
            return False
        if len(values) != 1 and self.guard:
            raise ConcurrentWriteError(
                f"通道 {self.key!r} 一步收到了 {len(values)} 个值；"
                f"分支通道应当用 guard=False"
            )
        self.value = values[-1]
        return True

    def get(self) -> Any:
        if self.value is MISSING:
            raise EmptyChannelError()
        return self.value

    def is_available(self) -> bool:
        return self.value is not MISSING

    def checkpoint(self) -> Any:
        return self.value

    def from_checkpoint(self, checkpoint: Any) -> "EphemeralValue":
        empty = EphemeralValue(self.key, self.guard)
        if checkpoint is not MISSING:
            empty.value = checkpoint
        return empty


class Topic(BaseChannel):
    """
    发布订阅：一步内累积收到的所有值。

    真实 LangGraph 里只有 `__pregel_tasks` 用它——Send 队列。
    节点把 Send 写进来，下一超步每个 Send 变成一个 PUSH 任务。

    accumulate=False：每次 update 先清空再收，所以只保留本步的值。
    """

    __slots__ = ("values", "accumulate")

    def __init__(self, key: str = "", accumulate: bool = False):
        super().__init__(key)
        self.accumulate = accumulate
        self.values: list[Any] = []

    def update(self, values: Sequence[Any]) -> bool:
        updated = False
        if not self.accumulate:
            if self.values:
                updated = True
            self.values = []
        # 展平一层：值本身是 list 时拆开（LangGraph 的 _flatten）
        flat: list[Any] = []
        for v in values:
            if isinstance(v, list):
                flat.extend(v)
            else:
                flat.append(v)
        if flat:
            updated = True
            self.values.extend(flat)
        return updated

    def get(self) -> list[Any]:
        if not self.values:
            raise EmptyChannelError()
        return list(self.values)

    def is_available(self) -> bool:
        return bool(self.values)

    def checkpoint(self) -> Any:
        return list(self.values)

    def from_checkpoint(self, checkpoint: Any) -> "Topic":
        empty = Topic(self.key, self.accumulate)
        if checkpoint is not MISSING:
            empty.values = list(checkpoint)
        return empty


class NamedBarrierValue(BaseChannel):
    """
    屏障：等所有指定的名字都到齐了，通道才变成"可用"。

    对应 LangGraph 的 `add_edge(["a","b"], "c")`——多源汇聚。
    编译出的通道名形如 `join:a+b:c`，每个源写入**自己的名字**作为"我到了"
    的信号；集齐后 c 被触发一次。

    这是"跑一次"和"跑两次"的分水岭：

        add_edge("a", "c"); add_edge("b", "c")   -> 两条独立边 -> c 触发两次
        add_edge(["a", "b"], "c")                -> 一条屏障边 -> c 触发一次
    """

    __slots__ = ("names", "seen")

    def __init__(self, key: str = "", names: Iterable[str] = ()):
        super().__init__(key)
        self.names: set[str] = set(names)
        self.seen: set[str] = set()

    def update(self, values: Sequence[Any]) -> bool:
        updated = False
        for v in values:
            if v not in self.names:
                raise ValueError(
                    f"通道 {self.key!r} 收到意外的值 {v!r}；期望是 {sorted(self.names)} 之一"
                )
            if v not in self.seen:
                self.seen.add(v)
                updated = True
        return updated

    def get(self) -> None:
        if self.seen != self.names:
            raise EmptyChannelError()
        return None

    def is_available(self) -> bool:
        return self.seen == self.names

    def consume(self) -> bool:
        # 触发之后重新武装屏障，这样下一轮汇聚还能再等一次
        if self.seen == self.names:
            self.seen = set()
            return True
        return False

    def checkpoint(self) -> Any:
        return set(self.seen)

    def from_checkpoint(self, checkpoint: Any) -> "NamedBarrierValue":
        empty = NamedBarrierValue(self.key, self.names)
        if checkpoint is not MISSING:
            empty.seen = set(checkpoint)
        return empty


# ---------------------------------------------------------------------------
# 状态字段 -> 数据通道
# ---------------------------------------------------------------------------

def channel_for_field(field: str, reducer: Reducer | None) -> BaseChannel:
    """
    把一个状态字段映射成对应的数据通道。

    判定规则与 LangGraph 一致（`graph/state.py` 的 _is_field_binop）：
    带 reducer -> BinaryOperatorAggregate；不带 -> LastValue。

    add_messages / merge_dict 本来就是 (old, new) -> merged 的二元 reducer，
    所以它们自动落进 BinaryOperatorAggregate，**不需要特殊处理**。

    注意 LAST_VALUE：State.__init_subclass__ 会把「没写 Annotated」的字段
    也登记成 reducers[field] = LAST_VALUE（而不是 None）。LAST_VALUE 本身
    就是"后写覆盖"，语义上等同于"没有 reducer"，所以要归到 LastValue 通道 ——
    它能在一步收到多个值时报错，而 BinaryOperatorAggregate 会静默取最后一个。
    """
    if reducer is None or reducer is LAST_VALUE:
        return LastValue(field)
    return BinaryOperatorAggregate(field, reducer)
