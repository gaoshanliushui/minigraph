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

from .constants import LAST_VALUE, MISSING, Reducer


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

    关于初值：用 `typ()`（list->[]、int->0），和 LangGraph 一致。
    于是字段**一开始就有值**，节点读一个还没被写过的字段拿到的是零值，
    而不是 KeyError；`get_state()` 的状态投影也因此和 LangGraph 一样
    带着这些零值字段。

    拿不到类型时（裸 dict 形态声明的 schema、类型不可实例化）退化成
    MISSING —— 此时首次写入走「直接落盘」的捷径：

        value = values[0]        # 等价于 reducer(MISSING, values[0])

    对 add_messages / merge_dict / operator.add 三者，这个捷径与
    reducer(MISSING, v) 的结果一致。
    """

    __slots__ = ("value", "operator", "typ")

    def __init__(self, key: str = "", operator: Reducer | None = None,
                 typ: Any = None):
        super().__init__(key)
        self.value: Any = MISSING
        self.operator = operator
        self.typ = typ
        # 初值 = typ()：list->[]、int->0、dict->{}
        # 类型造不出零值（泛型别名、抽象类型）就退回 MISSING。
        if typ is not None:
            try:
                self.value = typ()
            except Exception:  # noqa: BLE001 - 造不出零值就当没类型
                self.value = MISSING

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
        empty = BinaryOperatorAggregate(self.key, self.operator, self.typ)
        if checkpoint is not MISSING:
            empty.value = checkpoint
        return empty


class DeltaChannel(BaseChannel):
    """
    只在 checkpoint 里存一个哨兵，状态靠**重放祖先写入**重建。

    这是 LangGraph 里的 `DeltaChannel`（对齐 langgraph/channels/delta.py）。
    它解决的是"messages 越长，checkpoint 越大"的问题：

      普通通道（BinaryOperatorAggregate）
        每份 checkpoint 都**完整存一遍**累积后的值。
        100 轮对话 -> 100 份 checkpoint，每份都存全部消息 -> O(n²)。

      DeltaChannel
        每份 checkpoint 只存"这一段增量写了什么"，状态是
        「最近的快照 + 之后的增量重放」。

    ## 代价：reducer 必须满足"分结合并不变性"

        reducer(reducer(state, xs), ys) == reducer(state, xs + ys)

    因为重放时一批增量可能被合并成更大的批次（LangGraph 的
    `replay_writes` 就是一次调用传一批），如果 reducer 不满足这个性质，
    重放出来的值和原始运行不一致。

    `add_messages` 满足吗？满足——它按 id 追加/更新，与批次划分无关。
    `operator.add` 满足。**追加 list 的自定义 reducer 也满足**。

    ## 两个计数器决定何时落快照

      updates     这个通道被写了几次（写才 +1）
      supersteps  距上次快照过了几个超步（不管写没写都 +1）

    满足任一条件就落一份完整快照：
      updates >= snapshot_frequency
      supersteps >= DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT

    第二个上界很重要：某个通道可能很久不再被写，光靠 updates 会让
    重放深度无限增长。
    """

    __slots__ = ("value", "reducer", "typ", "snapshot_frequency")

    def __init__(self, reducer: Reducer | None = None, typ: Any = None, *,
                 snapshot_frequency: int = 1000):
        """
        `reducer` —— 合并规则。两种都接受：
          - 二元 reducer `(old, new) -> merged`（add_messages / operator.add）
          - 批量 reducer `(state, list[writes]) -> new_state`（BatchReducer）

        `typ` —— 值的类型，用来算零值初值（list->[]）。拿不到就当 list。

        签名刻意对齐 LangGraph 的 `DeltaChannel(reducer, typ, *,
        snapshot_frequency=1000)`。
        """
        if snapshot_frequency <= 0:
            raise ValueError(
                f"snapshot_frequency 必须是正整数，收到 {snapshot_frequency}"
            )
        self.key = ""
        self.reducer = reducer
        self.snapshot_frequency = snapshot_frequency
        self.typ = typ if typ is not None else list
        self.value: Any = MISSING

    # -- 读 ---------------------------------------------------------------
    def get(self) -> Any:
        if self.value is MISSING:
            raise EmptyChannelError()
        return self.value

    def is_available(self) -> bool:
        return self.value is not MISSING

    def copy(self) -> "DeltaChannel":
        new = self.__class__(self.reducer, self.typ,
                             snapshot_frequency=self.snapshot_frequency)
        new.key = self.key
        new.value = self.value
        return new

    # -- 写 ---------------------------------------------------------------
    def update(self, values: Sequence[Any]) -> bool:
        """
        把这一批写入合并进当前值。

        与 BinaryOperatorAggregate 的区别：这里**一次 reducer 调用**处理
        整批，而不是逐条。LangGraph 的 DeltaChannel reducer 签名是
        `(state, list[writes]) -> new_state`。

        但 minigraph 现有的 reducer 签名是**二元**的 `(old, new) -> merged`。
        为了兼容两套签名，这里做一层适配：

          - reducer 是"二元 reducer"（比如 add_messages / operator.add）
            -> 逐条调用：reducer(reducer(state, v1), v2) ...
          - reducer 是"批量 reducer"（自己接受 list）
            -> 一次调用：reducer(state, values)

        怎么区分？批量 reducer 是用 `DeltaChannel` 显式构造时传进来的，
        而 State 注解里写的是二元 reducer。两种都支持，靠 arity 判断太脆
        （内置函数 arity 不可靠），所以这里约定：
        **批量 reducer 必须是 `BatchReducer` 的实例**，二元 reducer 是普通
        callable。
        """
        if not values:
            return False
        base = self.typ() if self.value is MISSING else self.value
        if isinstance(self.reducer, BatchReducer):
            self.value = self.reducer.apply(base, list(values))
        else:
            for v in values:
                self.value = self.reducer(base, v)
                base = self.value
        return True

    # -- 重放（从 checkpoint 恢复） ---------------------------------------
    def replay_writes(self, writes: Sequence[Any]) -> None:
        """
        把祖先的写入按从旧到新的顺序重放一遍，重建当前值。

        `writes` 是这一段（最近快照之后）积累的写入值，已按时间排好序。
        分批调用 `update` —— 每批一次 reducer 调用，和原运行一致。
        """
        if not writes:
            return
        base = self.value
        if isinstance(self.reducer, BatchReducer):
            if base is MISSING:
                base = self.typ()
            self.value = self.reducer.apply(base, list(writes))
        else:
            for v in writes:
                if base is MISSING:
                    base = self.typ()
                base = self.reducer(base, v)
            self.value = base

    # -- 持久化 -----------------------------------------------------------
    def checkpoint(self) -> Any:
        """
        永远返回 MISSING —— checkpoint 里**不存**这个通道的值。

        真正的快照由 runtime 决定时机，直接把 `_DeltaSnapshot(ch.get())`
        写进 channel_values；非快照步则这个通道根本不出现，
        恢复时靠重放祖先写入重建。
        """
        return MISSING

    def from_checkpoint(self, checkpoint: Any) -> "DeltaChannel":
        """
        从 checkpoint 恢复。

        三种 blob：
          MISSING            -> 空（调用方接着重放写入）
          _DeltaSnapshot     -> 直接用快照里的值
          普通值             -> 老版本 BinaryOperatorAggregate 迁移过来的
        """
        new = self.__class__(self.reducer, self.typ,
                             snapshot_frequency=self.snapshot_frequency)
        new.key = self.key
        if isinstance(checkpoint, DeltaSnapshot):
            new.value = checkpoint.value
        elif checkpoint is not MISSING:
            new.value = checkpoint
        else:
            new.value = self.typ()
        return new


class BatchReducer:
    """
    标记一个"批量 reducer"：签名是 `(state, list[writes]) -> new_state`。

    minigraph 现有的 reducer 是二元的 `(old, new) -> merged`。
    DeltaChannel 两种都支持，用这个类做区分 —— 靠 arity 判断不可靠
    （内置函数的 arity 看不了），用一个显式的标记类最稳。

    对齐 LangGraph：它的 DeltaChannel reducer **就是**批量的，
    而 BinaryOperatorAggregate 的 reducer 是二元的。
    """

    __slots__ = ("fn",)

    def __init__(self, fn):
        self.fn = fn

    def apply(self, state: Any, values: list[Any]) -> Any:
        return self.fn(state, values)

    def __call__(self, state: Any, values: list[Any]) -> Any:
        return self.fn(state, values)


class DeltaSnapshot:
    """
    checkpoint 里存的一份完整快照。

    语义对齐 LangGraph 的 `_DeltaSnapshot`（langgraph/checkpoint/serde/types.py）：
    它是"祖先重放的终点"——重放走到这个 blob 就停，状态从快照直接恢复，
    不用再往更早的祖先走。
    """

    __slots__ = ("value",)

    def __init__(self, value: Any):
        self.value = value

    def __repr__(self) -> str:  # pragma: no cover
        return f"<DeltaSnapshot {type(self.value).__name__}>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DeltaSnapshot) and other.value == self.value


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

    "上一超步有人写过 Send,返回true"
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

def channel_for_field(field: str, reducer: Reducer | None,
                      typ: Any = None) -> BaseChannel:
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

    `typ` 是字段的类型（`Annotated[list, add]` 里的 `list`），只给
    BinaryOperatorAggregate 用：它决定通道的初值 `typ()`。
    """
    if reducer is None or reducer is LAST_VALUE:
        return LastValue(field)
    return BinaryOperatorAggregate(field, reducer, typ)
