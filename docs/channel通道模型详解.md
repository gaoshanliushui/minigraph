# 通道（channel）模型详解：什么时候 `is_available()` 返回真

> 本文对应 `minigraph/channels.py`。内核已经不是"边列表调度"，而是
> **channel 写入 + 通道版本 + 节点水位线**。
>
> 配套阅读：《`runtime超步循环详解.md`》（讲超步循环怎么用这些通道）、
> `minigraph/channels.py`（实现）、`scripts/channel_availability.py`（可运行的实测）。

---

## 0. 一句话定位

`is_available()` 只回答一件事：

> **这个通道现在有没有"可读的值"。**

它和"版本号新不新"是**两条独立的轴**。节点被触发要两个同时成立
（`runtime.py:617-640`）：

```python
channels[chan].is_available()              # 轴一：有信号吗
and versions[chan] > seen[node][chan]      # 轴二：这信号是新来的吗
```

`is_available()` 管轴一。轴二由通道版本号 + 节点水位线管。

---

## 1. 通用机制：可用性只由 `update()` 改变

所有通道的 `update(values)` 都返回 bool，含义统一（`channels.py:82-87`）：

> **返回 True = 通道的可用性（或内容）确实变了，值得推版本号。**

所以"可用性变化"和"版本号推进"是绑在一起的：`_apply_writes`（`runtime.py:698-739`）
只在 `update()` 返回 True 时才写 `versions[chan] = version`。

而 `update([])`（空写入）在不同类型里含义**完全不同**，这是后面所有行为差异的根源：

| 通道 | `update([])` 的含义 | 返回 |
| --- | --- | --- |
| `LastValue` / `BinaryOperatorAggregate` | **no-op**（数据留着） | False |
| `EphemeralValue` | **自清**（把值抹掉） | True（若原来有值） |
| `Topic(accumulate=False)` | **清空队列** | True（若原来非空） |
| `Topic(accumulate=True)` | no-op（继续累积） | False |
| `NamedBarrierValue` | **no-op**（seen 不变） | False |

---

## 2. 逐类型详解

### 2.0 总表

| 通道类型 | 用在哪 | 初始 | 写之后 | `update([])` 之后 | 什么时候变回 False |
| --- | --- | --- | --- | --- | --- |
| `LastValue` | 无 reducer 的状态字段 | False | True | **仍 True** | **从不** |
| `BinaryOperatorAggregate` | 有 reducer 的状态字段 | False | True | **仍 True** | **从不** |
| `EphemeralValue` | `branch:to:X` 分支触发 | False | True（**哪怕值是 None**） | **False** | 本超步没人写它 → 自清 |
| `Topic(accumulate=False)` | `__pregel_tasks` Send 队列 | False | True | **False** | 同上 |
| `Topic(accumulate=True)` | 通用发布订阅 | False | True | **仍 True** | 从不 |
| `NamedBarrierValue` | `join:a+b:c` 屏障 | False | **集齐才 True** | 仍 False | `consume()` 重新武装后 |

实测输出见文末 §7。

### 2.1 `LastValue`（`channels.py:166`）/ `BinaryOperatorAggregate`（`channels.py:221`）

```python
def is_available(self) -> bool:
    return self.value is not MISSING
```

判据只有一个：**值不再是 `MISSING` 哨兵**。写进去过一次就永远可用——因为
`update([])` 对它们是 no-op（`channels.py:149-150`、`206-207`），而超步末尾的
自清步骤（`runtime.py:190`）只在 `update([])` 返回 True 时才动手。

**含义**：状态字段一旦被写过，就一直在后续快照里；`update([])` 返回 False →
不推版本号 → 这个数据通道的版本号停在原地。

> `LastValue` 与 `BinaryOperatorAggregate` 的区别不在可用性，而在**合并语义**：
> 前者一步只收一个值（收多个直接报 `ConcurrentWriteError`），后者按 reducer 累积。

### 2.2 `EphemeralValue`（`channels.py:277`）—— 分支触发通道

判据同上（`value is not MISSING`），但它**能被清掉**（`channels.py:257-262`）：

```python
def update(self, values):
    if len(values) == 0:
        if self.value is not MISSING:
            self.value = MISSING
            return True        # <- 自清，且"可用性变了"
        return False
    ...
```

**这就是"分支信号只活一个超步"的实现**（时间线见 §4）。

关键点：**写进去的值是 `None`，但仍然算"可用"**。分支通道的信号不是"值"，
而是"被写过"这个事实——所以 `is_available()` 判的是"有没有被写过"，
不是"值是不是 None"。

> `guard=True/False` 只影响"一步收到多个值"时是否报错（`channels.py:264-268`），
> 与可用性无关。分支通道用 `guard=False`，因为两个并行节点同时
> `Command(goto="x")` 是合法的。

### 2.3 `Topic`（`channels.py:331`）—— `__pregel_tasks` Send 队列

```python
def is_available(self) -> bool:
    return bool(self.values)        # 队列非空
```

`accumulate=False`（minigraph 的实际配置）时，`update()` **先清空再收**
（`channels.py:307-312`）：

```python
if not self.accumulate:
    if self.values: updated = True
    self.values = []       # <- 先清空
...
self.values.extend(flat)  # <- 再收本步的
```

所以队列里**只有本超步写的 Send**，旧 Send 不会重复触发任务。

`update([])` 对它也是自清（`self.values = []`，返回 True）——队列同样只活一个超步。

`accumulate=True` 时则是真发布订阅：`values` 只增不减，`update([])` 不清空。

### 2.4 `NamedBarrierValue`（`channels.py:382`）—— 多源汇聚屏障

```python
def is_available(self) -> bool:
    return self.seen == self.names   # 集齐了才算可用
```

这是唯一一个**"可用"不等于"有值"**的类型：

| 状态 | `seen` | `is_available()` | `get()` |
| --- | --- | --- | --- |
| 没人到 | `{}` | False | 抛 `EmptyChannelError` |
| a 到了 | `{"a"}` | False | 抛 `EmptyChannelError` |
| a、b 都到 | `{"a","b"}` | True | 返回 `None` |

两个细节：

1. **半满的屏障不会被自清误伤**。`update([])` 的循环体一次都不执行 → 返回 False
   → `seen` 不变（`channels.py:365-375`）。否则 a、b 分两步到达时，b 的信号
   会在中间被抹掉。
2. **满的屏障是被 `consume()` 重新武装的，不是被自清清掉的**。`consume()`
   （`channels.py:385-389`）在订阅者跑完后把 `seen` 清空 → 重新武装下一轮。
   这个调用在"消费触发通道"步骤（`runtime.py:160-179`），**早于**自清步骤，
   顺序上不会打架。

### 2.5 `BaseChannel` 默认实现（`channels.py:94-100`）

```python
def is_available(self) -> bool:
    try:
        self.get()
    except EmptyChannelError:
        return False
    return True
```

所有具体子类都覆盖了它，所以这条只是"自己写新通道类型时的默认约定"：
**`get()` 不抛异常就算可用**。

---

## 3. 三类通道的"可用期"对比

| | 数据通道<br>(LastValue / BinaryOp) | 触发通道<br>(EphemeralValue) | 屏障通道<br>(NamedBarrier) |
| --- | --- | --- | --- |
| 何时变可用 | 第一次被写 | 被写（哪怕写 None） | 所有来源到齐 |
| 何时变不可用 | **从不** | 本超步没人写它 → 自清 | `consume()` 之后（重新武装） |
| 活多久 | 整个运行 | **恰好一个超步** | 从集齐到被消费 |
| 参与触发吗 | ❌ 不参与 | ✅ | ✅ |
| 会被 `update([])` 影响吗 | ❌ no-op | ✅ 被清 | ❌ no-op（满的靠 consume 清） |

---

## 4. 超步时间线

### 4.1 分支通道恰好活一个超步

```
step N 结束：A 写 branch:to:B   -> 可用=True, 版本=v
step N+1 开始：算触发            -> 可用=True -> B 被触发
step N+1 结束：B 没写 branch:to:B -> 自清 -> 可用=False, 版本=v'
step N+2 开始：算触发            -> 可用=False -> B 不再被触发
```

"可用"精确覆盖了"上一超步刚写、这一超步还能读到"这个窗口。

### 4.2 屏障的完整生命周期

```
step N   ：a、b 各自写 join:a+b:c = 自己的名字   -> seen={a,b} -> 可用
step N+1 ：c 被触发；c 跑完，consume() 把 seen 清空 -> 重新武装 -> 不可用
step N+2 ：若 a、b 又各写一次 -> seen={a,b} -> 又可用 -> c 再触发一次
```

屏障"重新武装"而不是"永久失效"，是多源汇聚边能出现在循环里的前提。

---

## 5. 可用性除了触发，还管三件事

**① 快照里有没有这个字段**（`runtime.py:747`）

```python
state = {f: channels[f].get() for f in reducers
         if f in channels and channels[f].is_available()}
```

没写过的状态字段在节点眼里是**不存在**（`s["x"]` 抛 KeyError），不是 `None`。
这个区别很实际：节点要区分"没这个字段"和"这个字段是 None"。

**② checkpoint / `get_state` 的状态投影**（`runtime.py:763`）—— 同理，
只含可用的数据通道；`branch:to:*` / `join:*` / `__pregel_tasks` 这些触发通道
不会泄漏到用户状态里。

**③ 自清时要不要动它**（`runtime.py:190`）：

```python
if ch.is_available() and ch.update([]):
    versions[chan] = next_version
```

只有"可用 **且** `update([])` 真的改变了它"才推版本号。所以：

- 数据通道：版本号停在原地（不会无谓地唤醒别人）
- 触发通道：被抹掉时版本推进（记录"这个信号死了"）

---

## 6. 两个易错点

**`None` 是合法值。** `LastValue` 存了 `None` → `value is not MISSING` → 可用。
分支通道正是靠这个成立（值是 `None`，意义在"被写过"）。**`MISSING` 才是"空"。**

**可用 ≠ 会触发。** 一个数据通道永远可用，但它不在任何节点的 `triggers` 里，
所以永远不触发任何东西。反过来，一个分支通道可能**可用但版本旧**
（`versions[chan] == seen[node][chan]`）——这时也不触发。

典型例子：`advance_watermarks`（`runtime.py:135-158`）会把节点**所有**触发通道的
水位线都推到当前版本。所以一个节点可能因为通道 C1 被触发，却同时把 C2 的水位线
也推高了——C2 之后就不会再触发它。

> 这条也是"候选集缩小不会漏触发"的依据：节点跑完后所有触发通道的水位线都追平了，
> 所以"还有通道比我新"⟺"有通道在我上次跑之后变过"⟺"它必然在 `updated_channels` 里"
> ⟺"我必然在候选集里"。

---

## 7. 怎么自己验

```bash
.venv/bin/python scripts/channel_availability.py
```

输出：

```
=== LastValue（无 reducer 的状态字段）===
    初始                                 is_available=False  值='<空>'
    update(['x'])                      is_available=True   值='x'
    update([])  —— 没人写它                is_available=True   值='x'
    update(['y'])                      is_available=True   值='y'

=== BinaryOperatorAggregate（有 reducer 的状态字段）===
    初始                                 is_available=False  值='<空>'
    update([1])  —— 首次直接落盘             is_available=True   值=1
    update([2])  —— reducer 合并         is_available=True   值=3
    update([])  —— 没人写它                is_available=True   值=3

=== EphemeralValue（分支触发通道 branch:to:X）===
    初始                                 is_available=False  值='<空>'
    update([None])  —— 值是 None         is_available=True   值=None
    update([])  —— 自清                  is_available=False  值='<空>'
    再次 update([None])                  is_available=True   值=None

=== Topic（__pregel_tasks 的 Send 队列）===
    初始                                 is_available=False  值='<空>'
    update([1,2])                      is_available=True   值=[1, 2]
    update([])  —— 自清（只留本步的值）          is_available=False  值='<空>'
    accumulate=True 写两次                is_available=True   值=[1, 2]

=== NamedBarrierValue（多源汇聚 join:a+b:c）===
    初始（没人到）                            is_available=False  值='<空>'
    a 到了（还差 b）                         is_available=False  值='<空>'
    b 到了（齐了）                           is_available=True   值=None
    consume() -> True
    consume() 之后（重新武装）                 is_available=False  值='<空>'

=== BaseChannel 默认实现（is_available 走 get()）===
    初始                                 is_available=False  值='<空>'
    update(['v'])                      is_available=True   值='v'
```

想在真实图里看通道状态，加 `config={"debug": True}`，每个超步会打印
`versions` / `seen` / `avail` 三张表（`runtime.py:196-203`）。

---

## 8. 与真实 LangGraph 的对应

| minigraph | LangGraph | 说明 |
| --- | --- | --- |
| `LastValue` | `LastValue` | 无 reducer 的字段；一步只收一个值 |
| `BinaryOperatorAggregate` | `BinaryOperatorAggregate` | 带 reducer 的字段 |
| `EphemeralValue` | `EphemeralValue` | 分支触发通道；值无意义，重要的是"被写过" |
| `Topic` | `Topic` | `__pregel_tasks`；发布订阅 |
| `NamedBarrierValue` | `NamedBarrierValue` | `add_edge([a,b], c)` 的多源汇聚 |
| `MISSING` | `MISSING` | "空"哨兵；`None` 是合法值 |

对照源码：`langgraph/channels/base.py`、`ephemeral_value.py`、`last_value.py`、
`binop.py`、`topic.py`、`named_barrier_value.py`。

`finish()`（`channels.py:107`）在 minigraph 里**从没被调用**——它是对齐
LangGraph `BaseChannel` API 预留的钩子。

---

## 9. 速查

```text
可用性判据（按类型）
  LastValue / BinaryOperatorAggregate : value is not MISSING        —— 写过就一直有
  EphemeralValue                      : value is not MISSING        —— 写过，本超步内有效
  Topic                               : bool(values)                —— 队列非空
  NamedBarrierValue                   : seen == names               —— 来源集齐
  BaseChannel（默认）                  : get() 不抛 EmptyChannelError

update([]) 的含义
  LastValue / BinaryOp / Barrier      : no-op（不清）
  EphemeralValue / Topic(accum=False) : 自清

触发判定（可用性只是其中一条）
  is_available()  AND  versions[chan] > seen[node][chan]
```
