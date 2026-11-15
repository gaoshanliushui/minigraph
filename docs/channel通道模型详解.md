# 通道（channel）模型详解：什么时候 `is_available()` 返回真

> 本文对应 `minigraph/channels.py`。内核已经不是"边列表调度"，而是
> **channel 写入 + 通道版本 + 节点水位线**。
>
> 配套阅读：《`runtime超步循环详解.md`》（讲超步循环怎么用这些通道）、
> 《`versions与seen水位线详解.md`》（版本号与水位线什么时候被改）、
> `minigraph/channels.py`（实现）、`scripts/channel_availability.py`（可运行的实测）。

---

## 0. 一句话定位

`is_available()` 只回答一件事：

> **这个通道现在有没有"可读的值"。**

它和"版本号新不新"是**两条独立的轴**。节点被触发要两个同时成立
（`runtime.py:852-875`）：

```python
channels[chan].is_available()              # 轴一：有信号吗
and versions[chan] > seen[node][chan]      # 轴二：这信号是新来的吗
```

`is_available()` 管轴一。轴二由通道版本号 + 节点水位线管。

---

## 1. 通用机制：可用性只由 `update()` 改变

所有通道的 `update(values)` 都返回 bool，含义统一（`channels.py:82-87`）：

> **返回 True = 通道的可用性（或内容）确实变了，值得推版本号。**

所以"可用性变化"和"版本号推进"是绑在一起的：`_apply_writes`（`runtime.py:933-969`）
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

## 1.5 可用性什么时候被更改：按源码逐个入口

> 本节回答："**这个通道的 `is_available()` 什么时候会从 False 变 True、
> 什么时候会变回 False？**"
>
> 结论先行：**能改变可用性的只有三个方法** —— `update()`、`consume()`、`finish()`。
> 而在 minigraph 的实际运行里，`finish()` **从没被调用过**，
> `consume()` **只有屏障通道有实质行为**。
>
> 所以真正的问题收敛成：**`update()` 在什么时机被调用、传进去什么**。

### 1.5.1 三个入口

| 方法 | 谁能调它 | minigraph 里实际调用的地方 | 会不会改可用性 |
| --- | --- | --- | --- |
| `update(values)` | `_apply_writes`（正常写入）、`expire_unwritten`（空写自清）、`_write_state_channels`（写数据通道）、`_begin`（开局写入口信号） | 见 §1.5.2 | **会**（取决于类型和入参） |
| `consume()` | `consume_triggers`（订阅者跑完后） | `runtime.py:175` | **只有屏障会**（重新武装） |
| `finish()` | —— | **从未调用** | 不会 |

`BaseChannel.consume()` 和 `BaseChannel.finish()` 的默认实现都返回 `False`
（`channels.py:103-109`）—— 即"什么都没变"。整个 `channels.py` 里
**只有 `NamedBarrierValue` 覆盖了 `consume()`**（`channels.py:397-402`）。

### 1.5.2 `update()` 的四个调用点

按调用时机排：

| # | 调用点 | 代码 | 传什么 | 影响谁 |
| --- | --- | --- | --- | --- |
| ① | **开局** `__start__` 点亮入口 | `rt.channels[chan].update([None])`（`runtime.py:574`） | `[None]` | 入口点的 `branch:to:X` → **变可用** |
| ② | **本步写入落地** `_apply_writes` | `ch.update(vals)`（`runtime.py:956`） | 本步该通道的**全部**写入值 | 数据通道 + 被写的触发通道 → **变可用** |
| ③ | **写状态字段** `_write_state_channels` | `ch.update([val])`（`runtime.py:1021`） | 单个值 | 仅数据通道 → **变可用** |
| ④ | **超步末尾自清** `expire_unwritten` | `ch.update([])`（`runtime.py:190`） | **空列表** | 没人写的通道 → **变不可用** |

看 ③ 这一处有个容易忽略的细节：

```python
ch.update([val])          # _write_state_channels，runtime.py:1021
```

**它不看返回值。** 所以写数据通道**不会**推版本号 —— 这和 ② 不同
（② 里 `changed = ch.update(vals)` 的返回值被用来决定推不推版本）。
数据通道本来就不参与触发判定，不推版本没有影响。

### 1.5.3 每个通道类型的可用性状态机

下表是**逐类型、从源码读出来的**状态机。
"变可用"指 `is_available()` 从 False 变 True，"变不可用"反之。

#### `LastValue`（数据字段，无 reducer）

```text
                    update([v])              永远
    False ──────────────────────────> True ─────────────> True
```

| 事件 | 可用性 | 说明 |
| --- | --- | --- |
| 构造 | False | `value = MISSING`（`channels.py:146`） |
| `update([v])` | **False → True** | `value = values[-1]`（`channels.py:158`） |
| `update([])` | 不变 | 直接 `return False`（`channels.py:149-150`） |
| `update([v1, v2])` | 抛异常 | `ConcurrentWriteError`（`channels.py:151-157`） |

**永不变回 False。** 一旦写过就一直可用。

#### `BinaryOperatorAggregate`（数据字段，有 reducer）

```text
    构造（有 typ）  →  True（初值 typ()）──> 永远 True
    构造（无 typ）  →  False ──update([v])──> True ──> 永远 True
```

| 事件 | 可用性 | 说明 |
| --- | --- | --- |
| 构造，**有** `typ` | **True** | 初值 = `typ()`，list→`[]`、int→`0`（`channels.py:209-214`） |
| 构造，**无** `typ` | False | 造不出零值 → 退化成 `MISSING`（`channels.py:212-214`） |
| `update([v])` | **→ True** | 首次写入直接落盘，之后走 reducer 合并（`channels.py:217-226`） |
| `update([])` | 不变 | 返回 False（`channels.py:218-219`） |

> ⚠️ **初值这件事容易被忽略**：真实编译出来的图里，带 reducer 的字段
> **一开局就可用**（`typ` 从 `Annotated[T, reducer]` 里取出来传进去）。
> 实测：
>
> ```text
>   items    BinaryOperatorAggregate    value=[]       is_available=True
>   n        BinaryOperatorAggregate    value=0        is_available=True
>   plain    LastValue                  value=<missing> is_available=False
> ```
>
> 所以 `get_state()` 拿到的状态里，**带 reducer 的字段从一开始就有零值**，
> 不带 reducer 的字段则"没有就是没有"。

#### `EphemeralValue`（`branch:to:X` 分支触发通道）

```text
    False ──update([None])──> True ──update([])──> False ──update([None])──> True
                ↑                              ↑
          哪怕值是 None                  自清（超步末尾）
```

| 事件 | 可用性 | 说明 |
| --- | --- | --- |
| 构造 | False | `value = MISSING`（`channels.py:267`） |
| `update([None])` | **False → True** | `value = values[-1]`；**值是 None 也算可用**（`channels.py:281`） |
| `update([])` | **True → False** | 自清（`channels.py:270-275`） |
| `update([])`（已是空） | 不变 | 返回 False（`channels.py:274`）—— **不推版本号** |

**这是唯一"生灭对称"的触发通道**：写进去变可用，没人写就变不可用。

> `guard=True/False` 只影响"一步收到多个值"时是否报错（`channels.py:276-279`），
> 与可用性无关。分支通道用 `guard=False`，因为两个并行节点同时
> `Command(goto="x")` 是合法的。

#### `Topic`（`__pregel_tasks` Send 队列）

```text
    accumulate=False:
    False ──update([...])──> True ──update([])──> False

    accumulate=True:
    False ──update([...])──> True ─────────────> True（只增不减）
```

| 事件 | 可用性 | 说明 |
| --- | --- | --- |
| 构造 | False | `values = []`（`channels.py:317`） |
| `update([...])` | **→ True** | `accumulate=False` 时**先清空再收**（`channels.py:319-324`） |
| `update([])`，`accumulate=False` | **True → False** | 清空队列（`channels.py:320-323`） |
| `update([])`，`accumulate=True` | 不变 | 继续累积（`channels.py:319`） |

`accumulate=False` 时队列里**只有本超步写的 Send** —— 旧 Send 不会重复触发任务。

#### `NamedBarrierValue`（`join:a+b:c` 屏障）

```text
    False ──update(["a"])──> False ──update(["b"])──> True
                                        ↑
    False <──consume()──────────────────┘   （重新武装）
```

| 事件 | 可用性 | 说明 |
| --- | --- | --- |
| 构造 | False | `seen = set()`（`channels.py:375`） |
| `update(["a"])`（还差 b） | **不变（False）** | `seen` 加了 a，但没集齐（`channels.py:377-387`） |
| `update(["b"])`（集齐） | **False → True** | `seen == names`（`channels.py:394`） |
| `update([])` | **不变** | 循环体不执行 → `seen` 不变（`channels.py:378-380`） |
| `consume()`（已集齐） | **True → False** | `seen = set()`，重新武装（`channels.py:397-402`） |
| `consume()`（没集齐） | 不变 | 返回 False |

**这是唯一一个"可用 ≠ 有值"的类型**：

| 状态 | `seen` | `is_available()` | `get()` |
| --- | --- | --- | --- |
| 没人到 | `{}` | False | 抛 `EmptyChannelError` |
| a 到了 | `{"a"}` | False | 抛 `EmptyChannelError` |
| a、b 都到 | `{"a","b"}` | True | 返回 `None` |

两个细节：

1. **半满的屏障不会被自清误伤**。`update([])` 的循环体一次都不执行 → 返回 False
   → `seen` 不变。否则 a、b 分两步到达时，b 的信号会在中间被抹掉。
2. **满的屏障是被 `consume()` 重新武装的，不是被自清清掉的**。`consume()`
   在"消费触发通道"步骤（`runtime.py:173-178`），**早于**自清步骤
   （`expire_unwritten`），顺序上不会打架。

### 1.5.4 一次超步里，可用性的完整变化顺序

`_advance`（`runtime.py:715-734`）的**四步顺序不能换**，
它决定了可用性在什么时候被改：

```text
① advance_watermarks(tasks)      只动 seen，不动通道
② next_version()                 算下一个版本号
③ consume_triggers(tasks, v)     屏障被 consume() 重新武装  ← 唯一让可用性变 False 的"写"
④ _apply_writes(..., v)          本步写入落地 → 可用性变 True
⑤ expire_unwritten(written, v)   没人写的通道 update([]) 自清 → 可用性变 False
```

对应到可用性：

| 步 | 谁变可用 | 谁变不可用 |
| --- | --- | --- |
| ③ `consume_triggers` | —— | **已集齐的屏障**（`consume()` 重新武装） |
| ④ `_apply_writes` | **本步被写的通道** | —— |
| ⑤ `expire_unwritten` | —— | **本步没人写、但还可用着的通道** |

所以一个超步结束时：

- **被写过的通道** → 可用（下一步能被读到）
- **没被写过的通道** → 不可用（被自清）
- **已集齐的屏障** → 不可用（被重新武装，等下一轮）

这正是"分支通道恰好活一个超步"的实现（§5.1）。

> ⚠️ 注意 ③ 在 ④ **之前**：屏障先重新武装，再应用本步写入。
> 顺序反了的话，屏障会在"本步有人写"的情况下被误判成"没人写"而清空。

### 1.5.5 实测

```bash
.venv/bin/python scripts/exp_availability_lifecycle.py
```

逐类型打印每一步的 `is_available()`，覆盖：

- `update()` 返回 True 但可用性没翻转（`Topic` 二次写入）
- `update([])` 对数据通道是 no-op
- `EphemeralValue` 写 `None` 也算可用
- `EphemeralValue` 二次自清返回 False（不推版本）
- `NamedBarrierValue` 的 `consume()` 是唯一让它变回不可用的入口
- `LastValue` 写同一个值也返回 True（内容变了 ≠ 可用性变了）

另有两个现成脚本：

```bash
.venv/bin/python scripts/channel_availability.py    # 逐类型打印可用性
.venv/bin/python scripts/channel_inventory.py        # 真实图里开局各通道的可用性
```

---

## 2. 通道清单：一张图里到底有哪些通道

通道不是用户写的，是**编译期自动生成**的（`builder.py:152-214`）。
一张图的通道表 = **状态字段数 + 节点数 + 屏障边数 + 2**。

以这张图为例：

```python
class S(State):
    n:     Annotated[int, operator.add]    # 有 reducer
    msgs:  Annotated[list, add_messages]    # 有 reducer
    plain: str                             # 无 reducer

g.add_edge(START, "a")
g.add_edge("a", "b")
g.add_edge("b", "c")
g.add_edge(["b", "c"], "d")              # 屏障边
g.add_conditional_edges("d", ...)         # 条件边
```

编译产物（`scripts/channel_inventory.py` 的真实输出）：

| 通道名 | 类型 | 角色 | 谁写它 | 谁订阅它 |
| --- | --- | --- | --- | --- |
| `n` | `BinaryOperatorAggregate` | 数据通道（状态字段） | 节点写 `{"n": ...}` | 无（只被读） |
| `msgs` | `BinaryOperatorAggregate` | 数据通道（状态字段） | 节点写 `{"msgs": ...}` | 无（只被读） |
| `plain` | `LastValue` | 数据通道（状态字段） | 节点写 `{"plain": ...}` | 无（只被读） |
| `branch:to:a` | `EphemeralValue` | 触发通道（分支信号） | `__start__` / 前驱节点 | `a` |
| `branch:to:b` | `EphemeralValue` | 触发通道（分支信号） | `a` | `b` |
| `branch:to:c` | `EphemeralValue` | 触发通道（分支信号） | `b` | `c` |
| `branch:to:d` | `EphemeralValue` | 触发通道（分支信号） | 条件边（运行时决定） | `d` |
| `__start__` | `EphemeralValue` | 触发通道（入口伪节点占位） | （没有人写） | `__start__` |
| `join:b+c:d` | `NamedBarrierValue` | 触发通道（多源汇聚屏障） | `b` 写 `"b"`、`c` 写 `"c"` | `d` |
| `__pregel_tasks` | `Topic` | 触发通道（PUSH 轨 / Send 队列） | 写 `Send(...)` 的节点 | 调度器 |

### 2.1 数据通道（每个状态字段一个）

| 通道名 | 类型 | 什么时候是这种 | 合并语义 |
| --- | --- | --- | --- |
| `<字段名>` | `LastValue` | 字段没写 `Annotated`，或 reducer 是 `LAST_VALUE` | last-write-wins；一步只收一个值，收多个直接报并发冲突 |
| `<字段名>` | `BinaryOperatorAggregate` | 字段写了 `Annotated[T, reducer]` | `value = reducer(old, new)`；首次写入直接落盘 |

**数据通道只被读，不触发任何人。** 它们出现在 `_read_state()` 生成的快照里，
节点通过 `state["字段名"]` 读到值；它们**不在任何节点的 `triggers` 里**。

判定规则见 `channel_for_field()`（`channels.py:418-423`）：

```text
reducer is None 或 reducer is LAST_VALUE  ->  LastValue
其它（operator.add / add_messages / merge_dict ...） -> BinaryOperatorAggregate
```

> **坑**：`from __future__ import annotations` 会把注解变成**字符串**。
> 如果不把字符串解析回真实类型，`Annotated[int, operator.add]` 会被当成
> "没写 reducer" -> 字段落到 `LastValue` 而不是 `BinaryOperatorAggregate`，
> 于是并行分支写同一字段直接报并发冲突，而不是累加。
> `State.__init_subclass__` 里的 `_own_annotations()` 负责这件事，
> 见《`runtime超步循环详解.md`》与 `tests/test_future_annotations.py`。

### 2.2 触发通道（每个节点 / 每条屏障边 / 全局各一个）

| 通道名 | 类型 | 谁写它 | 谁订阅它 | 语义 |
| --- | --- | --- | --- | --- |
| `branch:to:<节点>` | `EphemeralValue(guard=False)` | 前驱节点的静态边、条件边、`Command(goto=)` | 该节点 | "该跑你了" |
| `join:<a+b>:<目标>` | `NamedBarrierValue` | 各来源节点写**自己的名字** | 目标节点 | "人都到齐了" |
| `__pregel_tasks` | `Topic(accumulate=False)` | 写 `Send(...)` 的节点 | 调度器（PUSH 轨） | "派活单" |
| `__start__` | `EphemeralValue` | （没有人写） | `__start__` 伪节点 | 入口占位 |

几个容易忽略的点：

- **每个节点都有 `branch:to:X`**，哪怕它是终端节点、哪怕没有任何边指向它。
  通道先建好，边只决定"谁写它"。
- **`branch:to:X` 的值永远是 `None`**，有意义的是"被写过"这个事实。
  `guard=False` 是因为两个并行节点同时 `Command(goto="x")` 是合法的。
- **`join:` 通道名 = `join:` + 来源用 `+` 连 + `:` + 目标**。
  `add_edge(["a","b"], "c")` → `join:a+b:c`（`builder.py:196-205`）。
- **一个节点可以订阅多个通道**：既有普通入边又有屏障入边时，`triggers` 里会有两个
  （上例中 `d` 的 `triggers = ['branch:to:d', 'join:b+c:d']`）。**任一通道触发即可**。
- **`__start__` 通道是个占位**：真正发入口信号的是 `__start__` 伪节点在开局
  写 `branch:to:<入口>`；`__start__` 通道自己没人写，`__start__` 节点又被
  `is_start=True` 跳过，所以它永远不产生任务。

### 2.3 节点侧的订阅表

```text
a          triggers=['branch:to:a']                writers=['branch:to:b']
b          triggers=['branch:to:b']                writers=['branch:to:c', 'join:b+c:d']
c          triggers=['branch:to:c']                writers=['join:b+c:d']
d          triggers=['branch:to:d', 'join:b+c:d']  writers=[]  branches=1
__start__  triggers=['__start__']                  writers=['branch:to:a']
```

- `triggers` = 订阅了谁（能被什么唤醒）
- `writers` = 会写谁（静态边 + 屏障边编译来的）
- `branches` = 有几条条件边（运行时才决定写哪个 `branch:to:X`）

反向索引 `trigger_to_nodes`（`runtime.py:309-312`）就是 `triggers` 的转置，
调度时用它从"变化的通道"直接找到候选节点：

```text
branch:to:a -> ['a']
branch:to:b -> ['b']
branch:to:c -> ['c']
branch:to:d -> ['d']
join:b+c:d  -> ['d']
__start__   -> ['__start__']
```

### 2.4 数据通道 vs 触发通道

| | 数据通道 | 触发通道 |
| --- | --- | --- |
| 谁生成 | 每个状态字段一个 | 每个节点 / 每条屏障边 / 全局队列 |
| 命名 | 字段名 | `branch:to:X` / `join:a+b:c` / `__pregel_tasks` |
| 值有意义吗 | ✅ 就是状态本身 | ❌ 只有"被写过"有意义 |
| 参与触发吗 | ❌ | ✅ |
| 参与状态快照吗 | ✅ | ❌（且不会泄漏到 `get_state()`） |
| 活多久 | 整个运行 | 一个超步（屏障到被消费为止） |

---

## 3. 逐类型详解

### 3.0 总表

| 通道类型 | 用在哪 | 初始 | 写之后 | `update([])` 之后 | 什么时候变回 False |
| --- | --- | --- | --- | --- | --- |
| `LastValue` | 无 reducer 的状态字段 | False | True | **仍 True** | **从不** |
| `BinaryOperatorAggregate` | 有 reducer 的状态字段 | **True**（初值 `typ()`） | True | **仍 True** | **从不** |
| `EphemeralValue` | `branch:to:X` 分支触发 | False | True（**哪怕值是 None**） | **False** | 本超步没人写它 → 自清 |
| `Topic(accumulate=False)` | `__pregel_tasks` Send 队列 | False | True | **False** | 同上 |
| `Topic(accumulate=True)` | 通用发布订阅 | False | True | **仍 True** | 从不 |
| `NamedBarrierValue` | `join:a+b:c` 屏障 | False | **集齐才 True** | 仍 False | `consume()` 重新武装后 |

实测输出见文末 §8。

### 3.1 `LastValue`（`channels.py:166`）/ `BinaryOperatorAggregate`（`channels.py:221`）

```python
def is_available(self) -> bool:
    return self.value is not MISSING
```

判据只有一个：**值不再是 `MISSING` 哨兵**。`update([])` 对它们是 no-op
（`channels.py:149-150`、`206-207`），而超步末尾的自清步骤只在 `update([])`
返回 True 时才动手 —— 所以数据通道一旦有值，就永远可用。

**含义**：状态字段一旦被写过，就一直在后续快照里；`update([])` 返回 False →
不推版本号 → 这个数据通道的版本号停在原地。

> `BinaryOperatorAggregate` 的初值是 `typ()`（list→`[]`、int→`0`），和真实
> LangGraph 一致：于是节点读一个**还没被写过**的字段拿到的是零值，而不是
> `KeyError`。拿不到类型时（裸 dict 形态声明的 schema、类型不可实例化）
> 才退化成 `MISSING`，此时首次写入走「直接落盘」的捷径。
> `LastValue` 没有初值 —— 没有 reducer 的字段就是「没写过就没有」。

> `LastValue` 与 `BinaryOperatorAggregate` 的区别不在可用性，而在**合并语义**：
> 前者一步只收一个值（收多个直接报 `ConcurrentWriteError`），后者按 reducer 累积。

### 3.2 `EphemeralValue`（`channels.py:277`）—— 分支触发通道

判据同上（`value is not MISSING`），但它**能被清掉**（`channels.py:269-274`）：

```python
def update(self, values):
    if len(values) == 0:
        if self.value is not MISSING:
            self.value = MISSING
            return True        # <- 自清，且"可用性变了"
        return False
    ...
```

**这就是"分支信号只活一个超步"的实现**（时间线见 §5）。

关键点：**写进去的值是 `None`，但仍然算"可用"**。分支通道的信号不是"值"，
而是"被写过"这个事实——所以 `is_available()` 判的是"有没有被写过"，
不是"值是不是 None"。

> `guard=True/False` 只影响"一步收到多个值"时是否报错（`channels.py:264-268`），
> 与可用性无关。分支通道用 `guard=False`，因为两个并行节点同时
> `Command(goto="x")` 是合法的。

### 3.3 `Topic`（`channels.py:331`）—— `__pregel_tasks` Send 队列

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

### 3.4 `NamedBarrierValue`（`channels.py:382`）—— 多源汇聚屏障

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
   → `seen` 不变（`channels.py:377-387`）。否则 a、b 分两步到达时，b 的信号
   会在中间被抹掉。
2. **满的屏障是被 `consume()` 重新武装的，不是被自清清掉的**。`consume()`
   （`channels.py:385-389`）在订阅者跑完后把 `seen` 清空 → 重新武装下一轮。
   这个调用在"消费触发通道"步骤（`runtime.py:160-179`），**早于**自清步骤，
   顺序上不会打架。

### 3.5 `BaseChannel` 默认实现（`channels.py:94-100`）

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

## 4. 三类通道的"可用期"对比

| | 数据通道<br>(LastValue / BinaryOp) | 触发通道<br>(EphemeralValue) | 屏障通道<br>(NamedBarrier) |
| --- | --- | --- | --- |
| 何时变可用 | 第一次被写 | 被写（哪怕写 None） | 所有来源到齐 |
| 何时变不可用 | **从不** | 本超步没人写它 → 自清 | `consume()` 之后（重新武装） |
| 活多久 | 整个运行 | **恰好一个超步** | 从集齐到被消费 |
| 参与触发吗 | ❌ 不参与 | ✅ | ✅ |
| 会被 `update([])` 影响吗 | ❌ no-op | ✅ 被清 | ❌ no-op（满的靠 consume 清） |

---

## 5. 超步时间线

### 5.1 分支通道恰好活一个超步

```
step N 结束：A 写 branch:to:B   -> 可用=True, 版本=v
step N+1 开始：算触发            -> 可用=True -> B 被触发
step N+1 结束：B 没写 branch:to:B -> 自清 -> 可用=False, 版本=v'
step N+2 开始：算触发            -> 可用=False -> B 不再被触发
```

"可用"精确覆盖了"上一超步刚写、这一超步还能读到"这个窗口。

### 5.2 屏障的完整生命周期

```
step N   ：a、b 各自写 join:a+b:c = 自己的名字   -> seen={a,b} -> 可用
step N+1 ：c 被触发；c 跑完，consume() 把 seen 清空 -> 重新武装 -> 不可用
step N+2 ：若 a、b 又各写一次 -> seen={a,b} -> 又可用 -> c 再触发一次
```

屏障"重新武装"而不是"永久失效"，是多源汇聚边能出现在循环里的前提。

---

## 6. 可用性除了触发，还管三件事

**① 快照里有没有这个字段**（`runtime.py:976-984`）

```python
state = {f: channels[f].get() for f in reducers
         if f in channels and channels[f].is_available()}
```

没写过的状态字段在节点眼里是**不存在**（`s["x"]` 抛 KeyError），不是 `None`。
这个区别很实际：节点要区分"没这个字段"和"这个字段是 None"。

**② checkpoint / `get_state` 的状态投影**（`runtime.py:987-1000`）—— 同理，
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

## 7. 两个易错点

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

## 8. 怎么自己验

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

想看一张图**编译出了哪些通道**：

```bash
.venv/bin/python scripts/channel_inventory.py
```

它会打印通道表（含类型与角色）、每个节点的 `triggers` / `writers`、
反向索引 `trigger_to_nodes`，以及开局时每个通道的可用性。

---

## 9. 与真实 LangGraph 的对应

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

## 10. 速查

```text
通道清单（一张图的通道表 = 状态字段数 + 节点数 + 屏障边数 + 2）
  <字段名>            LastValue / BinaryOperatorAggregate   数据通道（只被读，不触发）
  branch:to:<节点>    EphemeralValue(guard=False)           触发通道（分支信号）
  join:<a+b>:<目标>   NamedBarrierValue                     触发通道（多源汇聚屏障）
  __pregel_tasks      Topic(accumulate=False)               触发通道（Send 队列）
  __start__           EphemeralValue                        入口伪节点占位

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

两个会静默出错的坑：

```text
Python 3.14 (PEP 649)   注解惰性化   -> 必须读 cls.__annotations__（不是 cls.__dict__）
from __future__ import annotations   注解变字符串 -> 必须解析回真实类型，否则 reducer 全丢
```
