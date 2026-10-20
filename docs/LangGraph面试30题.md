# LangGraph面试30题

# LangGraph 面试题 · 30 题（查缺补漏版）

> 用法：遮住答案先自己说一遍。说不到点上，再看「考点」和「参考答案」。
> 每题末尾的 `↗` 指向 minigraph 源码里的对应实现，方便你回去对照。
> 难度标记：`L1` 概念 · `L2` 原理 · `L3` 踩坑/设计取舍

---

## 第一组：状态模型（Q1–Q6）

### Q1 · 为什么 LangGraph 的节点返回的是「增量」而不是新状态？`L1`

**考点**：State / reducer 模型的核心动机

**参考答案**
节点是**无状态函数**，只返回 `{字段: 新值}` 这一小块 diff，主状态由框架统一合并。这样做换来三件事：

1. **并行安全**——多个节点同时跑，各自的 update 互不覆盖，最后由 reducer 决定合并语义；
2. **可检查点**——每步只存合并后的完整 `values`，不需要序列化「节点对象」；
3. **可回放**——状态演化是 `values + 一串 update` 的确定性函数，重放不依赖节点内部变量。

反面：如果节点返回完整新状态，并行分支就会互相覆盖，`operator.add` 这类累加语义也无处落脚。

↗ `graph.py` L96-133 `State.__init_subclass__`

---

### Q2 · `Annotated[list, operator.add]` 里那个 `operator.add` 到底什么时候被调用、以什么参数被调用？`L1`

**考点**：reducer 的签名与调用时机

**参考答案**
签名是 `reducer(old, new) -> merged`，**在一个超步的全部节点跑完之后**、统一合并时被调用，而不是节点返回时。

关键细节：同一超步多个分支写同一个 key，reducer 会被**连续折叠**，不是一次性传入列表——`values[k] = reducer(values[k], v)` 逐个写进去。所以 `operator.add` 处理的是「标量累加」；要「收集成列表」必须写 `lambda a, b: a + [b]` 或 `add_messages` 这种能处理追加的 reducer。

↗ `graph.py` L644-674 `_merge_writes`

---

### Q3 · 默认 reducer 是 last-write-wins。两个并行节点写同一个**没有 reducer** 的字段，会发生什么？`L2`

**考点**：这是 LangGraph 并行分支最常见的第一个坑

**参考答案**
**直接抛 **`InvalidUpdateError`，报错信息大意是「At key 'xxx': Can receive only one value per step」。

原因不是框架「做不到」，而是「最后写入」在并行下**没有确定语义**——取决于线程调度顺序。框架选择显式报错而不是静默覆盖。修法两条：给它配 reducer，或者让两个分支写不同字段、再在汇合节点里合并。

注意区分：如果两个节点**在不同超步**写同一字段，last-write-wins 是安全的（顺序确定），不会报错。

↗ `graph.py` L664-669

---

### Q4 · `add_messages` 和 `operator.add` 用在 `messages` 字段上，行为差在哪？`L2`

**考点**：消息 reducer 的「按 id 更新」语义

**参考答案**

|   | `operator.add` | `add_messages` |
| --- | --- | --- |
| 追加新消息 | ✅ | ✅ |
| **同 id 消息** | 列表里出现两条 | **原地替换**（就地更新，长度不变） |
| 字符串/字典输入 | 要求两边同类型 | 会自动 `coerce` 成 `HumanMessage` / `AIMessage` |
| 删除消息 | 做不到 | 支持 `RemoveMessage(id=...)` |

`add_messages` 的 id 语义是 HITL（人工编辑消息）和「重放时修正某条消息」的基础：你只推送一条同 id 的新消息，历史长度不变、内容被替换。

实测：连续两步各推一条 `id="k"` 的消息，最终 `messages` 长度为 1，内容为第二次的。

↗ `graph.py` L63-86 `add_messages`

---

### Q5 · 状态 schema 用 `TypedDict` 还是 `dataclass` / Pydantic，对运行时有什么影响？`L2`

**考点**：类型只是给 reducer 收集用的，不是校验器

**参考答案**
**对 reducer 的收集**：框架读的是 `__annotations__` 里的 `Annotated` 元数据，三种写法都能识别。

**对运行时**：`TypedDict` **不做任何运行时校验**，节点返回未声明的字段会被**静默忽略**（实测：schema 只有 `a`，节点返回 `{"b": 1}`，结果里既没有 `b` 也不报错）。Pydantic 模型才能拿到字段级校验和类型转换。

所以「状态字段写错名」这类 bug 在 TypedDict 下不会在编译期/运行期暴露，只会表现为「这个值怎么一直是空的」。工程上建议：给关键状态加 `pydantic` 校验，或至少加一条断言式测试。

↗ `graph.py` L108-117；minigraph 在这里比原版严格（未知字段直接 `ValueError`，L634-641）

---

### Q6 · 一个字段配了 `operator.add`，但初始输入里完全没有这个 key，第一步就写它会怎样？`L2`

**考点**：缺失值与零元的区别

**参考答案**
真实 LangGraph 会把它当作「首次写入」直接采用，不报错（`operator.add` 不会被调用，因为没有 old 值）。结果就是 `{'items': ['a']}`。

坑在于：`operator.add` 对「缺失」没有零元概念。如果 reducer 是 `lambda a, b: a + b` 处理字符串，缺初始值时会变成 `None + "x"` 报 `TypeError`。稳妥做法是**显式给初始值**，或在 reducer 里处理 `None`/`MISSING`。

minigraph 的实现用 `MISSING` 哨兵区分「没有」和「是 None」，比裸 `dict.get()` 精确。

↗ `graph.py` L641 `values.get(key, MISSING)`

---

## 第二组：执行模型（Q7–Q12）

### Q7 · 用一句话解释 Pregel / BSP 超步模型，并说明它保证了什么。`L1`

**考点**：整个执行内核的世界观

**参考答案**
执行被切成一个个**超步（superstep）**，每个超步三件事，**严格串行**：

1. **算**：本批任务并发执行，都读**同一份状态快照**；
2. **合**：本步所有写入一次性用 reducer 合并进主状态；
3. **路由**：根据合并后的状态决定下一批任务。

保证的是**确定性**：结果与线程调度顺序无关。代价是每个超步有同步屏障，慢节点拖住整步。

↗ `graph.py` L424-503 主循环

---

### Q8 · 同一个超步里 A、B 两个节点并行，A 写了 `x=1`，B 读 `x`。B 能读到 1 吗？`L2`

**考点**：快照隔离

**参考答案**
**不能。** B 读到的是本超步开始时的快照，`x` 还是旧值。实测构造：A 返回 `v += 100`，B 打印它看到的 `v`，B 看到的是 `0`。

这是刻意设计——只有这样，并行分支才不需要考虑彼此的先后。要让 B 看到 A 的结果，必须拆成两个超步（A → 汇合节点 → B）。

实际影响：所有「边跑边读上游结果」的写法在并行图里都不成立，必须显式加中间节点。典型翻车场景是「分支 A 生成检索词、分支 B 拿检索词去查库」——看着像并行，其实 B 永远查到空。

↗ `graph.py` L432 `snapshot = dict(values)`

---

### Q9 · `recursion_limit` 限制的到底是什么？它和「循环次数」是一回事吗？`L2`

**考点**：一个很多人说错的细节

**参考答案**
它限制的是**超步数**（superstep），不是某个节点的执行次数，也不是「回到同一个节点几次」。

因为一个超步里可能并行跑 5 个节点，也可能只跑 1 个，所以「limit=25」不代表「最多循环 25 次」。线性图里两者数值相同（每步一个节点），有并行/Send 扇出时就不一样了。

超限抛 `GraphRecursionError`，消息是「Recursion limit of N reached without hitting a stop condition」。修法：显式传 `config={"recursion_limit": N}`，或者更好——**检查你的循环退出条件是不是永远不满足**。

↗ `graph.py` L425-426

---

### Q10 · 三个分支都指向 `aggregator`，`aggregator` 会被执行几次？如果它每次都往 `operator.add` 字段加 1，最终值是几？`L2`

**考点**：fan-in 去重

**参考答案**
**只执行一次**（实测 log 为 `['a','b','c','agg']`，`agg` 只出现一次），最终值也只加 1。

机制：路由阶段收集到的下一批任务会按 `(节点名, 参数)` **去重**，同一节点在同一超步只调度一次。这是 fan-in（汇合）语义的默认行为。

推论：`aggregator` 里看到的状态已经是三个分支合并后的结果——这正是它叫「汇合节点」的原因。

↗ `graph.py` L487 `_dedupe_tasks`、L552

---

### Q11 · 想「动态决定扇出几个分支」，`add_conditional_edges` 的 path 函数怎么返回？`L2`

**考点**：`Send` 的用法与限制

**参考答案**
返回 `list[Send]`：`return [Send("worker", {"item": i}) for i in items]`。每个 `Send` 携带一份**独立入参**，构造出 N 个并行任务。

三个必须记住的限制：

1. `Send`** 只能从条件边（或 **`Command(goto=[Send...])`**）返回**，节点直接 `return [Send(...)]` 在真实 LangGraph 里会报 `InvalidUpdateError: Expected dict, got [Send(...)]`；
2. `Send(node, arg)` 的 `arg` **就是这个节点的完整入参**——它拿不到主状态的其他字段（实测 worker 只看到 `{'item': 'x'}`，看不到 `secret`）。要传上下文就得显式塞进 `arg`；
3. 扇出的结果靠 reducer 汇流，所以被调的节点写的字段**必须**有 reducer。

↗ `graph.py` L159-163、L604-606（minigraph 允许节点直接返回 `list[Send]`，属于放宽）

---

### Q12 · 同步 `invoke()` 下，两个 `Send` 分支是真的并发还是伪并发？`L3`

**考点**：并发模型（sync vs async）

**参考答案**
**真实 LangGraph 里，同步 **`invoke()`** 也是真并发**，且用的是**线程池**。两条实测证据：

- 3 个各 `sleep(0.1)` 的 worker，`invoke` 总耗时 0.105s、`ainvoke` 0.106s——串行至少 0.3s；
- 在 worker 里打印 `threading.get_ident()`，3 个任务落在 **3 个不同线程**上。

但要注意两件事：

1. **并发粒度只到「单个超步内」**。跨超步仍然是严格串行的 BSP 屏障，不存在「上一步还没完就开始下一步」。
2. **同步并发只在 I/O 或 **`sleep`** 上有效**。CPU 密集任务受 GIL 限制，线程池帮不上忙；纯 CPU 场景要么用 `ainvoke` + 多进程，要么老实串行。

面试加分点：能说清「BSP 屏障」和「步内并发」是两层不同的东西，很多人会混为一谈。

↗ `graph.py` L526-549 `_execute_batch`（minigraph 用 `ThreadPoolExecutor` 显式实现）

---

## 第三组：路由与控制流（Q13–Q16）

### Q13 · 条件边、`Command(goto=...)`、`Send` 三者分别适合什么场景？`L2`

**考点**：路由原语的选型

**参考答案**

| 原语 | 适合 | 关键差异 |
| --- | --- | --- |
| `add_conditional_edges` | 路由逻辑**独立、可复用**，多种入边共用同一个分派器 | 写在外层，节点返回普通 dict；支持 `path_map` 把 key 翻译成节点名 |
| `Command(goto=, update=)` | 路由决策**和节点逻辑强耦合**（如 supervisor 决定下一步交给谁） | 一个节点可以 **goto 多个目标**（list），可以跨层级跳转（`Command.PARENT`） |
| `Send(node, arg)` | **动态扇出**，数量在运行时才知道 | 每个任务有独立入参，构造 map-reduce |

选择口诀：**分支固定 → 条件边；决策在节点里 → Command；数量不定 → Send**。

两个实测出来的细节：

- `Command` 返回时**不走**静态边。即使你给这个节点 `add_edge("r","a")`，只要它返回 `Command(goto="a")`，也只按 goto 走一次——`Command` 是**完全接管**路由，不是「在静态边基础上追加」。
- `Command` 与普通 dict 返回值**不能同时给**：节点返回 `Command` 时，写状态必须走 `update=` 字段；在 `Command` 外面再包一层 dict 是无效的（实测那次 `update` 直接丢失，只有 goto 生效）。

↗ `graph.py` L149-163、L592-631 `_normalize`

---

### Q14 · `Command` 相比「条件边 + 返回 dict」，多解决了什么问题？`L3`

**考点**：为什么要有 Command 这个原语

**参考答案**
三个条件边做不到的：

1. **跨层级跳转**：子图里的节点可以 `Command(goto="other", graph=Command.PARENT)` 直接把控制权交回父图的某个节点，条件边只能在本图内路由；
2. **一手交钱一手交货**：路由决策和状态更新原子地产生——「我决定去 A，并且顺手把 `next_agent` 设为 A」，条件边需要在节点里写状态、再在外层读状态，两处逻辑分裂；
3. **多目标并发**：`goto=["a", "b"]` 一步开出两个并行分支；条件边也能返回多个 key，但语义表达上不如 Command 直白。

↗ minigraph 的 `Command` 只实现了 1、3（L595-602），跨层级是原版能力

---

### Q15 · 图跑完第一次 `invoke` 之后，用**同一个 **`thread_id` 再 `invoke({"n": 100}, cfg)`，会发生什么？`L3`

**考点**：thread 语义 + reducer 的叠加

**参考答案**
如果 `n` 是 `Annotated[int, operator.add]`：**不是覆盖，是累加**。

实测：run1 结束 `n=11`（1+10），run2 传入 `{"v":"start2","n":100}`，最终 `n=122`。因为带 checkpointer 时，同 thread 的新输入会与已有状态**按 reducer 合并**，而不是重置。

这条规则直接决定了两个常见 bug：

- 「为什么我的计数器越来越大」——没换 thread\_id；
- 「为什么攒了一堆历史消息」——`messages` 是 `add_messages`，新会话没开新 thread。

正确做法：新会话用新 `thread_id`；确实要在老 thread 上重置，就显式覆盖（对 last-write-wins 字段没问题，对 add 字段得单独处理）。

↗ `graph.py` L400-409（minigraph 这里是覆盖而非合并，属于简化，已实测确认与真实版不同）

---

### Q16 · 无条件边（`add_edge`）和条件边同时从一个节点出发，会怎样？条件边返回了一个 `path_map` 里不存在的 key 呢？`L2`

**考点**：路由的合并规则 + 一个非常容易静默出错的边界

**参考答案**
先说结论：**两者取并集**。实测一个节点同时 `add_edge("r", "a")` 和 `add_conditional_edges("r", ... -> "b")`，两个分支都跑了，`log` 为 `['r','a','b']`。

再说那个更值钱的坑——**条件边返回值不在 **`path_map`** 里时有三种不同后果**：

| 情况 | 实测行为 |
| --- | --- |
| 没传 `path_map`，返回一个不存在的节点名 | **不报错，静默丢弃**，只打印一行警告 `wrote to unknown channel branch:to:xxx, ignoring it`，图直接结束 |
| 传了 `path_map`，返回的 key 不在 map 里 | **抛 **`KeyError`，直接炸 |
| 传了 `path_map`，key 在 map 里但映射到 `END` | 正常结束 |

最阴险的是第一种：**不抛异常、不中断，只是少跑了一条分支**，表现为「结果缺一块」而不是「崩了」。所以线上代码必须显式传 `path_map`，并在 path 函数里对所有分支兜底（用 `if/elif/else` 收口，不要用裸 `dict.get` 返回未校验的字符串）。

↗ `graph.py` L613-631（minigraph 把未知 key 当节点名用，行为更接近第一种但会炸得更早）

---

## 第四组：持久化（Q17–Q21）

### Q17 · Checkpoint 在什么时刻落盘？是「每个节点执行完」还是「每个超步执行完」？`L2`

**考点**：检查点粒度

**参考答案**
**每个超步执行完落一次**，不是每个节点。

看 `get_state_history` 的 `metadata["step"]` 就能确认：一次 `a → b` 的线性运行，历史是 `step=-1`（输入）、`step=0`（`next=('a',)`）、`step=1`（`next=('b',)`）、`step=2`（`next=()`）。四个检查点对应「输入 + 三个超步边界」。

这个粒度不是随意选的：因为写入合并是超步级的，只有在屏障处状态才是自洽的——中途落盘会拿到「半合并」的状态。

↗ `graph.py` L497-501

---

### Q18 · `thread_id` 和 `checkpoint_id` 的区别是什么？「时间旅行」靠哪个？`L2`

**考点**：两个 ID 的层级关系

**参考答案**

- `thread_id`：一条**会话主线**，所有历史检查点都挂在它下面；
- `checkpoint_id`：这条线上**某一个时刻的快照**，由 `thread_id + checkpoint_id` 精确定位。

时间旅行靠 `checkpoint_id`：从 `get_state_history` 拿到某个旧快照的 `config`，把它作为 config 传回去 `invoke(None, old_cp.config)`，或者 `update_state(old_cp.config, {...})` 改写后继续。

⚠️ 实测一个反直觉现象：从中间检查点 `invoke(None, cp.config)` **不会**从那里分叉重跑完剩余部分，而是直接返回那个检查点当时的 `values`（因为 `next=()` 表示已结束）。真正要「从某步改道重跑」，得用 `update_state` 把 `next` 重新指向待执行节点。

↗ `graph.py` L332-366

---

### Q19 · `update_state(config, values, as_node=...)` 里 `as_node` 到底决定什么？`L3`

**考点**：人工干预的语义

**参考答案**
`as_node` 决定「**假装这条更新是哪个节点写的**」，它同时决定两件事：

1. **reducer 用谁的视角**——更新会按该字段的 reducer 合并（所以 `messages` 是追加、`count` 是累加，不会直接覆盖）；
2. **接下来从哪继续**——框架按 `as_node` 在该图中的出边推算 `next`。如果 `as_node` 是终点节点，`next` 就是空，图直接结束；如果是中间节点，就顺着它的出边继续。

所以「人工修正后继续跑」的写法是：`as_node` 设为你**希望它表现为刚执行完的那个节点**。不传则沿用当前 `next`。

实测：从 step=1 的状态 `update_state(..., as_node="b")`，因为 b 是终点，`next` 变成 `()`，图直接结束；想继续得选一个还没跑完的节点。

↗ `graph.py` L348-366

---

### Q20 · `InMemorySaver` 什么时候能上线？生产该换什么？`L2`

**考点**：checkpointer 的选型

**参考答案**
`InMemorySaver` 只适合**单进程、开发调试、测试**。它有三个硬伤：

1. 进程重启全丢——HITL 场景下用户审批要等几小时，进程早没了；
2. 多副本部署时，用户请求打到另一个 pod 就读不到状态；
3. 无 TTL、无容量上限，长会话内存会一直涨。

生产选择：**PostgresSaver**（最常用，支持 `checkpoint_ns`、按 thread 索引）、**RedisSaver**（低延迟、可设 TTL）、SQLite 用于单机持久化。选型时重点看三件事：**跨进程共享、写放大（每个超步一次写）、历史清理策略**。

↗ `graph.py` L212-240 `InMemorySaver`

---

### Q21 · 一张「无限循环」的图在什么情况下会把 checkpointer 撑爆？怎么防？`L3`

**考点**：写放大

**参考答案**
每个超步都落一个完整状态快照。如果状态里有 `messages`（几百条）、或者 `values` 里塞了大对象（原始文档、图片 base64），一个跑 50 步的 agent 会话会产生 **50 份完整快照**，每份都包含全部历史消息——存储量是 O(步数 × 状态大小)。

防御手段（按性价比排序）：

1. **别把大对象放进 state**——存引用（URL、ID），用时再取；
2. **缩短 thread 生命周期**，一个任务一个 thread，而不是一个用户一个 thread；
3. **定期裁剪历史**：Postgres 里按 `created_at` 清理旧 checkpoint，或只保留最近 N 个 + 关键节点；
4. 用支持增量/压缩的 saver。

↗ 对应 `_run` 里的 `saver.put(...)`（L497）

---

## 第五组：Human-in-the-loop 与中断（Q22–Q27）

### Q22 · `interrupt()` 是怎么「暂停」的？为什么它必须在一个节点**内部**调用？`L1`

**考点**：中断的实现原理

**参考答案**
它不是协程 `yield`，也不是线程挂起，而是**抛异常**：`raise GraphInterrupt(value)`（原版是 `GraphBubbleUp` 家族）。

异常从节点栈一路抛到执行器，执行器捕获后：把 `value` 存进检查点、把本超步任务列表存进 `tasks`、结束本次 `invoke`。所以「暂停」的本质是**这一次运行结束了**——进程可以退出，几小时后另起一个进程 `invoke(Command(resume=...))` 接着跑。

必须在节点内调用，是因为它依赖执行器注入的**上下文**（当前任务的 id / 命名空间）。在节点外调用拿不到这个上下文。

↗ `graph.py` L166-184

---

### Q23 · 恢复时，节点是「从 interrupt 那行接着跑」还是「从头重跑」？这意味着什么？`L2`

**参考答案**
**从头重跑**。`interrupt()` 第二次执行时不再是抛异常，而是**返回**你传入的 resume 值，于是节点继续往下走。

这带来一条铁律：`interrupt()`** 之前的代码必须幂等**。如果你在它前面调了一次付费 API、发了一封邮件、写了一条数据库记录——恢复时会再来一遍。

正确姿势：

```python
def node(state):
    # ❌ 副作用在 interrupt 之前
    result = expensive_api_call()      # 恢复时会重复调用
    ok = interrupt({"preview": result})
    ...
```

要么把副作用挪到 interrupt 之后，要么用 `idempotency_key` 之类的手段去重。

↗ `graph.py` L176-183

---

### Q24 · 同一个节点里调两次 `interrupt()`，第二次恢复时会不会拿错值？框架靠什么区分？`L3`

**考点**：中断的「暗号」机制

**参考答案**
不会拿错，但**它比你想的更微妙**。

实测（同一个节点里两个 `interrupt`）：

- 第一次运行挂起在第一个，`state.next = ('gate',)`；
- `invoke(Command(resume="A"))` 后**又挂起在第二个**，`state.next` **仍然是** `('gate',)`；
- 再 `invoke(Command(resume="B"))` 才完成。

关键点：LangGraph 用的是**按任务命名的 resume 通道 + 节点内执行序号**，而不是「节点名」这种粗粒度 key。如果只用节点名当 key，第二次挂起就会取到第一次的值——这是自己实现 HITL 时最典型的 bug。

minigraph 里把这个「暗号」显式写了出来：`slot = f"{节点名}@{步号}#{批内序号}"`，且恢复时要**回落到挂起所在的 step**，槽位才能对上。

↗ `graph.py` L436-439、L415-419

---

### Q25 · 一个超步里有 A、B 两个并行节点，B 挂起了。A 的写入在恢复后还在吗？A 会重跑吗？`L3`

**考点**：中断与并行的交互——这是本套题最难的一题

**参考答案**
**LangGraph 的真实行为（实测）**：A 的写入**保留**了（`get_state` 能看到 `log` 里已有 `'fast'`），并且 A **不会重跑**（调用计数保持 1）。B 单独重跑。

具体表现（`fast` 与 `slow` 同批，`slow` 挂起）：

- 挂起后 `state.values = {'log': ['fast']}`，`state.next = ('slow',)`；
- 恢复后最终 `log = ['fast', 'slow:ok']`，`fast` 只执行过 1 次。

所以真实实现是\*\*「已完成的兄弟任务的结果被写进检查点，恢复时只为未完成的任务重建调度」\*\*，而不是把整批都作废。

⚠️ 但有个魔鬼细节：同一节点如果**既是挂起者、又要重跑**，它读到的状态可能已经包含了兄弟的写入（实测某构造下 `slow` 三次进入、`fast` 看到的还是旧值 `v=0`）。这正说明「快照隔离只在一个超步内成立，跨中断点不保证」。

minigraph 选择了更保守的**整步作废**（L455-459、L471-484）：本超步写入全丢，恢复后整批重跑，因此**节点必须幂等**。两种模型都能自洽，但你在面试里能说出「我实现的是保守模型，LangGraph 用的是更精细的 per-task 记录」，是明确的加分项。

↗ `graph.py` L452-484

---

### Q26 · 中断之后，怎么知道「现在卡在哪个节点、在等什么」？`L2`

**参考答案**
三个入口：

1. `invoke`** 的返回值**里带 `__interrupt__` 键，值是 `[Interrupt(value=..., id=...)]`；
2. `get_state(config)`：`state.next` 给出待执行的节点元组，`state.tasks[i].interrupts` 给出该任务的中断详情；
3. `state.interrupts`（快照上的聚合视图）。

注意 `state.next` 在「同一节点多次挂起」时**不会前进**，所以你**不能**用 `next` 判断「这是第几次询问」——要用 `Interrupt.id` 或自己在状态里维护轮次。

↗ `graph.py` L324-327 `interrupted` 属性

---

### Q27 · 用户审批「打回重写」，用 `interrupt` 怎么实现？要不要回到起点？`L3`

**参考答案**
不需要回到起点，用 `Command(goto=...)` 在恢复值里**改道**即可：

```python
def review(state):
    decision = interrupt({"draft": state["draft"], "ask": "通过/打回"})
    if decision["action"] == "revise":
        return Command(goto="write", update={"feedback": decision["note"]})
    return Command(goto="publish")
```

要点：

- 恢复值可以是**结构化的 dict**，不只是字符串——审批意见、修改建议都能带回来；
- 用 `Command(goto=...)` 回到 `write` 节点，比在条件边里写 `if` 更内聚；
- 回到 `write` 时它读到的是**已提交的状态**（含上次草稿 + feedback），所以「基于上一版改写」天然可行；
- ⚠️ 别忘了 `recursion_limit`——用户反复打回会一直加步数。

↗ `graph.py` L595-602（Command 的 goto 解析）

---

## 第六组：工程与设计取舍（Q28–Q30）

### Q28 · 要在节点里调一个「偶发超时的 LLM 接口」，`retry_policy` 配了 `max_attempts=3`，为什么有时候根本不重试？`L3`

**考点**：retry 的触发条件

**参考答案**
`RetryPolicy` 默认只对**部分异常**重试，很多异常（比如普通的 `ValueError`、部分 `httpx` 异常）不在默认名单里，会直接往上抛。必须显式指定：

```python
RetryPolicy(max_attempts=3, retry_on=ValueError)          # 单一类型
RetryPolicy(max_attempts=3, retry_on=(TimeoutError, ...))  # 元组
```

实测：`max_attempts=5`、不写 `retry_on`，抛 `ValueError` 时**一次就失败**；加上 `retry_on=ValueError` 后重试到第 4 次才成功。

另外两个坑：

- **重试不等于幂等**——重试会重新执行整个节点，节点里的副作用会重复；
- **超时 + 重试的总时长**要算进你的 SLA，`max_attempts=3` × 30s 超时 = 最坏 90s。

↗ `graph.py` L677-693 `_run_with_retry`

---

### Q29 · 一个 ReAct agent 图（agent ⇄ tools 循环），工具调用结果怎么放进状态才不会「上一轮的工具结果被覆盖」？`L2`

**参考答案**
用 `add_messages` 语义的消息列表，而不是 `Annotated[list, operator.add]` 的裸列表——**关键在于按 **`tool_call_id`** 配对**。

两个必须配对的 id：

1. `AIMessage.tool_calls[i].id` ↔ `ToolMessage.tool_call_id`——LLM 侧靠它把结果和请求对上；
2. 消息 `id`——靠它做「同 id 替换」（HITL 里编辑某条消息）。

如果只 `operator.add` 追加、id 对不上，模型会看到「一个没有结果的 tool\_call」，很多厂商的 API 会直接 400。

↗ `graph.py` L63-86

---

### Q30 · 如果让你从零实现 LangGraph，最难的三处是什么？为什么？`L3`

**考点**：这题是「你到底是背概念还是真读过源码」的分水岭

**参考答案**（按难度排序，能给具体现象最好）

1. **中断的恢复语义**。难点不在「抛异常暂停」（这行代码就够），而在「恢复时精确对上号」：节点要重跑、槽位要唯一、`step` 要回落到挂起那一步、resume 值要跨线程传递。任一环节错，表现都是「取到了上一次的旧值」这种**静默错误**，不报错、只出错数据。
2. **并行 + 写入合并的原子性**。同一超步里的快照隔离、reducer 的连续折叠、无 reducer 时的并发写检测、以及「挂起时半批写入怎么处理」——四个决策互相牵扯。选错了会得到「不丢但重复」或「不重复但丢」两种 bug 之一。
3. **线程本地上下文的传播**。`threading.local` 不会被 `ThreadPoolExecutor` 继承，所以 slot / resume 必须**显式**塞进工作线程；异步版本里换成 `contextvars`，规则又不一样。这是典型的「逻辑写对了但跑起来不对」的地方。

加分回答：能说出「这三个难点共同指向一件事——**执行器必须为每个任务维护一份独立且可重建的上下文**」，说明你真正抽象出了设计本质。

↗ `graph.py` L513-517、L526-549

---

## 附录：五处实测差异——面试里最能加分的细节

下面这几条是我用真实 LangGraph 1.2.16 实测出来的，**与 minigraph 的行为不同**。面试时能主动说出差异，比背一百个概念都管用。

| 场景 | minigraph | 真实 LangGraph | 说明 |
| --- | --- | --- | --- |
| 同 thread 第二次 `invoke` 带输入 | **覆盖**为新输入 | 与现有状态**按 reducer 合并**（`n` 从 11 → 122） | real 版把「新输入」也当成一次 update 处理 |
| 节点直接 `return [Send(...)]` | 允许 | **报错**，必须从条件边返回 | Send 是路由原语，不是状态更新 |
| 并行批中一个节点挂起 | **整步作废**，全批重跑 | **已完成兄弟保留**，只重跑未完成的 | real 版按 task 粒度记录 |
| 写入未声明字段 | `ValueError` | **静默忽略** | TypedDict 不做运行时校验 |
| 孤立节点（无边） | `compile()` 报错 | 编译通过，静默不执行 | minigraph 更严格 |

> 结论不是「谁对谁错」，而是：**读源码的意义在于知道每个设计决策的备选方案**。你能说清「我选保守模型是因为它让幂等假设变简单」，这段对话就已经赢了。
