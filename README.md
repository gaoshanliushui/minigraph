# README

# mini-langgraph：用 600 行看懂 LangGraph 的内核

> 读源码前先读这份。它把 LangGraph 拆成 5 个可以独立理解的概念，
> 每个概念都对应 `minigraph/graph.py` 里的一段实现。

---

## 一句话结论

**LangGraph 不是"工作流引擎"，而是一个带 reducer 的状态机跑批器。**

节点是无状态函数，状态只有一份，每个超步（superstep）里：
**并发执行 → 收齐增量 → 用 reducer 合并 → 决定下一步**。
所谓 checkpoint、人工审批、时间旅行，全部是这个循环的副产品。

---

## 1. 状态与 reducer：LangGraph 最反直觉的设计

大多数"Agent 框架"让你直接改共享状态。LangGraph 不这么干：

```python
class S(State):
    messages: Annotated[list, add_messages]   # 追加，按 id 去重
    total:    Annotated[int, operator.add]    # 累加
    query:    str                             # 缺省：last-write-wins
```

节点返回的**不是新状态，是增量**：

```python
def node(state): return {"total": 1}   # 不是 {"total": state["total"] + 1}
```

为什么这么设计？因为**只有增量才能被并行合并**。如果节点读到 `total=5`
再写回 `total=6`，另一个并发节点也写入 `6`，最后结果就是 6 而不是 7——
这就是经典的 lost update。

> 所以 LangGraph 里，同一超步两个并行节点写同一个"无 reducer 字段"会**直接报错**。
> 这不是 bug，是它在强迫你表态：你要的到底是覆盖、累加，还是去重合并？
> （对应实现：`_merge_writes`）

**这是整个框架最重要的认知转变：从"共享内存"转向"增量 + 合并函数"。**

---

## 2. 超步调度：Pregel / BSP 模型

LangGraph 的执行模型抄自 Google 的 Pregel（图计算）和 BSP（Bulk Synchronous Parallel）。

```
超步 1:  [node_a] ──────┐
                        ├─→ 合并写入 → 重算状态
超步 2:  [node_b] [node_c] ─┐   ← 并行（线程池）
                        ├─→ 合并写入 → 重算状态
超步 3:  [node_d] ──────┘
```

三个必须记住的性质：

| 性质 | 含义 | 对应代码 |
| --- | --- | --- |
| **同步屏障** | 同一超步里的节点看到的是**同一份状态快照**，谁也别想读到别人的输出 | `snapshot = dict(values)` |
| **批量提交** | 所有写入在这一步末尾**一次性** reduce 进状态 | `_merge_writes` |
| **顺序无关** | reducer 决定合并语义，所以并发执行不会影响结果 | 用 `operator.add` 而非 `+` |

**循环 = 回边。** ReAct Agent 的 `tools → agent` 就是一条回边，不需要
`while not done` 这种宿主代码——循环次数由 `recursion_limit` 兜底。

---

## 3. 路由：三种方式，别混用

### 3.1 静态边 `add_edge("a", "b")`

最简单，`a` 跑完就跑 `b`。

### 3.2 条件边 `add_conditional_edges`

路由函数**只读状态、不做副作用**，返回 key：

```python
g.add_conditional_edges("agent", should_continue,
                        {"tools": "tools", END: END})
```

### 3.3 `Command`：节点自己决定去哪

当"下一步"依赖节点**内部**的复杂判断时，用 `Command` 把
"更新状态"和"路由"合并成一个返回值：

```python
def node(state):
    return Command(goto="retry" if err else "done", update={"attempts": 1})
```

> **实践建议**：能用条件边就别用 `Command`。`Command` 的问题是它让图结构
> 变得不可静态分析——你看 `add_conditional_edges` 一眼就知道有哪些分支，
> 看 `Command` 得去翻每个节点的函数体。

---

## 4. 扇出（Send）：动态并行

条件边只能走"预定义的分支"，但 map-reduce 需要**运行时才知道有几个任务**：

```python
def dispatch(state):
    return [Send("worker", {"topic": t}) for t in state["topics"]]
```

`Send` 的本质：**在超步末尾动态生成 N 个任务**，下一超步并发执行。
每个 `Send` 携带独立 `arg`，但写回的是同一个状态——所以 worker 的产出
必须落在带 reducer 的字段上（通常是 `Annotated[list, operator.add]`）。

> 这是 LangGraph 做 subagent / 并行研究 / 批量评测的标准姿势。
> 坑：`Send` 的任务不会自动"汇合"回一个节点，需要显式加一条
> `add_edge("worker", "aggregator")`；多条分支指向同一节点时框架会去重
> （对应 `_dedupe_tasks`），这就是 join 语义。

---

## 5. 检查点与 interrupt：断点续跑的全部秘密

`InMemorySaver` 存的东西极其朴素——**每个超步一个快照**：

```python
@dataclass
class Checkpoint:
    step: int
    values: dict          # 这一步结束时的完整状态
    tasks: list           # 下一步要跑谁（含 Send 的 arg）
    interrupt: dict|None  # 当前是否悬在某个中断点上
```

有了这个，三件事就免费了：

### 5.1 人工审批

```python
def review(state):
    decision = interrupt({"draft": state["draft"]})   # 冻结
    return {"status": decision}
```

恢复：`app.invoke(None, {"thread_id": tid, "resume": {"action": "approve"}})`

**关键细节（也是自己实现最容易踩的坑）**：恢复时从"挂起点"重跑该节点，
而 `interrupt()` 第二次执行必须能取到 resume 值。所以需要一个**稳定且唯一**
的槽位标识。mini 版用 `节点名@步号#批内序号`；如果你用 `节点名` 当 key，
节点在循环里第二次挂起就会取到上一个值（这个 bug 我在实现时真踩了，
见 `test_interrupt_can_fire_multiple_times_in_a_loop`）。

### 5.2 时间旅行

`get_state_history()` 拿到所有快照 → `update_state(..., as_node=...)` 从任意点
改写并分叉重跑。这不是什么高级特性，就是"多存了几份 dict"。

### 5.3 崩溃恢复

进程挂了，checkpoint 还在（生产上用 Postgres/SQLite saver），
下次 `invoke(None, thread_id)` 直接接着跑。

> 注意：LangGraph 的 checkpoint 是**超步级**的，不是**节点级**的。
> 一个超步里跑了 3 个节点，崩了要重跑这 3 个。所以节点要做到幂等。

### 5.4 一个容易忽略的语义：挂起 = 整步回滚

如果同一个超步里既有普通节点、又有节点调用了 `interrupt()`，那么**这一步的
****所有写入都会被丢弃**，恢复后整批重跑：

```
超步 3:  [fetch_data ✓]  [human_review → interrupt]
          ↓ 本步作废
checkpoint: values 不含 fetch_data 的写入, tasks = [fetch_data, human_review]
恢复后:  两个节点都重新执行
```

这不是实现偷懒，而是 BSP 屏障的必然结果——"一步要么全提交，要么全不提交"。
它直接推导出 LangGraph 反复强调的那条铁律：**节点必须幂等**。

（我实现时先后写错了三次才想明白：最初只重跑挂起节点 → 后面的任务丢失；
改成全批重跑但保留已完成写入 → `operator.add` 字段被重复累加。正确解法是
"挂起时整步不提交"，见 `test_interrupt_inside_parallel_batch_rolls_back_the_superstep`。）

---

## 对照速查表

| 概念 | LangGraph | minigraph | 文件位置 |
| --- | --- | --- | --- |
| 状态声明 | `Annotation.Root` | `class S(State)` + `Annotated` | graph.py §1 |
| 图构建 | `StateGraph(S)` | 同名 | graph.py §4 |
| 条件边 | `add_conditional_edges` | 同名 | graph.py §4 |
| 动态路由 | `Command(goto=...)` | 同名 | graph.py §2 |
| 并行扇出 | `Send(node, arg)` | 同名 | graph.py §2 |
| 人工中断 | `interrupt()` | 同名 | graph.py §2 |
| 持久化 | `InMemorySaver` / `SqliteSaver` | `InMemorySaver` | graph.py §3 |
| 执行内核 | Pregel loop | `CompiledGraph._run` | graph.py §5 |
| 流式输出 | `stream(stream_mode=...)` | `stream()` → `(node, step, update)` | graph.py §5 |

---

## 被砍掉的部分（以及为什么可以砍）

| 特性 | 为什么 mini 版不做 |
| --- | --- |
| `stream_mode="messages"` 等 | LLM token 级流式，纯 I/O 细节，不影响理解内核 |
| `Subgraph` / 子图 | 本质是把一个 `CompiledGraph` 当节点，机制完全复用 |
| `checkpoint_ns` | 为子图服务；单层图不需要 |
| `PostgresSaver` | 只是把 `dict` 换成 SQL，`put/get/list` 接口不变 |
| `durability` / `cache` | 性能优化，不改变语义 |
| 异步 (`ainvoke`) | `asyncio` 版的事件循环，结构完全对称 |
| `RetryPolicy` 全参数 | 保留了核心的重试 + 指数退避 |

---

## 推荐的阅读顺序

1. **先跑起来**：`python examples/01_react_agent.py`
看 `agent → tools → agent → END` 这 3 个超步怎么走。
2. **看内核**：读 `CompiledGraph._run`（约 90 行）。
重点看 `snapshot` 的创建位置和 `_merge_writes` 的调用位置。
3. **看并行**：读 `_execute_batch` + `_dedupe_tasks` + `_merge_writes`。
理解"并行写冲突为什么报错"。
4. **看中断**：跑 `examples/02_hitl_and_fanout.py`，然后读 `interrupt()`
的实现（只有 10 行）和挂起检查点的构造。
5. **最后看构建 API**：`StateGraph` 那段几乎全是参数校验，扫一眼即可。

---

## 一个判断标准

如果你能用一句话说清下面这个问题，就说明你懂了：

> 为什么 LangGraph 的节点必须返回 `dict` 而不能直接 `state["x"] = 1`？

答案：因为前者是**可合并的增量**，后者是**不可合并的副作用**。
在 BSP 模型下，副作用会让"并行执行"和"断点重跑"同时失效。
这就是 LangGraph 相对于 LangChain 的 `AgentExecutor` 的核心差异——
后者把循环写在 Python 的 `while` 里，状态藏在对象里，所以没法暂停、没法回放。
