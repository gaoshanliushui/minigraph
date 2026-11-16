# mini-langgraph：用一张通道网络看懂 LangGraph 的内核

> 读源码前先读这份。它把 LangGraph 拆成 6 个可以独立理解的概念，
> 每个概念都对应 `minigraph/` 包内的一段实现，并有真实 LangGraph 差分测试兜底。

---

## 一句话结论

**LangGraph 不是“按边查下一跳”的工作流引擎，而是一个由 channel 版本驱动的 Pregel 状态机。**

- **边不是数据结构，是一次 channel 写入**：`add_edge("a", "b")` 编译成
  `a` 写 `branch:to:b`，`b` 订阅这个通道。
- **触发不是看图，是看通道版本**：通道版本超过节点上次看到的水位线，节点才被唤醒。
- 每个超步只做：**找出变化通道的订阅者 → 并发执行 → 收齐写入 → 合并并推进版本**。

checkpoint、人工审批、时间旅行，都是这套循环加上持久化后的副产品。

---

## 1. 编译期：图如何变成通道网络

`StateGraph.compile()` 不再把边列表交给调度器，而是调用 `builder._compile()`，
产出通道表、节点规格，以及 `channel → subscribers` 反向索引。

```text
add_edge("a", "b")
    a.writers  += ["branch:to:b"]
    b.triggers += ["branch:to:b"]

add_edge(["a", "b"], "join")
    channels["join:a+b:join"] = NamedBarrierValue({"a", "b"})
    a.writers += ["join:a+b:join"]     # 写入自己的名字："a 到了"
    b.writers += ["join:a+b:join"]
    join.triggers += ["join:a+b:join"]
```

通道分两类，不能混为一谈：

| 类型 | 例子 | 作用 |
| --- | --- | --- |
| **数据通道** | `messages`、`total` | 保存状态字段，供节点读取；不是触发源 |
| **触发通道** | `branch:to:b`、`join:a+b:c` | 承载“该醒了”的信号；值通常不重要，版本才重要 |

`CompiledGraph.trigger_to_nodes` 是反向索引。每轮只需检查上一轮变化通道的订阅者，
不必扫描所有节点。对应实现：`minigraph/builder.py`、`minigraph/channels.py`。

---

## 2. 触发：版本号与水位线

运行时每节点都有一份水位线：

```python
channel_versions["branch:to:b"] = 7
versions_seen["b"]["branch:to:b"] = 5
```

节点 `b` 该执行，当且仅当它订阅的某个通道满足：

```python
channels[chan].is_available() and channel_versions[chan] > versions_seen[node][chan]
```

超步的核心结构是：

```python
tasks = _prepare_next_tasks(...)       # changed channels -> subscribers -> tasks
snapshot = _read_state(...)            # 全批节点看到同一份状态
results = _execute_batch(...)          # 单步内并发

# 先记录节点执行前看到的版本，再应用本步写入
versions_seen[node][chan] = channel_versions[chan]
updated_channels = _apply_writes(...)  # reducer 合并 + 推进版本
```

**“先记水位、再应用写入”是回边能转起来的关键。**如果顺序反过来，
节点写 `branch:to:自己` 后又把水位线推到新版本，自己的触发信号就被自己抹掉了，
循环会停在一轮。

分支通道使用 `EphemeralValue`：它只活一个超步，下一步没被重写就通过
`update([])` 自清。`Send` 则写入 `__pregel_tasks`（`Topic`），走 PUSH 任务轨；
PUSH 不会推进目标节点 `branch:to:X` 的水位线。

---

## 3. 状态与 reducer：增量为什么可以并行合并

```python
class S(State):
    messages: Annotated[list, add_messages]   # 按 id 追加/覆盖
    total:    Annotated[int, operator.add]    # 累加
    query:    str                             # last-write-wins
```

节点返回的不是新状态，而是增量：

```python
def node(state):
    return {"total": 1}      # 不是 {"total": state["total"] + 1}
```

如果节点读到 `total=5` 再写回 `total=6`，另一个并发节点也写回 `6`，
最后就是 6 而不是 7——经典 lost update。返回增量后，框架才能在超步末尾统一合并。

字段到数据通道的映射很直接：

| 字段声明 | 通道 | 并发写入语义 |
| --- | --- | --- |
| 无 reducer / `LAST_VALUE` | `LastValue` | 同一步多个值直接报错 |
| 有二元 reducer | `BinaryOperatorAggregate` | 逐个 fold：`value = reducer(value, update)` |

`add_messages` 和 `merge_dict` 也是二元 reducer，所以不需要特殊通道。
并发写冲突由 `LastValue.update()` 抛出 `ConcurrentWriteError`，
对应真实 LangGraph 的 `InvalidUpdateError`。

---

## 4. 路由与汇聚：三个容易混淆的语义

### 4.1 静态边与条件边

```python
g.add_edge("a", "b")
g.add_conditional_edges("agent", should_continue,
                        {"tools": "tools", END: END})
```

条件边看到的是“本节点写入**经过 reducer 合并后**的状态”，不是原始增量。
例如节点写 `{"n": 1}` 且 reducer 是 `operator.add`，路由函数看到的是 `old + 1`。

### 4.2 `Command(goto=...)`

```python
def node(state):
    return Command(goto="retry" if err else "done", update={"attempts": 1})
```

`goto` 编译为一次 `branch:to:<target>` 写入。**它不会抑制节点原有的静态边和条件边**；
若同时存在，静态目标和 `goto` 目标都会被触发。

### 4.3 两条入边不等于屏障

这是最容易凭直觉写错的地方：

```python
g.add_edge("a", "c")
g.add_edge("b", "c")
```

- `a`、`b` 在**同一超步**写入 `branch:to:c`：版本只推进一步，`c` 跑一次。
- `a`、`b` 在**不同超步**写入：`c` 会被触发两次。

要表达“等 a 和 b 都完成，再跑一次 c”，必须显式使用屏障边：

```python
g.add_edge(["a", "b"], "c")     # NamedBarrierValue；c 消费后屏障重新武装
```

---

## 5. `Send`：动态 PUSH 任务

```python
def dispatch(state):
    return [Send("worker", {"topic": t}) for t in state["topics"]]
```

每个 `Send` 写入 `__pregel_tasks`，下一超步变成一个 PUSH 任务，同批并发执行。
worker 的产出应写入带 reducer 的字段，例如 `Annotated[list, operator.add]`。

多个 worker 同时写 `branch:to:aggregator` 时，若它们在同一超步完成，
aggregator 下一超步只触发一次。若完成时间落在不同超步，则可能触发多次；
需要严格“全部到齐后一次”的语义时，使用屏障边。

---

## 6. checkpoint 与 interrupt

通道模型下，一个 checkpoint 不只保存状态值，还保存恢复调度所需的版本信息：

```python
Checkpoint(
    step=...,
    values=...,                 # 用户可见的状态投影（不含 branch:to:*）
    channel_values=...,         # 每个通道的快照
    channel_versions=...,       # 通道版本
    versions_seen=...,          # 每节点水位线
    next=..., tasks=...,
    interrupt=...,
)
```

因此恢复时不需要重新“查边列表”：重建通道后，`_prepare_next_tasks`
会根据版本与水位线重新算出该执行的任务。

### 6.1 人工审批

```python
def review(state):
    decision = interrupt({"draft": state["draft"]})
    return {"status": decision}
```

恢复：`app.invoke(None, {"thread_id": tid, "resume": {"action": "approve"}})`。
resume 槽位包含节点名、超步号、PULL/PUSH 类型和组内序号，保证循环或扇出中
同一节点多次挂起时不会拿错值。

resume 值有三种形态：

```python
{"resume": "approve"}                        # 标量：答复下一个待恢复的中断
{"resume": {"action": "approve"}}            # 字典：整个字典原样交给 interrupt()
{"resume": {"review": "approve"}}            # 字典：键是节点名，一次答复多个
{"resume": {"review": ["v1", "v2"]}}         # 列表：一个节点里多次 interrupt 的值
```

第二、三行都是字典，怎么区分？**看键对不对得上待恢复的节点名**：
对得上（`review` 确实是个节点）就按“批量答复”解释；对不上，整个字典就是
`interrupt()` 拿到的那个值 —— 这与真实 LangGraph 一致（`Command(resume=X)`
里 `interrupt()` 拿到的就是 X 本身，不做节点名匹配）。

同一超步里多个节点同时挂起时，它们**全都**会被记进 checkpoint —— 只记一个的话，
另一个重跑时永远等不到自己的值，两个中断会互相顶死。

### 6.2 时间旅行

`get_state_history()` 取历史快照，`update_state(..., as_node=...)` 改写状态。

`as_node` 的含义是“**假装这条更新是哪个节点写的**”。既然是它写的，被点亮的就是
**它的出边**，而不是它自己：

```python
g.add_edge("a", "b"); g.add_edge("b", "c")

app.update_state(cfg, {"n": 100}, as_node="b")
# 接下来跑的是 b 的后继 c —— 不是 b
app.get_state(cfg).next      # -> ("c",)
```

`as_node` 指向终点节点时（出边是 `END`），图直接结束 —— `next` 为空。
不给 `as_node` 时按水位线推断“最后一个写过状态的节点”；几个节点同时写过、
说不清是谁，就报错请显式指定。

### 6.3 minigraph 的挂起策略：整步回滚

minigraph 中，同一超步只要有节点调用 `interrupt()`，本步所有写入都不提交，
恢复后整批重跑。节点因此必须幂等。

**这是 minigraph 有意的教学简化，不应误写成真实 LangGraph 的唯一实现。**
实测 LangGraph 1.2.16 会保留已完成兄弟任务的结果、只重跑未完成任务；
在标准测试场景中两者最终状态一致，但节点副作用次数不同。

### 6.4 落盘：SqliteSaver

`InMemorySaver` 只活在进程里。要让图"重启后还能接着跑"，换一个 saver 即可
（接口完全一致，两者可互换）：

```python
from minigraph.checkpoint_sqlite import SqliteSaver

app = g.compile(checkpointer=SqliteSaver("checkpoints.db"))
```

快照用带类型标签的 JSON 存进 SQLite（`MISSING` 哨兵、`Send` 对象、`set`、
tuple、bytes 都能原样还原），不用 pickle —— 反序列化不可信文件等于执行任意代码。
代价是自定义类型的 state 值存不进去，会明确抛 `TypeError` 而不是悄悄丢数据。

### 6.5 DeltaChannel：只存增量，不存全量

普通通道（`BinaryOperatorAggregate`）每份 checkpoint 都**完整存一遍**累积后的值。
对话历史越长，每份快照越大，于是总空间是 O(n²)：

| 50 轮对话 | checkpoint 总大小 | 平均每份 |
| --- | --- | --- |
| 普通 `add_messages` | 2753 KB | 18.8 KB |
| `DeltaChannel` | **141 KB** | **0.96 KB** |

`DeltaChannel`（对齐 LangGraph 的 `langgraph/channels/delta.py`）换了个存法：
**checkpoint 里不存它的值**，只存每一步的**增量写入**；状态靠
「最近的快照 + 之后的增量重放」重建。

```python
from minigraph.state import DeltaChannel, State, add_messages

class AgentState(State):
    messages: Annotated[list, DeltaChannel(add_messages, list)]
```

三个机制：

**① 值不进 checkpoint。** delta 通道既不出现在 `channel_values`，也不出现在
`values` —— 值由「种子 + 增量写入」表达。恢复时沿 thread 的 checkpoint 链
往回走，找到最近一个存过这个通道的祖先（**种子**），把之后的增量按旧→新
重放回去。

**② 两个计数器决定何时落完整快照。**

| 计数器 | 含义 | 阈值 |
| --- | --- | --- |
| `updates` | 这个通道被写了几次 | `snapshot_frequency`（默认 1000） |
| `supersteps` | 距上次快照过了几个超步 | `DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT`（默认 5000） |

满足任一条就落一份 `DeltaSnapshot`（通道的完整值），并把计数归零。
第二个上界很重要：一个**很久没被写**的通道，光靠"写入次数"算，重放链会
一直变长；这个上界保证重放深度有天花板。

**③ reducer 要满足"分结合并不变性"。**

```
reducer(reducer(state, xs), ys) == reducer(state, xs + ys)
```

因为重放时增量可能被合并成更大的批次。`add_messages`（按 id 追加/更新）、
`operator.add`、追加 list 的自定义 reducer 都满足。

minigraph 的 reducer 一律是二元的 `(old, new) -> merged`；LangGraph 的
DeltaChannel reducer 是批量的 `(state, list[writes]) -> new_state`。两种都支持，
用 `BatchReducer` 显式标记批量 reducer（靠 arity 判断不可靠）：

```python
from minigraph.channels import BatchReducer

def merge(state, writes):
    out = list(state)
    for w in writes:
        out.extend(w)
    return out

class S(State):
    items: Annotated[list, DeltaChannel(BatchReducer(merge), list)]
```

**④ 存储结构：写入日志独立成表，祖先链显式化。**

对齐 LangGraph 的 `checkpoints` + `writes` 两表结构：

| 表 | 装什么 |
| --- | --- |
| `checkpoints` | 内核状态 blob（`values` / `channel_values` / 版本 / 水位线）+ `parent_checkpoint_id` |
| `writes` | 每份 checkpoint 当步的**全部**写入（数据 + 触发通道），`(checkpoint_id, idx)` 保序 |

写入日志**不分数据通道和控制通道**（对齐 LangGraph 的 `put_writes`）：
`messages` 的增量、`branch:to:X = null` 的触发信号、输入写入、`__start__`
点亮入口，全都按应用顺序记录。delta 重放只按通道名挑它要的那几条；
控制写入不参与重放，但让这份日志成为完整的执行记录 —— 谁写了什么、
谁唤醒了谁，都在表里。

特殊通道是**事件**而不是状态（对齐 LangGraph 的 reserved write keys）：

| 通道 | 何时记 | 值 |
| --- | --- | --- |
| `__interrupt__` | 超步因 interrupt 挂起 | 本步所有挂起点的 payload 列表 |
| `__resume__` | resume 恢复后的第一份 checkpoint | 用户给的 resume 值 |
| `__no_writes__` | 超步跑了但什么都没写 | `None` |

`__interrupt__` / `__resume__` 在 writes 表里用**负 idx**（-3 / -4，对齐
`WRITES_IDX_MAP`），与普通写入（0 起）错开。这些事件通道不参与 delta
重放；挂起/恢复的机制仍由 Checkpoint 的 `interrupts` / `resumes` / `slots`
字段承载。注意 LangGraph 的 `NO_WRITES` 是**按任务**记的，还用于
"已完成任务不重跑"；minigraph 整步回滚用不上那个用途，只把它当日志。

`Checkpoint.pending_writes` 是个**查询视图**：写入时拆进 `writes` 表，
读出时拼回 —— Checkpoint 对使用者仍是自包含的。

`parent_checkpoint_id` 解决的是**分叉**：`update_state(..., checkpoint_id=...)`
或 `invoke(..., {"checkpoint_id": ...})` 从历史点分叉后，两条线的 checkpoint
在存储里交错追加，按写入顺序扫祖先会把**另一条线**的写入也重放进来
（主线消息串进分叉线）。沿父链走则只有 on-path 祖先参与 —— LangGraph
原话："walks the parent chain, not `list(before=...)`: for forked threads,
only on-path ancestors contribute"。

旧格式的库打开时自动迁移：补 parent 列（旧线性历史按 seq 回填）、
旧 blob 里的 pending_writes 在 writes 表查不到时回退读取，不用搬数据。

实现落在 `minigraph/channels.py`（通道）、`minigraph/runtime.py`（快照节奏与
重放）、`minigraph/checkpoint.py` 与 `checkpoint_sqlite.py`（祖先链遍历、
两表存储）。

---

## 与真实 LangGraph 的已知差异

| 行为 | minigraph | LangGraph 1.2.16 |
| --- | --- | --- |
| 节点直接返回 `list[Send]` | 支持，并抑制该节点的静态/条件边 | 抛 `InvalidUpdateError`；请从条件边或 `Command` 发出 |
| 节点返回 `{"field": None}` | 视作 no-op，不写通道 | 将 `None` 写入通道 |
| PUSH 节点输入 | `Send.arg` 合并到状态快照，另附 `__state__` | 只传 `Send.arg` |
| interrupt 并行批 | 整步回滚、整批重跑 | 保留已完成任务结果，重跑未完成任务 |
| 并发写错误类型 | `ConcurrentWriteError(RuntimeError)` | `InvalidUpdateError` |
| 裸 dict 声明的 schema | 拿不到字段类型，reducer 通道初值为 `MISSING`（读未写过的字段会 `KeyError`） | TypedDict/Annotated 一定拿得到类型 |
| checkpoint 历史 | 每个超步存一份 | 额外存「输入」那一份（`step=-1`/`0`） |
| `DeltaChannel` reducer 签名 | 二元 `(old, new)` 为主，批量用 `BatchReducer` 标记 | 统一批量 `(state, list[writes])` |
| `DeltaChannel` 的 `Overwrite` 语义 | 不支持（minigraph 没有 `Overwrite` 概念） | 支持，作为重放时的重置点 |
| `writes` 表的写入顺序键 | `idx`（同一步内按产生顺序） | `writes_sort_key(task_path, task_id, idx)` |
| `recursion_limit` 的计数范围 | **单次调用**走了多少步 | **单次调用**走了多少步 |

这些差异均有注释或测试标注；其余核心调度行为通过差分场景验证。

---

## 验证

```bash
# 单元与通道回归测试
.venv/bin/python -m pytest tests/ -q

# 与真实 LangGraph 做同进程差分；解释器需同时能 import langgraph 和本仓库
python tests/difflang/run.py

# 差分测试也能用 pytest 跑（需要装了 langgraph 的解释器）
/tmp/lgvenv/bin/python -m pytest tests/test_diff_langgraph.py tests/test_diff_checkpoint.py -v
```

当前包含 69 个单元/回归测试和 23 个差分场景（14 个超步场景 + 9 个检查点场景）。
差分比较：
每超步节点集合、超步总数、最终状态；失败场景还会直接插桩比较报错前的节点调用次数。
注意真实 LangGraph 的 `stream_mode="updates"` 是**按任务**产出，不是按超步；
harness 使用 `stream_mode="debug"` 中的超步号还原并行批次。

---

## 对照速查表

| 概念 | LangGraph | minigraph | 文件位置 |
| --- | --- | --- | --- |
| 状态声明 | TypedDict + `Annotated` | `class S(State)` + `Annotated` | `state.py` |
| 通道 | `BaseChannel` 及各实现 | 同名精简实现 | `channels.py` |
| 图编译 | `attach_node/edge/branch` | `StateGraph._compile` | `builder.py` |
| 条件边 | `add_conditional_edges` | 同名 | `builder.py` |
| 动态路由 | `Command(goto=...)` | 同名 | `control.py` |
| 并行扇出 | `Send(node, arg)` | 同名 | `control.py` |
| 人工中断 | `interrupt()` | 同名 | `context.py` |
| 持久化 | `InMemorySaver` / `SqliteSaver` 等 | `InMemorySaver` / `SqliteSaver` | `checkpoint.py`、`checkpoint_sqlite.py` |
| 执行内核 | Pregel loop | `CompiledGraph._run` | `runtime.py` |
| 触发判定 | `prepare_next_tasks/_triggers` | 同名精简实现 | `runtime.py` |

---

## 推荐阅读顺序

1. 跑 `python examples/01_react_agent.py`，观察 ReAct 回边。
2. 读 `minigraph/builder.py` 的 `_compile()`：边如何变成通道写入和订阅。
3. 读 `minigraph/channels.py`：重点看 `EphemeralValue` 与 `NamedBarrierValue`。
4. 读 `minigraph/runtime.py` 的 `_prepare_next_tasks`、`_triggers`、`_apply_writes`，
   最后看 `CompiledGraph._run` 如何把它们串起来。
5. 跑 `python examples/02_hitl_and_fanout.py`，再读 checkpoint 恢复逻辑。
6. 跑 `python tests/difflang/run.py`，对照真实 LangGraph 的执行轨迹。
7. `cp .env.example .env` 填 key，选 `LLM_PROVIDER`（anthropic / minimax / 火山引擎·方舟），
   跑 `uv run --extra llm python examples/03_real_llm_agent.py`，看同一张图换成真模型驱动
   （官方 anthropic SDK + python-dotenv，minigraph 内核仍零依赖）。
8. 跑 `uv run --extra llm python examples/04_deep_agent.py`：`create_deep_agent`
   （模仿 deepagents）一键组出带规划（write_todos）、真实文件 backend、自动摘要、
   子代理（task）的深度代理，复用同一份 .env 配置；文件写入 `.deep_agent_workspace/`。

### 配套文档

| 文档 | 讲什么 |
| --- | --- |
| [`docs/LangGraph内核设计动机.md`](docs/LangGraph内核设计动机.md) | **为什么 LangGraph 要设计成"通道 + 版本 + 水位线"**：动机、权衡与代价（对照真实源码） |
| [`docs/runtime超步循环详解.md`](docs/runtime超步循环详解.md) | `CompiledGraph._run` 的超步循环：算触发 → 并发跑 → 收写入 → 推版本 |
| [`docs/channel通道模型详解.md`](docs/channel通道模型详解.md) | 各类通道的语义，重点是 `is_available()` 何时为真、通道活多久 |
| [`docs/versions与seen水位线详解.md`](docs/versions与seen水位线详解.md) | `versions` / `seen` 两张表分别什么时候被谁改，含一条真实日志的逐行解读 |

最后如果能一句话说清下面这个问题，就说明抓住了内核：

> 为什么节点写一次 `branch:to:b`，`b` 就会醒，而写普通状态字段不会？

答案：**前者是 b 订阅的触发通道，写入会推进通道版本并越过 b 的水位线；
后者只是数据通道，参与状态读取，不参与节点触发。**
