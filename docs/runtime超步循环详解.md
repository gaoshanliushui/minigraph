# `CompiledGraph._run` 超步循环详解（通道触发版）

> 本文对应当前的 `minigraph/runtime.py`。内核已经从“边列表调度”重写为
> “channel 写入 + 通道版本 + 节点水位线”。
>
> 对照真实 LangGraph：`pregel/_algo.py` 的 `prepare_next_tasks`、`_triggers`、
> `apply_writes`，以及 `graph/state.py` 的 `attach_node/edge/branch`。

---

## 0. 一句话定位

```python
while True:
    tasks = _prepare_next_tasks(       # 变化通道 -> 订阅者 -> 待执行任务
        channels, versions, seen, nodes,
        trigger_to_nodes=trigger_to_nodes,
        updated_channels=updated_channels,
    )
    if not tasks:
        break

    snapshot = _read_state(...)        # 同一超步共享同一状态快照
    results = _execute_batch(...)      # 本批任务并发执行
    writes = _normalize(...)           # 节点结果 -> 状态/触发通道写入

    # 有 interrupt：整步不提交，直接保存 checkpoint
    # 否则：
    # 1. 记录本步执行前的 versions_seen
    # 2. consume 触发通道（屏障重新武装）
    # 3. apply_writes 合并并推进版本
    # 4. EphemeralValue 自清，得到下一轮 updated_channels
```

这里的“图”不再参与运行时查下一跳。运行时只面对通道、通道版本和节点水位线。

---

## 1. 编译产物：运行前已经确定了什么

`StateGraph._compile()` 返回 `channels` 和 `nodes`，`CompiledGraph` 再建立
`trigger_to_nodes` 反向索引。

以 `a -> b` 为例：

```text
channels["branch:to:a"] = EphemeralValue(guard=False)
channels["branch:to:b"] = EphemeralValue(guard=False)

nodes["a"].triggers = ["branch:to:a"]
nodes["a"].writers  = ["branch:to:b"]
nodes["b"].triggers = ["branch:to:b"]

trigger_to_nodes["branch:to:a"] = ["a"]
trigger_to_nodes["branch:to:b"] = ["b"]
```

所以“a 跑完跑 b”被翻译成：**a 在超步末尾写一次 `branch:to:b`，
这个通道的版本前进一步；b 的水位线因此过期。**

状态字段则编译为数据通道：

```text
无 reducer / LAST_VALUE -> LastValue
有二元 reducer          -> BinaryOperatorAggregate
Send 队列               -> Topic("__pregel_tasks")
多源汇聚                -> NamedBarrierValue("join:a+b:c")
```

数据通道供节点读取，不参与 PULL 触发；触发通道负责唤醒节点。

---

## 2. 初始化与恢复

### 2.1 全新运行

1. 从编译出的通道原型复制一套运行期通道。
2. 已声明的输入字段写入数据通道；未声明输入键放进 `extra_values`，只读、不许写。
3. `__start__` 给所有入口点写 `branch:to:<entry>`。
4. 这些入口通道组成第一轮的 `updated_channels`。

多个 `add_edge(START, ...)` 会累加入口点，所以它们能在第一个超步并行执行。

### 2.2 从 checkpoint 恢复

checkpoint 中有三组内核状态：

```python
channel_values    # 通道值
channel_versions  # 通道版本
versions_seen     # 每节点、每触发通道的水位线
```

`_restore_channels()` 用 `from_checkpoint()` 重建通道，再恢复版本与水位线。
恢复后的第一轮 `updated_channels=None`，因此检查所有节点；之后重新启用
变化通道的候选节点优化。

`values` 只是用户可见的状态投影，包含状态字段和只读额外输入键，不包含
`branch:to:*`、`join:*`、`__pregel_tasks`。

---

## 3. 第一阶段：算出下一批任务

### 3.1 PUSH 轨

若 `__pregel_tasks` 非空，其中每个 `Send(node, arg)` 都生成一个任务：

```python
Task(name=send.node, send=send, idx=send 在 Topic 中的下标)
```

PUSH 不调用 `_triggers`，也不会推进目标节点 `branch:to:X` 的水位线。
这与真实 LangGraph 一致：Send 的目标是被显式派发的任务，不是被普通分支通道唤醒。

### 3.2 PULL 轨

有 `updated_channels` 时，先取候选节点：

```python
candidates = {
    node
    for chan in updated_channels
    for node in trigger_to_nodes.get(chan, ())
}
```

然后逐个执行触发判定：

```python
channels[chan].is_available() and (
    seen is None or channel_versions[chan] > versions_seen[node].get(chan, 0)
)
```

`seen is None` 表示节点从未运行，只要触发通道可用就可以执行。
候选节点最后按名字排序，保证同批任务顺序确定，interrupt 槽位才可稳定恢复。

---

## 4. 第二阶段：并发执行

`_read_state()` 把数据通道和 `extra_values` 合成快照。所有本批节点拿到同一份，
所以同超步内一个节点看不到另一个节点的写入——这就是 BSP 同步屏障。

- 单任务直接执行，避免线程池开销。
- 多任务交给 `ThreadPoolExecutor`。
- PULL 输入是状态快照。
- PUSH 输入由 `Send.arg` 和状态快照组成，这是 minigraph 的扩展。

resume 槽位形如：

```text
review@3#pull:0
worker@4#push:2
```

类型包含在槽位里，避免同一节点同一步既是 PULL 又是 PUSH 时槽位碰撞。

---

## 5. 第三阶段：把节点结果翻译成通道写入

`_normalize()` 不再返回“下一跳列表”，而是返回 `[(channel, value), ...]`。

| 节点结果 | 产生的写入 |
| --- | --- |
| `{"n": 1}` | 状态通道 `n` |
| `Command(update=...)` | 状态通道 |
| `Command(goto="b")` | `branch:to:b = None` |
| 条件边选择 `b` | `branch:to:b = None` |
| 静态边 `a -> b` | `branch:to:b = None` |
| 屏障边 `a -> join` | `join:... = "a"`（源节点自己的名字） |
| `Send("worker", arg)` | `__pregel_tasks` |

`Command(goto=...)` 不抑制静态边和条件边。节点直接返回 `list[Send]` 是
minigraph 的扩展，它会抑制该节点的静态/条件边。

条件边通过 `_fresh_view()` 读取“本节点写入经过 reducer 合并后”的状态。
首次写 reducer 字段时直接采用写入值，与 `BinaryOperatorAggregate` 的首次写入捷径一致。

---

## 6. 为什么必须先记水位线，再应用写入

假设自环节点 `n -> n`：

```text
本步开始前：branch:to:n version=7，n 的水位线=6
```

正确顺序：

```text
1. n 执行，先记 versions_seen[n][branch:to:n] = 7
2. 应用 n 的写入，branch:to:n version = 8
3. 下一步：8 > 7，n 再次触发
```

如果顺序反了：

```text
1. 应用写入，版本变成 8
2. 再把 n 的水位线记成 8
3. 下一步：8 不大于 8，自环错误停止
```

因此代码里“更新 versions_seen”必须位于 `_apply_writes()` 之前，
而且记录的是**节点执行前看到的版本**。

同一超步内所有变化通道共享一个新版本号，与真实 LangGraph 的全局版本计数方式一致。

---

## 7. `consume()` 与通道自清

这是两个不同动作：

### 7.1 `consume()`：订阅者已运行

本步 PULL 任务运行后，对它的触发通道调用 `consume()`。
多数通道是 no-op；`NamedBarrierValue` 会清空 `seen`，让屏障重新武装。
没有这一步，屏障 join 在循环里只能触发一次。

### 7.2 `update([])`：本步没人写入

超步末尾，对“本步没被写、但仍可用”的通道调用 `update([])`。
`EphemeralValue` 借此清空自己，所以分支信号只活一个超步。

只有更新后仍可用的通道才进入下一轮的 `updated_channels`。
屏障只写到一个来源时还未可用，不会唤醒 join；等所有来源到齐后才进入候选集合。

---

## 8. interrupt：为什么整批会重跑

minigraph 遇到 `GraphInterrupt` 时，跳过：

- `versions_seen` 更新；
- `consume()`；
- `_apply_writes()`；
- 通道自清。

然后保存执行前的通道快照和 interrupt 槽位。

恢复时，通道版本和水位线都与该步执行前完全相同，所以
`_prepare_next_tasks()` 自然算出同一批任务，整批重跑。这不需要单独维护
“哪些节点要重跑”的边列表。

注意：真实 LangGraph 1.2.16 在并行批中会保留已完成任务的结果，只重跑未完成任务；
minigraph 选择整步回滚作为教学简化。两者在现有测试场景最终状态一致，
但副作用次数不同，不能据此声称真实 LangGraph 也总是整批重跑。

---

## 9. checkpoint、流式输出与时间旅行

普通超步结束后：

1. `step += 1`。
2. `stream()` 产出 `(逗号连接的节点名, step, 状态字段更新)`。
   触发通道不会泄漏到 `update`。
3. 重新计算一次 `upcoming`，写入 checkpoint 的 `next/tasks`。
4. 保存通道值、版本、水位线。

`update_state(..., as_node="b")` 的做法是：

1. 把人工修改写入状态通道，走 reducer。
2. 写 `branch:to:b` 并推进版本，保证 b 会触发。
3. 把其他节点的水位线补齐，避免它们被旧信号唤醒。

`MISSING` 在 `copy/deepcopy` 后仍保持单例，否则空通道会在 checkpoint 恢复时
被误认为“有值”。

`recursion_limit` 只在“还有任务要跑”时检查；图恰好在第 limit 步完成是正常结束。

---

## 10. 两条容易混淆的汇聚语义

### 同一超步的两条入边

```text
step 2: a、b 都写 branch:to:c
        -> 同一超步共享一次版本推进
step 3: c 执行一次
```

### 不同超步的两条入边

```text
step 2: a 写 branch:to:c
step 3: c 第一次执行；b2 写 branch:to:c
step 4: c 第二次执行
```

要等待多个来源后只执行一次，使用：

```python
g.add_edge(["a", "b"], "c")
```

它会创建 `NamedBarrierValue`，而不是复用两条独立的 `branch:to:c` 写入。

---

## 11. 如何验证而不是凭感觉

- `tests/test_minigraph.py`：原有 16 个行为测试。
- `tests/test_channel_runtime.py`：通道、水位线、屏障重用、PUSH、恢复等回归测试。
- `tests/difflang/`：同一张图分别跑 minigraph 与真实 LangGraph。

```bash
.venv/bin/python -m pytest tests/ -q
python tests/difflang/run.py   # 解释器需要安装 langgraph
```

差分 harness 用 `debug` 流中的超步号还原真实 LangGraph 的并行批次；
`updates` 流按任务产出，不能直接把每个 chunk 当成一个超步。
对失败场景，harness 在 `add_node` 处插桩，直接比较报错前的节点调用次数。

阅读代码时最值得逐步调试的三个位置是：

1. `_prepare_next_tasks()`：候选节点如何从变化通道产生。
2. `_run()` 中水位线更新与 `_apply_writes()` 的先后顺序。
3. `NamedBarrierValue.consume()`：屏障如何在下一轮重新武装。
