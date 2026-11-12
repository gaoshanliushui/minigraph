# `CompiledGraph._run` 超步循环详解（通道触发版）

> 本文对应当前的 `minigraph/runtime.py`。内核已经从“边列表调度”重写为
> “channel 写入 + 通道版本 + 节点水位线”。
>
> 对照真实 LangGraph：`pregel/_algo.py` 的 `prepare_next_tasks`、`_triggers`、
> `apply_writes`，以及 `graph/state.py` 的 `attach_node/edge/branch`。
>
> 配套阅读：《`channel通道模型详解.md`》（各类通道的语义与可用性规则）。

---

## 0. 一句话定位

```python
while True:
    tasks = self._next_tasks(rt)      # ① 变化通道 -> 订阅者 -> 待执行任务
    if not tasks:
        break

    snapshot = rt.read()               # ② 同一超步共享同一状态快照
    runs = self._assign_slots(...)     # ③ 给每个任务发一个 interrupt 槽位
    results = _execute_batch(...)      #   本批任务并发执行
    writes, ran, hit = self._collect_writes(...)   # ④ 节点结果 -> 状态/触发通道写入

    if hit is not None:                # 有 interrupt：整步不提交，存 checkpoint 后返回
        self._suspend(...)
        return

    self._advance(rt, tasks, writes)   # ⑤⑥ 落盘：水位线 / consume / apply / 自清
```

`_advance()` 内部固定四小步，顺序不能换：

```python
rt.advance_watermarks(tasks)          # 1. 记录本步执行前的 versions_seen
next_version = rt.next_version()
rt.updated = set()
rt.consume_triggers(tasks, ...)       # 2. consume 触发通道（屏障重新武装）
rt.updated |= _apply_writes(...)      # 3. apply_writes 合并并推进版本
rt.expire_unwritten(written, ...)     # 4. EphemeralValue 自清，得到下一轮 updated_channels
```

这里的“图”不再参与运行时查下一跳。运行时只面对通道、通道版本和节点水位线。

### 代码地图

| 想看什么 | 去哪 |
| --- | --- |
| 一个超步的骨架 | `CompiledGraph._run()` |
| 三张表（通道值 / 版本 / 水位线）长什么样 | `_RuntimeState`（`runtime.py` 顶部） |
| 触发判定 | `_triggers()`、`_prepare_next_tasks()` |
| 节点返回值怎么变成通道写入 | `_normalize()`、`_control_writes()` |
| 版本号与水位线怎么推进 | `_RuntimeState.advance_watermarks()` / `consume_triggers()` / `expire_unwritten()` |
| 中断怎么冻结、恢复怎么重跑 | `_suspend()`、`_assign_slots()` |

调试时加 `config={"debug": True}`，每个超步会打印三张表的快照（版本号 / 水位线 /
可用通道），不用再靠脑内模拟。

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

### 8.1 多个节点同时挂起

同一超步里可能有多个节点各自调 `interrupt()`。它们**全都**要记下来 ——
只记一个的话，另一个重跑时永远等不到自己的值，两个中断会互相顶死
（一个答复完，另一个又挂起，永远转不完）。

因此 checkpoint 里除了 `interrupt`（第一个，兼容旧写法）还存了：

| 字段 | 内容 |
| --- | --- |
| `interrupts` | 本步**所有**挂起点：`{slot, node, idx, push, step, value}` |
| `resumes` | 已收集到的 resume 值：`槽位 -> 值队列` |
| `slots` | 本批任务的槽位分配（重跑时槽位不能漂移） |

**`resumes` 必须跨 resume 累积**：整步回滚意味着挂起的节点会重跑，
重跑时还得拿到当初那个值。槽位存的是**队列**而不是单个值，
一个节点里连续多次 `interrupt()` 时按调用顺序出队。

### 8.2 resume 的三种形态

```python
# 标量：答复下一个待恢复的中断（对齐 LangGraph 的 "resume the next interrupt"）
app.invoke(None, {**cfg, "resume": "批准"})

# 字典：一次性答复多个（键是节点名或槽位）—— 对齐 LangGraph 的 resume map
app.invoke(None, {**cfg, "resume": {"a": "A", "b": "B"}})

# 字典 + 列表：一个节点里连续多次 interrupt 的值，按调用顺序取
app.invoke(None, {**cfg, "resume": {"ask": ["V1", "V2"]}})
```

逐个答复时（标量形态），每次答复一个，剩下的继续挂着：

```text
第 1 次 invoke  -> 挂起（a、b 都在等）
第 2 次 resume  -> a 拿到值，b 还在等        -> 仍挂起
第 3 次 resume  -> b 也拿到值，整批提交      -> 图跑完
```

---

## 9. checkpoint、流式输出与时间旅行

普通超步结束后：

1. `step += 1`。
2. `stream()` 产出 `(逗号连接的节点名, step, 状态字段更新)`。
   触发通道不会泄漏到 `update`。
3. 重新计算一次 `upcoming`，写入 checkpoint 的 `next/tasks`。
4. 保存通道值、版本、水位线。

挂起时（有人调了 `interrupt()`）额外存三样，见 §8.1：
`interrupts`（所有挂起点）、`resumes`（已收集的 resume 值）、`slots`（本批槽位）。

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

- `tests/test_minigraph.py`：行为测试（含 schema 提取、自环水位线规则）。
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
2. `_advance()` 中水位线更新与 `_apply_writes()` 的先后顺序。
3. `NamedBarrierValue.consume()`：屏障如何在下一轮重新武装。

`config={"debug": True}` 会把每个超步的三张表打出来 —— 上面三处都能靠它验证。

---

## 12. 版本兼容：Python 3.14 的注解陷阱

`State.__init_subclass__` 从**本类自己的**注解里提取 reducer。它读的是
`cls.__annotations__`，不是 `cls.__dict__["__annotations__"]`：

```python
# 对（3.13 / 3.14 都成立）
for name, ann in getattr(cls, "__annotations__", {}).items():
```

Python 3.14 起 (PEP 649) 注解是惰性求值的，类字典里**没有**
`"__annotations__"` 这一项 —— 用 `__dict__` 取会拿到 `None`，于是 reducer
静默消失（`reducers` 变空 -> 运行时每个字段写入都报“写了未声明的字段”）。

两个版本下 `cls.__annotations__` 都只含本类自己声明的注解（3.10+ 起不再沿
MRO 继承），所以它对 3.13 / 3.14 是同一份语义。
