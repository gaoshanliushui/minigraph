# `CompiledGraph._run` 超步循环详解

> 目标文件：`minigraph/runtime.py` 的 `CompiledGraph._run`（L90–L224）
>
> 这个方法只有 130 多行，但它是整个 minigraph（以及真 LangGraph）的**执行内核**。
> checkpoint、人工审批、时间旅行、流式输出，全都是这个 `while` 循环的副产品。
>
> 本文按代码顺序逐段拆解，并回答"为什么必须这么写"。

---

## 目录

- [0. 一句话定位](#0-一句话定位)
- [1. 进入 `_run` 之前](#1-进入-_run-之前)
- [2. 阶段 A：初始化与恢复判定（L91–L138）](#2-阶段-a初始化与恢复判定l91l138)
- [3. 阶段 B：超步循环（L143–L222）](#3-阶段-b超步循环l143l222)
- [4. 阶段 C：收尾（L224）](#4-阶段-收尾l224)
- [5. 辅助函数逐个击破](#5-辅助函数逐个击破)
- [6. 完整追踪：01_react_agent.py](#6-完整追踪01_react_agentpy)
- [7. 中断续跑的完整追踪](#7-中断续跑的完整追踪)
- [8. 边界情况与已知限制](#8-边界情况与已知限制)
- [9. 设计决策速查](#9-设计决策速查)

---

## 0. 一句话定位

```python
while tasks:                          # 没有任务 = 图跑完了
    snapshot = dict(values)            # ① 屏障：全批节点看同一份快照
    results = _execute_batch(...)      # ② 并发：节点无状态执行，返回"增量"
    values = _merge_writes(...)        # ③ 屏障：收齐所有增量，用 reducer 一次性合并
    tasks = _dedupe_tasks(nxt)         # ④ 路由：基于合并后的状态决定下一批
```

这就是 **Pregel / BSP（Bulk Synchronous Parallel）** 模型。LangGraph 之所以能
并行、能断点续跑、能顺序无关，全靠这四步的严格分离。

---

## 1. 进入 `_run` 之前

对外 API 只有三个，它们都是 `_run` 的薄包装：

| API | 实现 | 与 LangGraph 的关系 |
| --- | --- | --- |
| `invoke(input, config)` | L40–L43：`for _ in self.stream(...): pass`，返回 `dict(self._last_values)` | 把生成器跑干，取最终状态 |
| `stream(input, config)` | L50–L51：`yield from self._run(...)` | 生成器，每超步 yield 一次 |
| `get_state / get_state_history / update_state` | L53–L87 | 直接操作 `checkpointer`，不经过 `_run` |

**关键**：`_run` 本身是**生成器**。所以

```python
for node, step, update in app.stream(...):   # 消费者中途 break
```

会让 `_run` 挂起在 L212 的 `yield` 上 —— 此时 `self._last_values` 已经更新，
但 **L217 的 `saver.put` 还没执行**。如果用了 checkpointer 又提前退出流，
`get_state()` 看到的会比内存里的状态少最后一步。

### 两个跨方法的共享变量

```python
self._last_values                    # L211 每步刷新；invoke 的返回值来源
self._interrupted                    # L98 每次 _run 开头清空；挂起时写入 hit.value
```

`interrupted` 属性（L45–L48）读的就是 `_interrupted`。

---

## 2. 阶段 A：初始化与恢复判定（L91–L138）

### 2.1 解析 `config`（L91–L98）

```python
thread_id   = cfg.get("thread_id")              # 断点续跑的 key
debug       = cfg.get("debug", False)           # 打印每步调度
limit       = cfg.get("recursion_limit", 50)    # 超步数上限，防死循环
has_resume  = "resume" in cfg
resume_val  = cfg.get("resume")
self._interrupted = None
```

> 注意 `has_resume` 用的是 `"resume" in cfg` 而不是 `resume_val is not None` ——
> 因为 resume 值**允许是 `None` 或 `False` 这种假值**（用户在审批弹窗点了"取消"）。
> 用 `is not None` 判断会让这种恢复永远不生效。

### 2.2 载入 checkpoint（L99）

```python
snap = saver.get(thread_id, cfg.get("checkpoint_id")) if (saver and thread_id) else None
```

- 没配 checkpointer、或没给 `thread_id` → `snap = None` → 只能全新运行
- 给了 `checkpoint_id` → 精确回到某个历史点（时间旅行分支）
- 只给 `thread_id` → `saver.get` 取 `history[-1]`，即最新一份

### 2.3 两条互斥的初始化路径（L101–L128）

```python
if input is None and snap is not None:      # ---- 路径 B：从 checkpoint 恢复
else:                                       # ---- 路径 A：全新运行
```

| | 路径 A：全新运行 | 路径 B：恢复 |
| --- | --- | --- |
| `values` | `dict(input or {})` | `snap.values.copy()` |
| `tasks` | `[graph.entry]` | `list(snap.tasks)` |
| `step` | `0` | `snap.step` |
| 触发条件 | 传了 input，或压根没 checkpoint | `input=None` 且有 snap |

**路径 A 的两个细节**：

```python
values = dict(input or {})
```

注意注释明确说了：**输入不做字段校验**。图内只读的键（比如 `user_id`、租户 ID）
可以直接塞进 input，只有"节点写入"才要求字段在 `State` 里声明过。这是刻意的
非对称 —— 见 §5 的 `_merge_writes`。

```python
tasks = [graph.entry] if graph.entry else []
```

没设入口时 `tasks` 为空列表 → `while tasks` 直接不进 → `_run` 立即返回空状态。

**路径 B 的中断处理**（L106–L114）：

```python
if snap.interrupt:
    if not has_resume:
        raise RuntimeError(f"该 thread 停在 interrupt({snap.interrupt['value']!r})，需要 config={{'resume': ...}} 才能继续")
    _ctx()["resume"] = {snap.interrupt["slot"]: resume_val}
```

这一行是整个 interrupt 机制的**接缝**：

- checkpoint 里记的是「哪个**槽位**在等值」，不是「哪个节点在等值」
- 恢复时把用户传进来的 `resume_val` **按槽位**塞进 thread-local 上下文
- 节点里 `interrupt()` 靠比对自身 slot 命中这个 store，取出值并 pop

为什么必须是槽位而不是节点名？因为循环图里同一个节点会反复挂起（见
`test_interrupt_can_fire_multiple_times_in_a_loop`）。用节点名当 key，第二次
挂起会误取上一轮的 resume 值。槽位 `节点名@步号#批内序号` 保证唯一。

### 2.4 `pending_*`：把槽位还原到正确的位置（L130–L138）

```python
pending_node = pending_slot = None
pending_idx = 0
pending_step = None
if snap is not None and snap.interrupt:
    pending_node = snap.interrupt["node"]
    pending_slot = snap.interrupt["slot"]
    pending_idx = snap.interrupt.get("idx", 0)
    pending_step = snap.interrupt["step"]
```

这四个变量存在的唯一理由：**恢复后要重跑整个超步，而超步里的槽位必须和当初一模一样**，
否则 `interrupt()` 取不到 resume 值。

拆开看四个字段各自解决什么问题：

| 变量 | 用途 | 对应 checkpoint 字段 |
| --- | --- | --- |
| `pending_node` | 判断"是哪个任务该沿用旧槽位" | `interrupt["node"]` |
| `pending_slot` | 该任务沿用这个槽位（而不是按新 step 算） | `interrupt["slot"]` |
| `pending_idx` | 批内第几个，防止认错同名任务 | `interrupt["idx"]` |
| `pending_step` | **步号不自增**，保持和挂起时一致 | `interrupt["step"]` |

`pending_step` 尤其关键：如果恢复后 step 变成 `挂起步号 + 1`，重算出的槽位是
`节点名@4#0` 而 checkpoint 里记的是 `节点名@3#1`，对不上，resume 值永远取不到。

---

## 3. 阶段 B：超步循环（L143–L222）

### 3.1 循环条件与递归上限（L143–L148）

```python
while tasks:
    if step >= limit:
        raise RuntimeError(f"超过 recursion_limit={limit}，疑似死循环")
    step = pending_step if pending_step is not None else step + 1
    pending_step = None
```

- **循环的终止条件是"没有待执行任务"**，不是"跑到 END" —— `END` 只是被过滤掉的
  一个普通字符串（L320/L350），过滤后 `nxt` 为空，下一轮循环自然结束。
- 全新运行时 `step = 0`，第一次进循环 `step = 1`。所以 `recursion_limit=50`
  实际允许跑 50 个超步。
- `pending_step` 用完立刻置 `None`，所以**只有恢复后的第一个超步**不吃掉自增，
  第二个超步就恢复正常节奏。

### 3.2 建立屏障：快照（L151）

```python
batch = list(tasks)
snapshot = dict(values)      # 本超步所有节点看到的都是同一快照
```

**这是 BSP 同步屏障的物理实现。** 一行 `dict()` 做了两件事：

1. **浅拷贝**——节点拿到的是独立 dict，节点内部随便改不会污染主状态
2. **只拷贝一次**——本超步所有节点拿的是**同一个对象**

所以：节点 A 无论跑多快，都读不到同批节点 B 的输出。反过来，节点返回值如果
不是 dict（而是原地改了 `state`），这次写入只污染快照、不会进入 `values` ——
**这就是"节点必须返回 dict 而不能改 state"的强制力来源**（答案不靠文档，靠类型
系统 + 这行浅拷贝）。

⚠️ 浅拷贝意味着嵌套对象是共享的：`{"draft": {...}}` 里的 `...` 仍是同一个
dict，被节点原地修改会串到主状态。要避免这点需自行拷贝，或只写标量。

### 3.3 分配槽位（L152–L159）

```python
runs: list[tuple[Any, str, str]] = []
for idx, task in enumerate(batch):
    name = task.node if isinstance(task, Send) else task
    slot = pending_slot if (pending_node == name and idx == pending_idx) \
        else f"{name}@{step}#{idx}"
    runs.append((task, name, slot))
```

`runs` 里每项是 `(原始任务, 节点名, 槽位)` 三元组。

**为什么要槽位？** 槽位是 `interrupt()` 的钥匙。它必须满足：

- **稳定**：恢复后重算出来要一样 → 依赖 `step` 和 `idx` 都是确定性的
- **唯一**：同一超步里同名节点可能出现多次（循环回边导致 `agent` 同时排了两份），
  所以必须加批内序号 `#idx`

`f"{name}@{step}#{idx}"` 三段式正好满足这两条。只有当 `(name, idx)` 匹配上
`pending_node/pending_idx` 时，才用 checkpoint 里记的旧槽位。

### 3.4 并发执行（L161）

```python
results = _execute_batch(graph, runs, snapshot)
```

详见 §5.1。三个要点：

1. 单任务时**不起线程池**，直接调用（省掉线程切换，绝大多数超步走这条路）
2. 多任务时 `ThreadPoolExecutor(max_workers=min(16, len(runs)))`，
   `f.result()` 逐个收集 —— **保证返回顺序和 `runs` 顺序一致**（不是完成顺序）
3. 所有线程共享同一个 `snapshot` 对象

### 3.5 结果分类（L171–L184）

```python
for idx, (task, name, slot) in enumerate(runs):
    out = results[idx]
    if isinstance(out, GraphInterrupt):     # ---- 挂起
        hit, hit_node, hit_slot, hit_idx = out, name, slot, idx
        continue
    if isinstance(out, _NodeError):         # ---- 失败
        raise RuntimeError(f"节点 {name!r} 执行失败：{out.exc}") from out.exc
    ran.append(name)
    update, routes = _normalize(name, out, snapshot, graph)
    writes.extend((name, k, v) for k, v in update.items() if not _is_noop(out, k))
    nxt.extend(routes)
```

三个分支：

- **`GraphInterrupt`**：记下挂起信息，`continue` 跳过它的写入和路由。
  注意 `hit` 会被后面的中断**覆盖**，所以同超步多个节点同时中断时只记录最后一个
  （见 §8 边界情况）
- **`_NodeError`**：立即 `raise`。此时线程池里所有任务都已跑完（`.result()` 是阻塞的），
  但 `writes` 一条都没合并 —— **失败的超步同样是零提交**，和挂起一样干净
- **正常**：`ran` 记录本步跑了谁（给 `yield` 和 `debug` 用），
  `_normalize` 拆出 `(update, routes)`，`writes` 收进待合并队列

`_is_noop(out, k)` 过滤掉 `{key: None}`（详见 §5.5）。**这一步过滤很关键**：
如果不过滤，一个节点写 `{"debug": None}` 而另一个写 `{"debug": x}`，
就会因为 `debug` 字段没有 reducer 而误报"并发写冲突"。

### 3.6 屏障：批量合并（L186–L188）

```python
# 本超步全部写入在此一次性合并 —— 并行分支就是在这里汇流的
if writes and hit is None:
    values = _merge_writes(reducers, values, writes)
```

**这是整个内核最重要的一行。**

- `values` 是**新对象**（`_merge_writes` 内 `out = dict(base)`），不会污染 `snapshot`
- `hit is None` —— 一旦挂起，本步**所有写入全部丢弃**，包括那些已经成功跑完的
  兄弟节点。这就是 BSP 的"一步要么全提交，要么全不提交"
- 并行分支在这里汇流：`by_key` 把同字段的多个写入聚成一组，交给 reducer 逐个 reduce

```python
# 超步 3:  [fetch_data ✓]  [human_review → interrupt]
#          ↓ 本步作废
# checkpoint: values 不含 fetch_data 的写入, tasks = [fetch_data, human_review]
# 恢复后:     两个节点都重新执行
```

直接推论：**节点必须幂等**。不是因为文档要求，而是因为挂起/崩溃会让整个超步重跑。

### 3.7 挂起处理（L190–L203）

```python
if hit is not None:
    ckpt = Checkpoint(
        id=saver.new_id() if saver else "ephemeral",
        step=step, values=values,                        # ← 合并前的状态！
        next=tuple(_names(batch)), tasks=list(batch),     # ← 不是 nxt！
        interrupt={"slot": hit_slot, "node": hit_node, "idx": hit_idx,
                   "step": step, "value": hit.value},
    )
    if saver and thread_id:
        saver.put(thread_id, ckpt)
    self._last_values = values
    self._interrupted = hit.value
    yield hit_node, step, {}
    return
```

三个容易看错的点：

1. **`values` 是合并前的** —— 因为 L187 的 `hit is None` 拦住了合并。这是**正确的**：
   既然本步作废，checkpoint 就该存"进入本步之前"的状态
2. **`next` / `tasks` 是 `batch` 不是 `nxt`** —— 本步作废，重跑的就是 `batch`
   （原批次，包括已经跑成功的兄弟节点）。`nxt` 是"如果没挂起"的假想下一跳，不作数
3. **`yield hit_node, step, {}`** —— 第三个值（update）是**空 dict**，因为本步
   没有任何写入被采纳。流式消费者在这里看到的是"停住了"，而不是"这个节点写了什么"

`return` 结束生成器 → `while` 不再循环 → 图被冻住，等下次带 `resume` 重新 `invoke`。

### 3.8 汇合去重（L205–L206）

```python
nxt = _dedupe_tasks(nxt)
```

fan-in 语义：同超步里多个分支都指向 `join` 节点 → 只调度一次。
`Send` 永不去重（每个 `Send` 携带不同 `arg`，本来就是 N 个独立任务）。
详见 §5.4。

### 3.9 流式输出（L208–L214）

```python
if debug:
    print(f"  step {step}: {ran} -> {_names(nxt)}")

self._last_values = values
yield ",".join(ran), step, {
    k: values.get(k) for k in dict.fromkeys(k for _, k, _ in writes)
}
```

`stream()` 每次 yield 一个三元组 `(节点名, 步号, 增量视图)`：

- `节点名`：本超步**跑过的所有节点**用逗号连接 —— 并行时会看到 `"fetch,analyze"`
- `步号`：从 1 开始
- `增量视图`：**合并之后**的字段值（注意不是节点原始返回值），所以多个节点写
  `messages` 时这里看到的是已经 `add_messages` 合并完的完整列表。
  `dict.fromkeys(...)` 保序去重，且因为 L183 过滤了 noop，这里不会出现 `None`

这个"yield 合并后的值"和真 LangGraph 的 `stream_mode="updates"` 不同 ——
后者给的是节点原始返回值。minigraph 这里选择给合并后视图，配合
`add_messages` 的按 id 去重，用起来更接近 `stream_mode="values"`。

### 3.10 保存 checkpoint（L216–L220）

```python
if saver and thread_id:
    saver.put(thread_id, Checkpoint(
        id=saver.new_id(), step=step, values=values.copy(),
        next=tuple(_names(nxt)), tasks=list(nxt),
    ))
```

每个超步存一份完整快照。注意 `next` 只存**节点名**，`tasks` 才保留 `Send` ——
`Send.arg` 是上游节点**运行期**才决定的，不存进 checkpoint 就永久丢失，
恢复后无法重建并行分支。详见 `checkpoint.py` L35–L40 的注释。

> ⚠️ **checkpoint 顺序在 `yield` 之后**（L212 yield，L217 put）。消费者中途
> `break` 掉 stream，最后这份 checkpoint 不会落盘。

### 3.11 推进（L221–L222）

```python
tasks = nxt
_ctx()["resume"] = {}
```

清空 resume store 很重要：否则下一个超步的节点会捡到上一轮残留的 resume 值。
（其实 `_ctx()["slot"]` 也随之变化，即使不清也不会误命中 —— 但显式清空更安全，
且和 §2.3 路径 B 的 `_ctx()["resume"] = {}` 对称。）

---

## 4. 阶段 C：收尾（L224）

```python
_ctx()["slot"] = None
```

`while` 自然结束（图跑到 END 或 tasks 耗尽）后清理 slot，防止下次
`invoke` 时有节点在图外误调 `interrupt()`。

注意挂起路径是 `return` 提前退出的，**不会**执行这行 —— 这没关系，
因为下次 `_run` 开头会在 `_run_task` 里重新注入 slot。

---

## 5. 辅助函数逐个击破

### 5.1 `_execute_batch`（L245–L268）

```python
def arg_for(task):
    return snapshot if not isinstance(task, Send) else _wrap_send(task, snapshot)

resume = dict(_ctx().get("resume") or {})     # 必须在主线程读一次

if len(runs) == 1:
    task, name, slot = runs[0]
    return [_run_task(name, graph.nodes[name]["fn"], arg_for(task),
                      graph.nodes[name].get("retry"), slot, resume)]

with ThreadPoolExecutor(max_workers=min(16, len(runs))) as pool:
    futures = [pool.submit(...) for task, name, slot in runs]
    return [f.result() for f in futures]
```

- **`resume` 必须在主线程提前取出**。`_ctx()` 是 `threading.local`，工作线程
  看不到主线程的属性。这是"槽位/恢复值"机制里第二个容易踩的坑（第一个是 §2.3 的槽位唯一性）
- **`[f.result() for f in futures]` 而非 `as_completed`** —— 保证结果顺序与
  `runs` 对齐，这样 L172 的 `results[idx]` 才对得上
- **上限 16**：扇出 100 个 `Send` 也不会开 100 个线程

### 5.2 `_run_task`（L232–L242）

```python
def _run_task(name, fn, arg, retry, slot, resume=None):
    _ctx()["slot"] = slot          # 重新注入到本线程
    _ctx()["resume"] = dict(resume or {})
    try:
        return _run_with_retry(name, fn, arg, retry)
    except GraphInterrupt as gi:
        return gi                  # 挂起不是错误，包装成返回值带回主线程
    except Exception as exc:
        return _NodeError(exc)      # 节点错误也包装成返回值
```

**用返回值而不是异常来传递结果**，是因为：异常从线程抛出来会被 `Future` 包一层，
丢失原始类型。统一包成"正常返回值"后，主线程 L173/L179 就能干净地分类处理，
而不需要在 `except` 里做类型判断。

`GraphInterrupt` 必须**排除在 retry 之外**（见 `_run_with_retry` 的 L407–408）——
挂起是控制流，不是故障，重试它只会立刻再挂一次。

### 5.3 `_normalize`（L311–L350）

把节点的返回值拆成 `(update, routes)`。四条分支，**优先级从上到下**：

```python
# ① Command：节点自己决定去哪 + 顺带写状态
if isinstance(out, Command):
    update = dict(out.update or {})
    goto = out.goto
    routes: list[Any] = []                        # ← goto is None 时留空
    if goto is not None:
        routes = list(goto) if isinstance(goto, (list, tuple)) else [goto]
    routes = [r for r in routes if r != END]
    return update, routes                          # ← 提前 return！
```

**注意这个 `return` 提前返回** —— `Command` 一旦命中，后面 L332 起的静态边 /
条件边解析**完全不会执行**。

所以 `Command(update={"x": 1})`（只写状态、不给 `goto`）的实际语义是
**"写完状态就停在这里"**，`nxt = []` → 循环结束，节点后续的静态边和条件边全部被忽略：

```python
g.add_edge("a", "b")                              # 这条边不会生效
g.add_node("a", lambda s: Command(update={"x": "hi"}))
g.add_node("b", lambda s: {"y": "there"})
g.set_entry_point("a")
app.invoke({})   # -> {'x': 'hi'}     b 从未执行
```

而 `control.py` L18–L20 的字段注释明写 `None → 不指定下一跳，走默认静态边`。
**注释与实现不一致**：要修的话把 L321 改成仅在 `goto is not None` 时提前 return，
否则让 `routes` 继续向下走到 L332 的边解析。用的时候记住这个坑：**用了 `Command`
就要给全 `goto`**。

```python
# ② 直接返回 list[Send]：等价于"在条件边里做扇出"
if isinstance(out, (list, tuple)) and all(isinstance(x, Send) for x in out):
    return {}, list(out)

# ③ 普通 dict / None
if out is not None:
    if not isinstance(out, Mapping):
        raise TypeError(f"节点 {node!r} 必须返回 dict / None / Command，收到 {type(out).__name__}")
    update = dict(out)

# ④ 路由
statics = [dst for src, dst in graph.edges if src == node]
cond = next((b for b in graph.branches if b["source"] == node), None)
if cond is None:
    return update, [d for d in statics if d != END]
```

**条件边的路由（L338–L350）**：

```python
view = {**values, **update}           # 合并后的状态视图
decision = cond["path"](view)
```

注释说得很准：**"path 看到的是『合并之后』的状态更合理"**。但注意它只叠加了
**本节点自己的** `update`，不是同超步其他节点的。因为 §3.6 的合并发生在 `_normalize`
**之后**，条件边路由时全局 `values` 还没更新。这意味着：**扇出场景下，一个节点
的条件边看不到兄弟节点的写入**（跨超步才看得到）。这是有意的取舍 ——
`Command` 存在的意义之一就是给需要"路由时看自己内部状态"的场景用。

路由结果三种处理：

| path 返回 | 处理 |
| --- | --- |
| `Send` 实例 | 直接作为任务 |
| `END` | 过滤掉 |
| 其他 key | 有 `path_map` 就翻译，没有就当节点名 |

多个返回值（`return ["a", "b"]`）→ 扇出到两个节点，下个超步并发。

### 5.4 `_dedupe_tasks`（L271–L283）

```python
seen: set[str] = set()
for t in tasks:
    if isinstance(t, Send):
        out.append(t); continue      # Send 永不去重
    if t in seen:
        continue
    seen.add(t); out.append(t)
```

**join 语义**：N 条分支指向同一节点 → 只调度一次。

`Send` 不去重是必须的：`Send("worker", {"topic": "a"})` 和
`Send("worker", {"topic": "b"})` 虽然 `node` 相同，但 `arg` 不同，是两个独立任务。

### 5.5 `_is_noop`（L290–L295）

```python
return isinstance(out, Mapping) and out.get(key) is None
```

节点返回 `{key: None}` 在很多图里表示"这次不改这个字段"，视作空更新。
**这里和真 LangGraph 有分歧** —— 真 LangGraph 会把 `None` 写进通道。minigraph
为了并行友好（避免 `{"debug": None}` 撞上另一个节点的 `{"debug": x}` 而误报
并发冲突）把它丢掉。代价：`{"x": None}` 无法表达"把 x 置空"。

### 5.6 `_wrap_send` / `_SendView`（L298–L308）

```python
def _wrap_send(send, values):
    if isinstance(send.arg, Mapping):
        return {**values, **send.arg, "__state__": values}   # arg 覆盖主状态
    return _SendView(send.arg, values)
```

- `Send.arg` 是 dict → 合并进主状态，且 **arg 同名 key 优先**（写在后面）。
  额外挂一个 `__state__` 键，让节点能只读访问未被子集覆盖的原始主状态
- `Send.arg` 是任意对象 → 包成 dict 子类，底层存主状态（节点仍可按 key 读状态），
  `arg` 挂在 `self.arg`

> `control.py` L47–L49 的注释说 `_SendView` 有"`__getattr__` 优先返回 arg 的属性"，
> 但实现里并没有定义 `__getattr__`（实测：`'_SendView' object has no attribute ...`）。
> 所以 `Send(worker, obj)` 这种写法拿不到对象属性，只能当纯 dict 用。注释与实现有出入。

### 5.7 `_merge_writes`（L363–L393）

```python
by_key = {}
for node, key, val in writes:
    by_key.setdefault(key, []).append((node, val))

unknown = set(by_key) - set(reducers)
if reducers and unknown:
    raise ValueError(f"节点写了未声明的状态字段 {sorted(unknown)}（写入者 {who}）；已声明 {sorted(reducers)}")

out = dict(base)
for key, items in by_key.items():
    reducer = reducers.get(key, LAST_VALUE)
    if len(items) > 1 and reducer is LAST_VALUE:
        raise RuntimeError(f"同一超步有多个节点（{who}）写字段 {key!r}；并发写必须给它配 reducer")
    acc = out.get(key, MISSING)
    for _, v in items:
        acc = reducer(acc, v)
    out[key] = acc
return out
```

三道校验/语义：

1. **字段声明检查**：`if reducers and unknown` —— 只有声明了 schema 才校验。
   没 schema（`StateGraph()`）时随便写，因为根本没东西可对
2. **并发写保护**：多节点写同字段 + 没显式 reducer → **报错**。
   理由写在 docstring 里：**"最后一次写入"在并行下是不确定的**。
   这不是给用户添堵，是在强迫你表态：你要的到底是覆盖、累加还是去重合并？
3. **reducer 是幂等恢复的基础**：`add_messages` 按 id 覆盖而非追加，
   `merge_dict` 浅合并，`operator.add` 累加 —— 节点被重跑时不会把结果算重

对比 `_apply`（L353–L360，被 `update_state` 用）：它**没有并发检查**，
因为 `update_state` 是"人工单次改写"，不存在两个写入者。

### 5.8 `_run_with_retry`（L396–L412）

```python
if retry is None:
    return fn(state)
policy = retry if isinstance(retry, Mapping) else {"max_attempts": int(retry)}
attempts = int(policy.get("max_attempts", 3))
backoff = float(policy.get("backoff", 0.0))
for i in range(attempts):
    try:
        return fn(state)
    except GraphInterrupt:
        raise                      # 挂起不重试
    except Exception as exc:
        last = exc
        if i + 1 < attempts and backoff:
            time.sleep(backoff * (2 ** i))     # 指数退避
raise RuntimeError(f"节点 {node!r} 重试 {attempts} 次仍失败：{last}") from last
```

`retry` 参数两种形态：整数（= max_attempts）或 dict
（`{"max_attempts": 3, "backoff": 0.5}`）。退避是 `backoff * 2**i`。

---

## 6. 完整追踪：01_react_agent.py

图结构（`examples/01_react_agent.py`）：

```
START → agent ──[有 tool_calls?]──→ tools → agent   （回边形成循环）
                   └──[无]────────→ END
```

输入：`{"messages": [{"id": "h1", "role": "human", "content": "北京"}], "steps": 0}`，
`config = {"thread_id": "t1", "debug": True}`。

**实际输出**（已验证）：

```
[start] {'messages': [{'id': 'h1', 'role': 'human', 'content': '北京'}], 'steps': 0}
  step 1: ['agent'] -> ['tools']
YIELD -> node=agent step=1 update_keys=['steps', 'messages']
  step 2: ['tools'] -> ['agent']
YIELD -> node=tools step=2 update_keys=['messages']
  step 3: ['agent'] -> []
YIELD -> node=agent step=3 update_keys=['steps', 'messages']
final steps = 2
ckpts = [(1, ('tools',), 'ckpt-0001-e01e'), (2, ('agent',), 'ckpt-0002-3ced'), (3, (), 'ckpt-0003-0934')]
```

逐步对应的代码路径：

| step | L143 tasks | L151 snapshot | L161 执行 | L188 合并 | L206 nxt | L217 ckpt |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `["agent"]` | `{messages:[h1], steps:0}` | `call_model` → 发 tool_calls | `messages` 追加 ai-0；`steps` 0→1 | 条件边 `should_continue` 返回 `"tools"` → `path_map` 翻译成 `"tools"` | `next=('tools',)` |
| 2 | `["tools"]` | 含 ai-0 | `tool_node` → 查天气 | `messages` 追加 tool-0-0；`steps` 不变 | 静态边 `tools→agent` | `next=('agent',)` |
| 3 | `["agent"]` | 含 tool 结果 | `call_model` → 直接作答 | `messages` 追加 ai-1；`steps` 1→2 | 条件边返回 `END` → 被过滤，`nxt=[]` | `next=()` |

`while tasks` 下一轮 `tasks=[]` → 循环结束。`ckpts` 有 3 份，最后一份 `next=()`
表示"跑完了"。**超步数 = 3，checkpoint 数 = 3，1:1 对应。**

> ⚠️ 注意：step 3 的 `ckpt-0003` 在 L217 保存，而 L212 的 yield 发生在它**之前**。
> 示例跑到底所以都保存了；如果消费者提前 break 就会少这份。

---

## 7. 中断续跑的完整追踪

对应 `examples/02_hitl_and_fanout.py` / `test_interrupt_inside_parallel_batch_rolls_back_the_superstep`。

```
超步 3:  [fetch_data ✓ 返回 {"raw": [...]}]
          [human_review → interrupt({"draft": ...})]
```

代码路径：

1. `_execute_batch` 启 2 线程。`fetch_data` 正常返回；
   `human_review` 抛 `GraphInterrupt`
2. `_run_task` 把它包成返回值 `gi`（L239–240）
3. L173 识别出 `GraphInterrupt` → `hit = gi`，`hit_slot = "human_review@3#1"`，`continue`
4. L187 的 `if writes and hit is None` → **`hit` 不是 None，`fetch_data` 的写入被丢弃**，
   `values` 保持进入 step 3 之前的样子
5. L191 构造 checkpoint：`values` 是**合并前**的，`tasks = list(batch)` = `[fetch, review]`，
   `interrupt` 记下 `slot/node/idx/step/value`
6. L202 `yield "human_review", 3, {}` → L203 `return`

恢复：

```python
app.invoke(None, {"thread_id": tid, "resume": {"action": "approve"}})
```

1. L99 载入带 `interrupt` 的最新 checkpoint
2. L103–L105 还原 `values` / `tasks` / `step`；L112 把 resume 值按**挂起槽位**塞进 ctx
3. L134–L138 还原 `pending_node="human_review"`、`pending_slot="human_review@3#1"`、
   `pending_idx=1`、`pending_step=3`
4. L147 `step = pending_step = 3`（**不自增**）→ 槽位重算结果与当初完全一致
5. L157–L158：`idx=1` 且 `name=="human_review"` 匹配 → 沿用 `pending_slot`
6. 整批重跑。`human_review` 第二次调 `interrupt()`，slot 命中 store（L39）→ 返回 resume 值
7. 正常返回 → 本步提交 → 继续往下走

**三条推论**：

- 为什么 checkpoint 是**超步级**不是节点级 —— 因为整批重跑，框架根本没能力
  只重跑其中一个
- 为什么节点必须**幂等** —— `fetch_data` 会跑第二次。它要是把数据 append 到某个
  外部列表、或调用了有副作用的 API，就会重复
- 为什么槽位要 `名@步@序号` 而不是 `名@序号` —— `step` 参与计算才能让同一节点在
  循环的不同轮次拿到不同槽位

---

## 8. 边界情况与已知限制

| 场景 | 当前行为 | 位置 |
| --- | --- | --- |
| 没设入口点 | `tasks=[]`，`_run` 立即返回空 `{}` | L124 |
| 没配 checkpointer 也没 thread_id | 只支持单次运行，不能 resume | L99 |
| `resume` 传 `None` / `False` | **可以**正常恢复（靠 `"resume" in cfg` 判断） | L95 |
| checkpoint 停在中断、没传 resume | `RuntimeError`，提示要传 `resume` | L107–L111 |
| 循环死转 | `recursion_limit` 兜底（默认 50） | L144 |
| 同一超步多节点中断 | 只记录**最后一个**的挂起信息；恢复后其他中断节点再次抛 `GraphInterrupt`，**需要多次 resume** | L177 |
| `Command(update={...})` 不带 `goto` | `routes=[]` → 节点后续的静态边/条件边被**整个跳过**，图在此终止（与 `control.py` 的注释不一致，见 §5.3） | L314–L321 |
| 消费者提前 `break` 掉 `stream` | 最后一份 checkpoint 不落盘（`yield` 在 `put` 之前） | L212 vs L217 |
| `Send` 嵌套浅拷贝 | `snapshot` 是浅拷贝，节点原地改嵌套 dict 会污染主状态 | L151 |
| 编译期检查 | `_validate` 只查"悬空节点"，不查不可达节点 / 必达 END | L30–L37 |

"同超步多中断需多次 resume" 这条值得展开：L112 只往 store 塞了**一个**槽位，
其他挂起节点重跑时 `interrupt()` 查不到自己的 slot（L38–L39）→ 再次抛
`GraphInterrupt` → L177 覆盖 `hit` → 又存一份 checkpoint → 又 `return`。
每一轮 resume 解开一个挂起点。这也符合 BSP 语义：**一次只解一个超步的屏障**。

---

## 9. 设计决策速查

| 决策 | 位置 | 理由 |
| --- | --- | --- |
| 循环条件是 `while tasks` 而非"跑到 END" | L143 | `END` 只是被过滤的普通字符串；任务耗尽才是唯一终止条件 |
| `snapshot = dict(values)` 放在循环内 | L151 | 每个超步一个新快照 → 跨超步才能读到上一步的写入 |
| 单任务不起线程池 | L257 | 绝大多数超步只有一个任务，省线程切换 |
| `[f.result() for f in futures]` | L268 | 保证顺序对齐 `runs`，不用 `as_completed` |
| 异常包装成返回值而非跨线程抛 | L239–L242 | 保留原始类型，主线程统一分类 |
| `GraphInterrupt` 排除出 retry | L407–408 | 挂起是控制流不是故障 |
| 挂起时 `values` 不合并 | L187 | BSP 全提交或全不提交 |
| 挂起时 `tasks` 存 `batch` 不存 `nxt` | L194 | 本步作废，重跑的就是 batch |
| `pending_step` 恢复时不 +1 | L147 | 槽位必须和当初逐字节一致 |
| 槽位含 `step` 和 `idx` | L158 | 同名节点在同超步可能多份，且循环里会反复出现 |
| `resume` 提前在主线程取出 | L255 | `threading.local` 不被子线程继承 |
| 无 reducer 的并发写直接报错 | L383–L388 | "最后一次写入"在并行下不确定 |
| `hit is None` 才合并 writes | L187 | 挂起则整步回滚 |
| input 不做字段校验 | L122 | 只读键（user_id 等）合法，只有"写"要声明 |
| noop（`{k: None}`）过滤在收集阶段 | L183 | 否则会误触发并发写冲突检查 |
| yield 在 checkpoint 之前 | L212 | 流式消费者先看到更新；代价是 break 会丢最后一份 |

---

## 附：一句判断标准

如果你能用一句话说清下面这个问题，就说明真读懂了这个循环：

> 为什么 `_merge_writes` 敢在**同超步内**把多个节点对同一字段的写入按
> `writes` 列表顺序依次 reduce，而不会因为"哪个节点先跑完"而出错？

答案：**因为所有节点读的是同一个 `snapshot`，谁也没看到谁的输出，
所以"顺序"只由 `_normalize` 的循环顺序（即 `runs` 顺序）决定，与线程完成
顺序无关。**reducer 只需要对"合并顺序"负责，不需要对"执行顺序"负责 ——
而后者在 BSP 下根本不存在。

对应测试：`test_parallel_execution_is_actually_concurrent`（并发是真的）、
`test_concurrent_write_without_reducer_is_rejected`（顺序无关的代价）、
`test_interrupt_inside_parallel_batch_rolls_back_the_superstep`（全不提交）。

---

## 附录：本文的实测记录

文中标注"已验证 / 实测"的几条，都是在本仓库 `.venv` 下跑出来的：

```bash
$ .venv/bin/python examples/01_react_agent.py            # §6 的完整输出
$ .venv/bin/python -c "... debug=True 跑同一张图 ..."     # §6 的 [start]/step 调试行
```

| 断言 | 实测结果 |
| --- | --- |
| §6 ReAct 图跑出 3 个超步、3 份 checkpoint | ✅ `ckpts = [(1,('tools',)), (2,('agent',)), (3,())]` |
| §5.3 `Command(update=...)` 不带 goto 会终止图 | ✅ `invoke({}) -> {'x': 'hi'}`，节点 `b` 未执行 |
| §5.6 `_SendView` 无 `__getattr__` | ✅ `AttributeError: '_SendView' object has no attribute ...` |
| §2.1 `resume=None` 可以正常恢复 | ✅ `invoke(None, {...,"resume": None})` 正常返回 |

未实测、纯代码推导的条目：`§3.10` 的 yield/put 顺序问题（依赖消费者提前 break）、
`§8` 的"同超步多中断需多次 resume"。这两条建议补一个回归测试再定论。