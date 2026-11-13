# `versions` 与 `seen` 详解：两张表什么时候被改

> 本文回答一个具体问题：`_RuntimeState` 里的 `versions` 和 `seen`
> **分别在什么时刻、被谁、改成什么值**。
>
> 这两张表是整个触发内核的全部机制：没有它们，"这个节点该不该醒"就无从判断。
> 配套阅读：《`runtime超步循环详解.md`》（超步骨架）、《`channel通道模型详解.md`》（通道语义）。

---

## 0. 一句话定位

```python
versions: dict[str, int]          # 通道 -> 版本号。全局单调递增，所有通道共用一个序列
seen:     dict[str, dict[str, int]]   # 节点 -> {通道: 该节点上次跑时看到的版本}
```

一句话说清分工：

| 表 | 键是什么 | 回答什么问题 | 谁写它 |
| --- | --- | --- | --- |
| `versions` | **通道** | "这个通道被改动过几次？" | 通道自己变了的时候（写/消费/自清） |
| `seen` | **节点** | "这个节点上次看到的是第几版？" | 节点跑完之后（推水位线） |

**触发判定 = 两者比较**（`runtime.py:786`）：

```python
channels[chan].is_available()                     # 通道当前有值
and versions.get(chan, 0) > seen.get(chan, 0)     # 版本比这个节点的水位线新
```

所以：

- `versions` 前进 = "有新动静了"
- `seen` 前进 = "我已经看过了"
- 两者一比，就是"还有我没看过的新动静吗"

**关键**：两张表**独立前进**，谁也不等谁。这是理解全部困惑的钥匙。

> **怎么读这篇文档**：
> §1–§3 是规则（谁在什么时候改哪张表），
> §4 是两个最小场景的完整时间线，
> §5 是**一条真实日志的逐行解读**（Map-Reduce 扇出）—— 第一次读可以直接跳到 §5。
> §6 是速查问答，§8 是一页表格。

---

## 1. `versions`：通道版本号

### 1.1 它是什么

一个整数。全图**共用一条递增序列**，不是每个通道各自计数：

```python
def _next_version(versions: dict[str, int]) -> int:
    """全局版本计数器：所有通道共用一个递增序列（对齐 LangGraph）。"""
    return max(versions.values(), default=0) + 1
```

（`runtime.py:841`）

含义：版本号**只在通道状态真的变了时**前进；一次超步里所有变化的通道**共享同一个**新版本号。

### 1.2 全部修改点

`versions` 一共只有 **5 个**地方会被改：

| # | 位置 | 时机 | 改什么 |
| --- | --- | --- | --- |
| ① | `runtime.py:464` / `405-406` | 运行初始化 / 恢复 | 全置 `0`，恢复时用 checkpoint 覆盖 |
| ② | `runtime.py:490` | 开局 `__start__` 写入口信号 | 所有入口通道 = 同一个 `start_version` |
| ③ | `runtime.py:879` | `_apply_writes`：本步的写入落地 | 被写且**状态真变了**的通道 = `next_version` |
| ④ | `runtime.py:176` | `consume_triggers`：屏障被消费 | 屏障通道 = `next_version` |
| ⑤ | `runtime.py:191` | `expire_unwritten`：没人写的通道自清 | 被清空的通道 = `next_version` |

下面逐条看。

#### ① 初始化 / 恢复

全新运行（`runtime.py:464`）：

```python
versions = {c: 0 for c in channels}
```

从 checkpoint 恢复（`runtime.py:405-406`）：

```python
versions = {c: 0 for c in channels}
versions.update(snap.channel_versions or {})
```

即：**先全填 0，再用 checkpoint 里的值覆盖**。checkpoint 里没有的通道（运行中才出现的）保持 0。

#### ② 开局：`__start__` 写入口信号（`runtime.py:485-492`）

```python
start_spec = self.nodes.get(START)
if start_spec:
    start_version = rt.next_version()
    for chan in start_spec.writers:          # branch:to:<每个入口点>
        rt.channels[chan].update([None])
        rt.versions[chan] = start_version
        rt.updated.add(chan)
```

一次赋值，**所有入口通道拿到同一个版本号**。这就是"多个 `add_edge(START, ...)` 的入口点能在第一个超步并行跑"的原因。

> 注意这里的 `update([None])`：分支通道 `EphemeralValue` 存的值是 `None`，
> 它的意义不是"值是什么"，而是"我被写过了"。

#### ③ 应用写入：`_apply_writes`（`runtime.py:846-882`）

```python
version = _next_version(versions) if next_version is None else next_version
for chan, items in by_chan.items():
    ch = channels.get(chan)
    vals = [v for _, v in items]
    changed = ch.update(vals)          # 通道自己判断"可用性变了吗"
    if changed:
        versions[chan] = version        # 只有真变了才推版本
        if ch.is_available():
            updated.add(chan)
```

两个要点：

1. **只有 `update()` 返回 `True` 才推版本。** 通道状态没真变（比如被 `update([])` 空写、或写了个一模一样的值）时，版本不动 —— 所以"写了一次但没变化"**不会**白白唤醒下游。
2. **本步所有变化通道共享同一个 `version`。** 这就是 `_advance` 里先算一次 `next_version` 再传下去的原因。

#### ④ 消费触发通道：`consume_triggers`（`runtime.py:160-178`）

```python
for task in tasks:
    if task.is_push: continue
    spec = self.nodes.get(task.name)
    for chan in spec.triggers:
        ch = self.channels.get(chan)
        if ch is not None and ch.consume():
            self.versions[chan] = next_version
            if ch.is_available():
                self.updated.add(chan)
```

全图只有 `NamedBarrierValue`（屏障通道 `join:a+b:c`）的 `consume()` 返回 `True`：

```python
def consume(self) -> bool:
    # 触发之后重新武装屏障，这样下一轮汇聚还能再等一次
    if self.seen == self.names:
        self.seen = set()
        return True
    return False
```

即：屏障凑齐 → 触发下游 → **清空屏障内的 seen 集合重新武装** → 通道版本推到 `next_version`。

为什么屏障要推版本？因为屏障是"多源汇聚"，同一个 `join:` 通道可能被不同轮次反复使用，不推版本的话第二轮的汇聚会因为 `versions > seen` 不成立而不触发。

> ⚠️ 这里的 `seen` 是 **`NamedBarrierValue` 通道自己的** `seen` 集合（记录"哪些源到了"），
> 和 `_RuntimeState.seen`（节点水位线）**完全是两回事**，只是同名。
> 读代码时注意区分 —— 本文说的是后者。

屏障的实测（`add_edge(["a","b"], "c")`）：

```text
  step 1 触发前
    versions: {..., 'branch:to:a': 1, 'branch:to:b': 1, 'branch:to:c': 0, 'join:a+b:c': 0}
    seen    : {'__start__': {'__start__': 0}}
    avail   : {'branch:to:a': None, 'branch:to:b': None}
  step 1: ['a', 'b'] -> changed=['x', 'join:a+b:c', 'x', 'join:a+b:c']
  step 2 触发前
    versions: {..., 'branch:to:a': 2, 'branch:to:b': 2, 'branch:to:c': 0, 'join:a+b:c': 2}
    seen    : {..., 'a': {'branch:to:a': 1}, 'b': {'branch:to:b': 1}}
    avail   : {'join:a+b:c': None}
  step 2: ['c'] -> changed=['x']
```

两个要点：

1. `join:a+b:c` 的版本 0 → 2，但 `avail` 里它的值是 `None` ——
   屏障通道"可用"意味着**所有源都到齐**（`is_available()` 返回 `self.seen == self.names`），
   不是"有个非 None 的值"。
2. `c` 被触发后，屏障 `consume()` 清空内部 seen 集合**重新武装**，
   于是下一轮 `a`、`b` 再写这个通道时，屏障又能凑齐、又能触发一次。

#### ⑤ 清理过期通道：`expire_unwritten`（`runtime.py:181-193`）

```python
for chan, ch in self.channels.items():
    if chan in written: continue
    if ch.is_available() and ch.update([]):     # 空写入 = 自清
        self.versions[chan] = next_version
        if ch.is_available():
            self.updated.add(chan)
```

`EphemeralValue.update([])` 会把自己抹掉（`channels.py:270-274`）：

```python
if len(values) == 0:
    # 空写入 = "本步没人写我" -> 自清
    if self.value is not MISSING:
        self.value = MISSING
        return True
    return False
```

**分支通道因此"恰好活一个超步"**。清空时也推版本，是为了让这个通道的版本**严格大于**任何节点的水位线 —— 水位线永远追不上一个已被清空的通道。

### 1.3 `versions` 的不变式

- **单调递增**：只会变大，从不变小（除非从 checkpoint 恢复，那是从旧值继续）。
- **同一个超步内的变化共享一个版本号。**
- **只在通道可用性真变时前进。**
- 通道被清空（自清 / 屏障消费）**也算**一次变化，照样推版本。

---

## 2. `seen`：每节点的水位线

### 2.1 它是什么

两层字典：外层键是**节点名**，内层键是该节点订阅的**触发通道**，值是"我上次跑的时候，这个通道是第几版"。

```python
seen = {
    "agent": {"branch:to:agent": 3},
    "tools": {"branch:to:tools": 4},
}
```

**每个节点各存一份** —— 这就是为什么循环能成立：`agent` 的水位线只管 `agent` 自己的触发通道。

### 2.2 三种存在形态（很重要）

| 形态 | 含义 | 触发行为 |
| --- | --- | --- |
| `seen.get(name) is None` | 这个节点**从没跑过** | 只要通道**可用**就触发（不看版本） |
| `seen[name] == {}` | 跑过，但是**通过 Send（PUSH）** 进来的 | 走正常版本比较（`seen.get(chan,0)` 默认 0，所以任何正版本都触发） |
| `seen[name] == {chan: v}` | 正常跑过，已有水位线 | `versions[chan] > v` 才触发 |

对应 `_triggers`（`runtime.py:765-788`）：

```python
def _triggers(channels, versions, seen, trig) -> bool:
    if seen is None:
        # 从没跑过 —— 此时只要通道可用就触发
        for chan in trig:
            ch = channels.get(chan)
            if ch is not None and ch.is_available():
                return True
        return False
    for chan in trig:
        ch = channels.get(chan)
        if ch is None or not ch.is_available():
            continue
        if versions.get(chan, 0) > seen.get(chan, 0):
            return True
    return False
```

### 2.3 全部修改点

`seen` 一共只有 **3 个**地方会被改：

| # | 位置 | 时机 | 改什么 |
| --- | --- | --- | --- |
| ① | `runtime.py:477` / `408` | 运行初始化 / 恢复 | 全新 = `{}`；恢复 = checkpoint 里的 `versions_seen` |
| ② | `runtime.py:492` | 开局 `__start__` | `seen[START] = {"__start__": 该通道版本}` |
| ③ | `runtime.py:135-157` | **`advance_watermarks`：本步跑完的节点** | 把它看过的触发通道版本记进去 |

**全代码里 `advance_watermarks` 只有一个调用点**（`runtime.py:640`，在 `_advance` 里）——
也就是说，**只有"节点真的跑了"才会推进水位线**，别无其它来源。

`update_state` **不会**推进任何水位线：人工改状态时没有节点"跑过"，
`seen` 原封不动地从 checkpoint 恢复（`runtime.py:309`）。

### 2.4 ② 开局：`__start__` 给自己记水位线（`runtime.py:492`）

```python
rt.seen[START] = {start_spec.triggers[0]: rt.versions[start_spec.triggers[0]]}
```

`__start__` 的触发通道是保留通道 `__start__` 自己。记上水位线，它就不会重复触发。

### 2.5 ③ 核心：`advance_watermarks`（`runtime.py:135-157`）

```python
def advance_watermarks(self, tasks: list[Task]) -> None:
    """
    本步跑过的节点，把水位线推到**它当时看到的版本**。

    注意顺序：必须用**写入前**的版本，而不是写入后的版本。
    否则节点自己写 branch:to:自己 就会把自己的触发信号抹掉，
    循环（回边）就再也转不起来。
    """
    for task in tasks:
        # PUSH 的触发源是保留通道 __pregel_tasks，不是目标节点的
        # branch:to:X；因此它不能推进 X 的水位线（对齐真实 LangGraph）。
        if task.is_push:
            self.seen.setdefault(task.name, {})
            continue
        spec = self.nodes.get(task.name)
        if spec is None or not spec.triggers:
            continue
        node_seen = self.seen.setdefault(task.name, {})
        for chan in spec.triggers:
            if chan in self.versions:
                node_seen[chan] = self.versions[chan]
```

逐条读：

1. **只处理本步实际跑过的任务**（`tasks` 参数就是本步的任务清单）。
2. **PUSH（Send）任务不推进水位线。** 只 `setdefault(name, {})` 建个空条目。
   理由：Send 的触发源是保留通道 `__pregel_tasks`，不是目标节点的 `branch:to:X`，
   推进 `branch:to:X` 的水位线等于凭空抹掉一个还没发生过的信号。真实 LangGraph 同样如此。
3. **只记该节点自己订阅的通道**（`spec.triggers`）。`seen[node]` 永远只含它自己的触发通道。
4. **记的是"它当时看到的版本"** —— 也就是**写入前**的值。

#### 为什么必须是"写入前"

这是最容易踩坑的地方。`_advance`（`runtime.py:628-647`）的顺序是：

```python
rt.advance_watermarks(tasks)          # 1. 先记水位线（此时版本还是旧值）
next_version = rt.next_version()
rt.updated = set()
rt.consume_triggers(tasks, next_version)   # 2. 消费屏障
rt.updated |= _apply_writes(..., next_version)  # 3. 后应用写入、推版本
rt.expire_unwritten(written, next_version)      # 4. 清旧通道
```

**步骤 1 必须在步骤 3 之前**。因为步骤 3 里，节点自己的写入会把 `branch:to:<自己>` 的版本推进到 `next_version`。如果步骤 1 在步骤 3 之后，水位线就会记成 `next_version`，而版本也是 `next_version` —— `versions > seen` 不成立，**节点永远不会再被自己唤醒，回边循环就死了**。

以回边 `agent -> tools -> agent` 为例（实测数字见 §4.1）：

```text
step 1: agent 跑，写 branch:to:tools
        → advance_watermarks: seen["agent"]["branch:to:agent"] = 1   ← 写入前的版本 1
        → _apply_writes:       versions["branch:to:tools"]  = 2

step 2: tools 跑，写 branch:to:agent
        → advance_watermarks: seen["tools"]["branch:to:tools"] = 2
        → _apply_writes:       versions["branch:to:agent"]  = 3

step 3: 算触发
        agent: versions["branch:to:agent"]=3 > seen["agent"]["branch:to:agent"]=1  ✓ 触发
```

如果 `advance_watermarks` 排在 `_apply_writes` 之后，step 1 里
`seen["agent"]["branch:to:agent"]` 会被记成 2（写入后的值），
step 3 就变成 `versions["branch:to:agent"]`（3）`>` `seen`（2）—— 表面上还能触发，
但**水位线从此紧跟版本**，每一步的"落后一格"就没了：
真正致命的是自环（agent 写 `branch:to:agent` 自己），
那时 `versions == seen`，`>` 不成立，循环立刻停转。
**这就是那条注释在说的事。**

### 2.6 `update_state` 路径（`runtime.py:339-344`）

```python
rt = _RuntimeState(
    nodes=self.nodes, reducers=self.reducers, channels=channels,
    versions=versions, seen=seen, extra=extra_values,
    updated=set(updated), step=snap.step,
)
rt.expire_unwritten({c for c, _ in writes}, next_version)
```

`update_state` 里：

- `versions` **会被改**（`_apply_writes` 推版本，见 `runtime.py:332`）；
- `seen` **完全不动** —— 它只是从 checkpoint 恢复过来（`runtime.py:309`）。

也就是说，人工改完状态之后，节点的水位线仍是"上一次真正跑过"时的那条线，
触发判定照常按它算。

---

## 3. `seen` 什么时候**不**被改

这三处的"不动"同样是语义的一部分：

### 3.1 中断挂起：整步回滚（`runtime.py:650-689`）

有节点调 `interrupt()` 时：

```python
if hits:
    self._suspend(rt, runs, hits, store_before, saver, thread_id)
    yield hits[0].task.name, rt.step + 1, {}
    return
```

`_suspend` 在 `_advance` **之前**返回，也就是说：

- 写入一个都不应用
- `versions` 一个都不推
- `seen` 一个都不动

于是恢复时 `_next_tasks` 算出**完全相同的一批任务**，整批重跑。这就是 BSP 屏障
"一步要么全提交，要么全不提交"的含义，也是 checkpoint 必须存在**超步级**的根本原因。

实测（`a` 正常、`b` 里 `interrupt()`）：

```text
  step 1 触发前
    versions: {..., 'branch:to:a': 1, 'branch:to:b': 0}
    seen    : {'__start__': {'__start__': 0}}
  step 1: ['a'] -> changed=['x', 'branch:to:b']
  step 2 触发前
    versions: {..., 'branch:to:a': 2, 'branch:to:b': 2}
    seen    : {..., 'a': {'branch:to:a': 1}}
  step 2: b 挂起
--- checkpoint after interrupt ---
channel_versions: {'x': 2, 'branch:to:a': 2, 'branch:to:b': 2, '__start__': 0, '__pregel_tasks': 0}
versions_seen   : {'__start__': {'__start__': 0}, 'a': {'branch:to:a': 1}}
step: 2
```

注意 checkpoint 里的状态**停在 step 1 结束时的样子**：

- `seen` 里**没有 `b`** —— b 跑过（挂起也算跑），但因为整步回滚，水位线没记；
- `versions` 是 step 1 结算后的值，step 2 的任何推进都没发生。

恢复时重新算触发，还是得出 `['b']` 这同一批任务，b 重跑。

### 3.2 `consume_triggers` / `expire_unwritten` 不碰 `seen`

这两个方法只改 `versions`。它们改的是**通道侧**的状态（屏障重武装、ephemeral 自清），
和节点水位线是两回事。

### 3.3 `update_state` 不推水位线

人工改写状态时，不会有节点"跑过"，所以没有节点的水位线该被推进。
`seen` 原封不动地从 checkpoint 恢复过来。

---

## 4. 两张表怎么配合：完整时间线

用最简单的链 `a -> b` 走一遍（`a` 是入口）：

```text
【开局 _begin】
  versions = {branch:to:a: 0, branch:to:b: 0, ...}
  seen     = {}
  __start__ 写 branch:to:a
    → versions["branch:to:a"] = 1
    → seen["__start__"] = {"__start__": 0}

【step 1】算触发
  a: seen.get("a") is None → 从没跑过 → branch:to:a 可用 → ✓ 触发
  b: seen.get("b") is None → branch:to:b 不可用 → ✗

【step 1】跑 a，a 写 branch:to:b
  advance_watermarks:  seen["a"] = {"branch:to:a": 1}      ← 写入前的版本 1
  consume_triggers:    a 的触发通道 branch:to:a 是 ephemeral，consume() 返回 False
  _apply_writes:       versions["branch:to:b"] = 2
  expire_unwritten:    branch:to:a 本步没人写 → update([]) 自清
                       → versions["branch:to:a"] = 2
  updated = {"branch:to:b"}

【step 2】算触发
  a: seen["a"]={"branch:to:a":1}，versions=2 > 1 ✓？——但 branch:to:a 已被自清，
     is_available() 为 False → ✗ 不触发    ← 水位线和自清一起保证"一个超步只触发一次"
  b: seen.get("b") is None → 从没跑过 → branch:to:b 可用 → ✓ 触发

【step 2】跑 b
  advance_watermarks:  seen["b"] = {"branch:to:b": 2}
  _apply_writes:       无
  expire_unwritten:    branch:to:b 自清 → versions["branch:to:b"] = 3

【step 3】算触发
  a: branch:to:a 已清空 → ✗
  b: branch:to:b 已清空 → ✗
  → 没有任务，图结束
```

> 这段时间线**取自真实运行输出**（`config={"debug": True}`）。实际打印：
>
> ```text
>   step 1 触发前
>     versions: {'__pregel_tasks': 0, '__start__': 0, 'branch:to:a': 1, 'branch:to:b': 0}
>     seen    : {'__start__': {'__start__': 0}}
>     avail   : {'branch:to:a': None}
>   step 1: ['a'] -> changed=['x', 'branch:to:b']
>   step 2 触发前
>     versions: {'__pregel_tasks': 0, '__start__': 0, 'branch:to:a': 2, 'branch:to:b': 2}
>     seen    : {'__start__': {'__start__': 0}, 'a': {'branch:to:a': 1}}
>     avail   : {'branch:to:b': None}
>   step 2: ['b'] -> changed=['x']
> ```
>
> 注意 step 2 的 `branch:to:a` 版本已经是 **2**（自清时推的），
> 但它已被清空所以 `avail` 里没有它 —— 版本推了，值没了，于是不再触发。

注意 step 2 里 `a` 的判定：版本确实比水位线新（2 > 1），但**通道不可用**，
所以没触发。**两个条件是"与"的关系，缺一不可** —— 这是最容易被忽略的一点。

### 4.1 循环（回边）长什么样

再看回边 `agent -> tools -> agent`（实测输出）：

```text
  step 1 触发前
    versions: {..., 'branch:to:agent': 1, 'branch:to:tools': 0}
    seen    : {'__start__': {'__start__': 0}}
  step 1: ['agent'] -> changed=['x', 'branch:to:tools']
  step 2 触发前
    versions: {..., 'branch:to:agent': 2, 'branch:to:tools': 2}
    seen    : {..., 'agent': {'branch:to:agent': 1}}
  step 2: ['tools'] -> changed=['x', 'branch:to:agent']
  step 3 触发前
    versions: {..., 'branch:to:agent': 3, 'branch:to:tools': 3}
    seen    : {..., 'agent': {'branch:to:agent': 1}, 'tools': {'branch:to:tools': 2}}
    avail   : {'branch:to:agent': None}
  step 3: ['agent'] -> ...
```

step 3 是关键：`versions["branch:to:agent"] = 3`，
但 `seen["agent"]["branch:to:agent"]` 还停在 **1**（step 1 记的写入前版本），
`3 > 1` → agent 再次被唤醒。

**agent 的水位线总是比通道版本落后一个超步** —— 这个"落后一格"就是循环的燃料。
如果 `advance_watermarks` 记的是写入后的版本（3），水位线会追平版本，
`3 > 3` 不成立，循环立刻停转。

---

## 5. 完整实例：Map-Reduce 扇出逐行解读

这一节把前面所有规则用**一条真实日志**串起来。
图来自 `examples/02_hitl_and_fanout.py` 的 B 段（Map-Reduce 扇出）。

### 5.1 图和编译产物

```text
__start__ → split → dispatch → research_one ×3 → report → END
```

编译成通道和订阅关系：

| 节点 | triggers（订阅，能被谁唤醒） | writers（写） |
| --- | --- | --- |
| `split` | `branch:to:split` | `branch:to:dispatch` |
| `dispatch` | `branch:to:dispatch` | `__pregel_tasks`（Send 队列） |
| `research_one` | `branch:to:research_one` | `branch:to:report` |
| `report` | `branch:to:report` | —（出边是 END，不需要通道） |

> ⚠️ `research_one` 虽然订阅了 `branch:to:research_one`，但那条通道**从头到尾没人写**（版本一直是 0）——
> 因为 `dispatch` 写的是 `__pregel_tasks`（Send），不是 `branch:to:research_one`。
> 静态边 `add_edge("dispatch", "research_one")` 在这里只是**为了让图不悬空**，
> 真正唤醒 `research_one` 的是 Send。

节点代码（`examples/02_hitl_and_fanout.py:124-142`）：

```python
def split(state):
    return {"subtasks": ["市场规模", "竞争格局", "监管风险"]}

def dispatch(state):
    """返回 list[Send]：每个子任务一份独立输入，下一超步并行执行。"""
    return [Send("research_one", {"question": state["question"], "aspect": t})
            for t in state["subtasks"]]

def research_one(state):
    time.sleep(0.05)
    return {"findings": [f"[{state['aspect']}] 关于「{state['question']}」的调研结论"]}

def reduce_report(state):
    body = "\n".join(f"- {f}" for f in sorted(state["findings"]))
    return {"report": f"# {state['question']}\n\n{body}"}
```

### 5.2 完整日志

```text
step 1 触发前
  versions: {'__pregel_tasks': 0, '__start__': 0, 'branch:to:dispatch': 0, 'branch:to:report': 0, 'branch:to:research_one': 0, 'branch:to:split': 1}
  seen    : {'__start__': {'__start__': 0}}
  avail   : {'branch:to:split': None}
step 1: ['split'] -> changed=['subtasks', 'branch:to:dispatch']
step 2 触发前
  versions: {'__pregel_tasks': 0, '__start__': 0, 'branch:to:dispatch': 2, 'branch:to:report': 0, 'branch:to:research_one': 0, 'branch:to:split': 2}
  seen    : {'__start__': {'__start__': 0}, 'split': {'branch:to:split': 1}}
  avail   : {'branch:to:dispatch': None}
step 2: ['dispatch'] -> changed=['__pregel_tasks', '__pregel_tasks', '__pregel_tasks']
step 3 触发前
  versions: {'__pregel_tasks': 3, '__start__': 0, 'branch:to:dispatch': 3, 'branch:to:report': 0, 'branch:to:research_one': 0, 'branch:to:split': 2}
  seen    : {'__start__': {'__start__': 0}, 'dispatch': {'branch:to:dispatch': 2}, 'split': {'branch:to:split': 1}}
  avail   : {'__pregel_tasks': [Send(node='research_one', arg={'question': '固态电池产业', 'aspect': '市场规模'}), Send(node='research_one', arg={'question': '固态电池产业', 'aspect': '竞争格局'}), Send(node='research_one', arg={'question': '固态电池产业', 'aspect': '监管风险'})]}
step 3: ['research_one', 'research_one', 'research_one'] -> changed=['findings', 'branch:to:report', 'findings', 'branch:to:report', 'findings', 'branch:to:report']
step 4 触发前
  versions: {'__pregel_tasks': 4, '__start__': 0, 'branch:to:dispatch': 3, 'branch:to:report': 4, 'branch:to:research_one': 0, 'branch:to:split': 2}
  seen    : {'__start__': {'__start__': 0}, 'dispatch': {'branch:to:dispatch': 2}, 'research_one': {}, 'split': {'branch:to:split': 1}}
  avail   : {'branch:to:report': None}
step 4: ['report'] -> changed=['report']
```

### 5.3 逐超步拆解

#### Step 1 触发前 —— 开局

```text
versions: {..., 'branch:to:split': 1, 'branch:to:dispatch': 0, ...}
seen    : {'__start__': {'__start__': 0}}
avail   : {'branch:to:split': None}
```

开局状态（`_begin`，`runtime.py:463-493`）：

- 所有通道版本 = 0。
- `__start__` 的唯一工作是"点亮所有入口点"。唯一入口是 `split`，
  所以它写 `branch:to:split`，版本 → **1**。
- `seen["__start__"] = {"__start__": 0}` —— `__start__` 记下自己看过
  `__start__` 通道的第 0 版，免得重复触发。
- `avail` 里只有 `branch:to:split`。`EphemeralValue` 存的值是 `None`，
  但**被写过就算"可用"** —— 值是什么不重要，重要的是"被写过"。

**算触发**：`split` 的 `seen.get("split")` 是 `None` → 从没跑过 → 通道可用 → ✓ 触发。

#### Step 1 —— `split` 跑

```text
step 1: ['split'] -> changed=['subtasks', 'branch:to:dispatch']
```

`split` 返回 `{"subtasks": [...]}`；它的出边让它再写 `branch:to:dispatch`。

落盘四步：

| 步 | 动作 | 结果 |
| --- | --- | --- |
| ① `advance_watermarks` | `split` 记水位线 | `seen["split"] = {"branch:to:split": 1}` ← **写入前**的版本 |
| ② `next_version` | | 2 |
| ③ `consume_triggers` | `branch:to:split` 是 ephemeral，`consume()` 返回 False | 版本不动 |
| ④ `_apply_writes` | `subtasks`、`branch:to:dispatch` 变了 | `versions["branch:to:dispatch"] = 2` |
| ⑤ `expire_unwritten` | `branch:to:split` 本步没人写 → `update([])` 自清 | `versions["branch:to:split"] = 2` |

#### Step 2 触发前

```text
versions: {..., 'branch:to:dispatch': 2, 'branch:to:split': 2}
seen    : {..., 'split': {'branch:to:split': 1}}
avail   : {'branch:to:dispatch': None}
```

**两个关键点：**

**① `split` 没有被再次触发。** 它的触发通道 `branch:to:split`：
- 版本 2 > 水位线 1 ✓ 版本是新的
- 但通道**已被自清**，`is_available()` 为 False ✗

**两个条件是"与"的关系，缺一不可** —— 版本推进了不代表会触发。

**② `branch:to:split` 的版本是 2 而不是 1。** 尽管它从 step 2 起再没被写过，
但自清也算一次变化，照样推版本。这样任何节点的水位线都追不上一个已被清空的通道。

**算触发**：`dispatch` 从没跑过 → `branch:to:dispatch` 可用 → ✓ 触发。

#### Step 2 —— `dispatch` 跑（Send 登场）

```text
step 2: ['dispatch'] -> changed=['__pregel_tasks', '__pregel_tasks', '__pregel_tasks']
```

`dispatch` 返回 `list[Send]`，三个 Send 包被写进保留通道 `__pregel_tasks`。

`changed` 里 `__pregel_tasks` 出现**三次** —— 三个 Send 各算一次写入。

落盘：

| 步 | 动作 | 结果 |
| --- | --- | --- |
| ① | `dispatch` 记水位线 | `seen["dispatch"] = {"branch:to:dispatch": 2}` |
| ④ | 写 `__pregel_tasks` ×3 | `versions["__pregel_tasks"] = 3` |
| ⑤ | `branch:to:dispatch` 没人写 → 自清 | `versions["branch:to:dispatch"] = 3` |

**注意 `branch:to:research_one` 的版本一直是 0** —— Send 走 `__pregel_tasks`，
**完全绕开了分支通道**。

#### Step 3 触发前 —— PUSH 轨道

```text
versions: {..., '__pregel_tasks': 3, 'branch:to:research_one': 0}
avail   : {'__pregel_tasks': [Send(...), Send(...), Send(...)]}
```

`avail` 里 `__pregel_tasks` 的值是三个 Send 包 —— 这是 `Topic` 通道，存的是列表。

**触发是怎么发生的**（`_prepare_next_tasks`，`runtime.py:808-816`）：

```python
tc = channels.get(TASKS)
if tc is not None and tc.is_available():
    for idx, packet in enumerate(tc.get()):
        if isinstance(packet, Send):
            tasks.append(Task(name=packet.node, send=packet, idx=idx))
```

**Send 任务根本不走版本判定** —— 只要 `__pregel_tasks` 里有 Send 包，就变成任务。
所以 `research_one` 被触发了，尽管 `branch:to:research_one` 的版本是 0（从没被写过）。

反过来说，`research_one` 在 PULL 轨上也不会被选中：
它只订阅 `branch:to:research_one`，那通道从没被写过（版本 0），
`seen` 里也没有它 —— 版本判定不成立。

#### Step 3 —— `research_one` ×3 并行跑

```text
step 3: ['research_one', 'research_one', 'research_one']
  -> changed=['findings', 'branch:to:report', 'findings', 'branch:to:report', 'findings', 'branch:to:report']
```

三个 `research_one` 在同一超步内**并发**执行（`ThreadPoolExecutor`，`runtime.py:972`），
每个都写 `findings` 和 `branch:to:report`。

并发写不冲突，因为：
- `findings` 配了 `operator.add` → `BinaryOperatorAggregate`；
- `branch:to:report` 是 `EphemeralValue(guard=False)` —— guard=False 允许一步收多个值，
  因为"两个并行节点同时 `Command(goto='x')`"是合法的。

**`seen` 的关键变化：**

```python
if task.is_push:
    self.seen.setdefault(task.name, {})   # 建个空条目，但不推进水位线
    continue
```

```text
seen: {..., 'research_one': {}, ...}
```

**`seen["research_one"] = {}` 是空字典！** PUSH 任务**不推进**目标节点的水位线 ——
Send 的触发源是保留通道 `__pregel_tasks`，不是目标节点的 `branch:to:research_one`；
推进 `branch:to:research_one` 的水位线等于凭空抹掉一个还没发生过的信号。
（真实 LangGraph 同样如此。）

落盘：

| 步 | 动作 | 结果 |
| --- | --- | --- |
| ① | 三个 PUSH 任务 → `setdefault` 空条目 | `seen["research_one"] = {}` |
| ④ | `findings`、`branch:to:report` 被写 | `versions["branch:to:report"] = 4` |
| ⑤ | `branch:to:dispatch` 没人写 → 自清 | `versions["branch:to:dispatch"] = 4` |

**`branch:to:report` 被写了 3 次，但只推一个版本号** ——
同一超步内所有变化通道共享同一个 `next_version`。

#### Step 4 触发前 —— `report` 该醒了

```text
versions: {..., 'branch:to:report': 4, 'branch:to:research_one': 0}
seen    : {..., 'research_one': {}, ...}
avail   : {'branch:to:report': None}
```

`report` 被触发：它从没跑过（`seen.get("report")` 是 `None`），
`branch:to:report` 可用 → ✓

#### Step 4 —— `report` 跑，图结束

```text
step 4: ['report'] -> changed=['report']
```

`report` 写状态字段 `report`（数据通道），**不写任何分支通道**（出边是 END）。

落盘：

| 步 | 动作 | 结果 |
| --- | --- | --- |
| ① | `report` 记水位线 | `seen["report"] = {"branch:to:report": 4}` |
| ④ | 写数据通道 `report` | `versions["report"] = 5` |
| ⑤ | `branch:to:report` 没人写 → 自清 | `versions["branch:to:report"] = 5` |

数据通道的版本也会推，但它**从不参与触发判定** —— 只有订阅它的节点才看版本，
而 `report` 字段没有节点订阅它。

所有分支通道都自清了，`avail` 为空 → 没有任务 → 图结束。

### 5.4 一张表看完整条链

| step | 跑的节点 | `versions` 变化 | `seen` 变化 |
| --- | --- | --- | --- |
| 开局 | — | `branch:to:split`: 0→1 | `seen["__start__"] = {"__start__": 0}` |
| 1 | `split` | `branch:to:dispatch`: 0→2<br>`branch:to:split`: 1→2（自清） | `seen["split"] = {"branch:to:split": 1}` |
| 2 | `dispatch` | `__pregel_tasks`: 0→3<br>`branch:to:dispatch`: 2→3（自清） | `seen["dispatch"] = {"branch:to:dispatch": 2}` |
| 3 | `research_one`×3 | `branch:to:report`: 0→4<br>`branch:to:dispatch`: 3→4（自清） | `seen["research_one"] = {}` ← **空** |
| 4 | `report` | `report`: 0→5<br>`branch:to:report`: 4→5（自清） | `seen["report"] = {"branch:to:report": 4}` |

### 5.5 这条日志里三个最容易看错的点

**① 版本新 ≠ 会触发。** step 2 里 `split` 的通道版本 2 > 水位线 1，
但通道已被自清，所以不触发。可用性和版本是**与**的关系。

**② `research_one` 的 `seen` 是空的 `{}`。** 不是 `None`，
也不是 `{"branch:to:research_one": 0}`。PUSH 不推进水位线。
这个 `{}` 意味着它走的是版本比较分支而不是"从没跑过"分支 ——
对它没有实际影响（它是被 `__pregel_tasks` 唤醒的），
但如果它同时还有一条静态入边，行为就会不一样。

**③ `branch:to:research_one` 全程是 0。** Send 完全绕开分支通道。

---

## 6. 常见困惑速查

### Q: `versions` 和 `seen` 哪个先变？

看**同一个超步内**的顺序：

```text
advance_watermarks (改 seen)
    ↓
consume_triggers (改 versions)
    ↓
_apply_writes (改 versions)
    ↓
expire_unwritten (改 versions)
```

**`seen` 在本步开头改（用旧版本），`versions` 在本步后段改（用新版本）。**
所以在一次 `_advance` 结束后，通常 `versions[chan] >= seen[node][chan]`。

### Q: 为什么水位线不会"超前"于版本？

因为 `advance_watermarks` 读的是**写入前**的 `self.versions[chan]`，
而写完以后 `versions[chan]` 只会更大。所以 `seen` 永远 `<=` 当时的版本。

唯一的例外是 **PUSH（Send）**：`seen[target]` 是空字典 `{}`，
`seen.get(chan, 0)` 默认取 0，相当于"水位线在 0"。

### Q: 节点跑完了但没被再次触发，为什么？

三种可能，按顺序排查：

1. **通道不可用**了 —— 分支通道一个超步就自清（`expire_unwritten`）。
2. **版本没前进** —— 本步没有节点写这个通道，或写了但通道状态没真变（`_apply_writes` 里 `changed` 为 False）。
3. **水位线追上了** —— `seen[node][chan] == versions[chan]`。

### Q: 节点没跑过就被触发了吗？

会。`seen.get(name) is None` 时只看**可用性**，不看版本 —— 入口节点就是靠这个被点亮的。

### Q: Send 进来的节点，`seen` 长什么样？

`{"node": {}}` —— 空字典，**不是 None**。所以它走的是版本比较分支而不是"从没跑过"分支。
这是 `advance_watermarks` 里 `if task.is_push: setdefault(name, {}); continue` 的直接结果。

实测（`fan` 返回两个 `Send("worker", ...)`）：

```text
  step 2 触发前
    versions: {..., '__pregel_tasks': 2, 'branch:to:fan': 2, 'branch:to:worker': 0}
    seen    : {'__start__': {...}, 'fan': {'branch:to:fan': 1}}
    avail   : {'__pregel_tasks': [Send(node='worker', arg={'x': 'w1'}), Send(node='worker', arg={'x': 'w2'})]}
  step 2: ['worker', 'worker'] -> changed=['x', 'branch:to:fan', 'x', 'branch:to:fan']
  step 3 触发前
    versions: {..., '__pregel_tasks': 3, 'branch:to:fan': 3, 'branch:to:worker': 0}
    seen    : {'__start__': {...}, 'fan': {'branch:to:fan': 1}, 'worker': {}}
    avail   : {'branch:to:fan': None}
```

三个观察：

1. **`seen["worker"] = {}` 是空的** —— PUSH 不推进目标节点的水位线，
   而且**永远不会**推进（每轮 PUSH 跑完还是 `{}`）。
2. **`branch:to:worker` 的版本始终是 0** —— PUSH 轨道完全绕开了分支通道。
   worker 是靠 `__pregel_tasks` 里的 Send 包被唤醒的，不是靠 `branch:to:worker`。
3. `worker` 跑完写的 `branch:to:fan` 却**正常推版本**（→ 3）——
   PUSH 只影响"自己怎么被唤醒"，不影响"自己跑完写了什么"。

### Q: `seen[node]` 里的键会变多吗？

会。节点每多订阅一个通道，`seen[node]` 就多一个键。实践中：

- 普通节点：`seen["b"] = {"branch:to:b": v}`
- 被屏障触发的节点：`seen["c"] = {"branch:to:c": v, "join:a+b:c": v}` —— 屏障通道也在 `spec.triggers` 里
- Send 进来的节点：`seen["w"] = {}`

实测（`add_edge(["a","b"], "c")`）：

```text
triggers of c: ['branch:to:c', 'join:a+b:c']
advance_watermarks tasks: [('a', False), ('b', False)]
  -> seen: {'__start__': {...}, 'a': {'branch:to:a': 1}, 'b': {'branch:to:b': 1}}
advance_watermarks tasks: [('c', False)]
  -> seen: {..., 'c': {'branch:to:c': 0, 'join:a+b:c': 2}}
```

注意 `seen["c"]["branch:to:c"] = 0` —— `branch:to:c` 从没被写过，
版本一直是 0，所以水位线记 0。**水位线记的是"我看到的版本"，
不是"我看了几次"** —— 通道从没变过，记 0 是对的。

> ⚠️ 观察时机：`debug` 输出打印的是**该超步执行前**的快照。
> 所以 `c` 的水位线要等到**下一步**的 dump 才看得见。
> 想看某节点跑完后的水位线，看它在**下一个超步**的 dump 里。

### Q: 版本号会溢出或者重复吗？

不会重复：同一超步内所有变化通道共享同一个新版本号，
不同超步的版本号严格递增（`_next_version` 取的是全局 max）。

### Q: `versions` 和 `seen` 存在 checkpoint 里吗？

存。`_save`（`runtime.py:692-703`）和 `_suspend`（`runtime.py:671-684`）都写：

```python
channel_versions=dict(rt.versions),
versions_seen={k: dict(v) for k, v in rt.seen.items()},
```

恢复时由 `_restore_channels`（`runtime.py:405-408`）原样读回。

### Q: `update_state` 之后这两张表怎么变？

- `versions`：走 `_apply_writes`（`runtime.py:332`），被写的通道推版本；随后 `expire_unwritten`（`runtime.py:344`）清掉没写的触发通道并推版本。
- `seen`：**不动**（从 checkpoint 原样恢复，`runtime.py:309`）。

---

## 7. 调试

`config={"debug": True}` 会打印每一步的三张表（`runtime.py:195-210`）：

```python
def dump(self, tag: str) -> None:
    trig = {c: ch for c, ch in self.channels.items() if c not in self.reducers}
    print(f"  {tag}")
    print("    versions:", {k: v for k, v in sorted(self.versions.items())
                          if k not in self.reducers})
    print("    seen    :", {k: dict(sorted(v.items())) for k, v in sorted(self.seen.items())})
    print("    avail   : {c: trig[c].get() for c in sorted(trig)
                           if trig[c].is_available()})
```

只打**触发通道**（`branch:to:*` / `join:*` / `__pregel_tasks`）——
数据通道一旦被写就始终可用，从不参与触发判定，混进来只会淹没真正要看的信息。

看的时候对着这张表读：

```text
versions[chan]  = V     ← 通道现在第几版
seen[node][chan] = S   ← 这个节点上次看到第几版
avail[chan]     = ?    ← 通道当前有没有值

触发 = avail[chan] AND V > S
```

> ⚠️ `dump` 打印的是**该超步执行前**的快照。所以某个节点跑完后的水位线，
> 要在**下一个超步**的 dump 里才看得到。

本文所有时间线都可以跑出来复现：

```bash
.venv/bin/python scripts/versions_seen_trace.py
```

五个场景（线性链 / 回边 / 屏障 / Send 扇出 / Map-Reduce 扇出）
+ 三条不变式断言（`versions` 单调、`seen <= versions`、循环里水位线落后一格）。

§5 的 Map-Reduce 完整日志来自 `examples/02_hitl_and_fanout.py` 的 B 段，
场景 5 就是它的一个可独立运行的复刻。

---

## 8. 一页速查表

| 事件 | `versions` | `seen` |
| --- | --- | --- |
| 初始化（全新） | 全部 = 0 | = `{}` |
| 初始化（恢复） | checkpoint 值覆盖默认 0 | checkpoint 值原样恢复 |
| `__start__` 写入口 | 入口通道 = `start_version` | `seen[__start__]` = 该通道版本 |
| 节点跑完（正常） | — | 该节点的触发通道 = **写入前**的版本 |
| 节点跑完（Send 进来） | — | `seen[node] = {}`（空） |
| 本步有写入 | 被写且状态真变的通道 = `next_version` | — |
| 屏障被消费 | 屏障通道 = `next_version` | — |
| 没人写的通道自清 | 被清空的通道 = `next_version` | — |
| 有节点 interrupt | **不动**（整步回滚） | **不动**（整步回滚） |
| `update_state` | 被写的通道 = `next_version` | **不动** |

---

## 9. 对照真实 LangGraph

| minigraph | LangGraph |
| --- | --- |
| `versions[chan]` | `checkpoint["channel_versions"][chan]` |
| `seen[node][chan]` | `checkpoint["versions_seen"][node][chan]` |
| `advance_watermarks` | `apply_writes` 里写 `versions_seen` 的部分 |
| `consume_triggers` | 屏障通道的 `consume` / `finish` 语义 |
| `expire_unwritten` | `apply_writes` 末尾 "Channels that weren't updated in this step are notified of a new step" |
| `_triggers` | `pregel/_algo.py` 的 `trigger` 函数 |

真实的 `apply_writes` 里，`versions_seen` 记的同样是**写入前**的
`checkpoint["channel_versions"][chan]` —— 这就是 `advance_watermarks`
必须排在 `_apply_writes` 之前的直接依据。
