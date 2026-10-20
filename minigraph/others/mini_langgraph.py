"""
mini_langgraph — 一个 ~200 行的 LangGraph 最小实现
====================================================
目标：用最短代码讲清 LangGraph 的 5 个核心原理，可直接 python 运行。

核心对应关系（真实 langgraph -> 本实现）：
  StateGraph(...)                    StateGraph（state 由 channel + reducer 定义）
  add_node(name, fn)                 add_node（节点 = 纯函数：读 state，返回部分写入）
  add_edge(src, dst)                 add_edge（普通边）
  add_conditional_edges(src, fn, mp) add_conditional_edges（条件路由）
  compile().invoke(inputs)           compile().invoke（编译 + 执行）
  Send(node, arg)                    Send（动态并行分支，map-reduce 基础）

五个原理：
  1. State 是唯一的数据载体。整个图共享一份 state，节点只读它、只返回“对它的局部写入”。
  2. 图 = 有向图。节点 + 边 + 条件边，编译后固定。
  3. Super-step（超步）是执行单元：每一轮把“当前所有就绪节点”并行执行，
     收集各自写入 -> 用 reducer 合并 -> 推进 state -> 按边解析下一批就绪节点，
     直到全部到达 END。环在这里被 recursion_limit 拦住。
  4. Reducer 定义“多人写同一 channel 怎么合并”：默认覆盖，可自定义如求和/拼接。
  5. Send 在超步中派发独立子任务，各自带参数，结果在下游用 reducer 汇合。

语法：python3 mini_langgraph.py
"""

import functools
from typing import Any, Callable, Dict, List, Tuple, Union

# 特殊节点
START = "__start__"
END = "__end__"


def overwrite(a: Any, b: Any) -> Any:
    """默认 reducer：后写覆盖先写"""
    return b


class GraphRecursionError(Exception):
    pass


class Send:
    """动态并行分支：向 node 派发一个带独立参数 arg 的子任务"""

    __slots__ = ("node", "arg")

    def __init__(self, node: str, arg: Any):
        self.node = node
        self.arg = arg


class StateGraph:
    def __init__(self, schema: Dict[str, Callable] = None):
        # schema: {channel_name: reducer}；reducer 合并同一超步内多个节点对同一 channel 的写入
        self.schema = schema or {}
        self.nodes: Dict[str, Callable] = {}
        self.edges: List[Tuple[str, str]] = []
        self.cond_edges: List[Tuple[str, Callable, Dict[str, str]]] = []

    # ---------- 构建 API ----------
    def add_node(self, name: str, fn: Callable) -> "StateGraph":
        if name in (START, END):
            raise ValueError(f"节点名 {name} 是保留字")
        self.nodes[name] = fn
        return self

    def add_edge(self, src: str, dst: str) -> "StateGraph":
        self.edges.append((src, dst))
        return self

    def add_conditional_edges(
        self, src: str, router: Callable[[Dict], str], mapping: Dict[str, str]
    ) -> "StateGraph":
        self.cond_edges.append((src, router, mapping))
        return self

    def compile(self) -> "CompiledGraph":
        valid = set(self.nodes) | {START, END}
        for s, d in self.edges:
            if s not in valid or d not in valid:
                raise ValueError(f"边 {s}->{d} 引用了不存在的节点")
        for s, _, m in self.cond_edges:
            if s not in valid:
                raise ValueError(f"条件边起点 {s} 不存在")
            for k, v in m.items():
                if v not in valid:
                    raise ValueError(f"条件边 {s} 的路由键 {k} 指向不存在的节点 {v}")
        return CompiledGraph(self)


class CompiledGraph:
    def __init__(self, graph: StateGraph):
        self.graph = graph
        # 预计算出边表，执行时 O(1) 查路由
        self._out: Dict[str, List[str]] = {}
        self._cond: Dict[str, Tuple[Callable, Dict[str, str]]] = {}
        for s, d in graph.edges:
            self._out.setdefault(s, []).append(d)
        for s, router, mapping in graph.cond_edges:
            self._cond[s] = (router, mapping)

    # ---------- 原理3/4：合并写入（reducer） ----------
    def _apply_writes(self, state: Dict, writes: List[Tuple[str, Dict]]) -> Dict:
        """一个超步结束后：把多个节点的写入按 channel 用 reducer 合并，叠加到旧 state。"""
        new = dict(state)  # 未被写入的 channel 原样保留
        grouped: Dict[str, List[Any]] = {}
        for _, update in writes:
            if update is None:
                continue
            for k, v in update.items():
                grouped.setdefault(k, []).append(v)
        for k, values in grouped.items():
            if k == "__send__":
                continue  # 控制通道，不写入 state
            reducer = self.graph.schema.get(k, overwrite)
            # reducer 与「通道既有值 + 本超步全部写入」合并（默认 overwrite 后写覆盖）
            if k in new:
                new[k] = functools.reduce(reducer, [new[k]] + values)
            else:
                new[k] = functools.reduce(reducer, values) if len(values) > 1 else values[0]
        return new

    # ---------- 原理2：按边解析下一批就绪节点 ----------
    def _next_tasks(self, state: Dict, done: List[str]) -> List[str]:
        pending: List[str] = []
        for node in done:
            for dst in self._out.get(node, []):
                if dst != END:
                    pending.append(dst)
            if node in self._cond:
                router, mapping = self._cond[node]
                key = router(state)
                if key not in mapping:
                    raise KeyError(f"路由键 '{key}' 不在映射中: {list(mapping)}")
                dst = mapping[key]
                if dst != END:
                    pending.append(dst)
        # 同一超步内相同节点只执行一次（保留首次出现顺序）
        seen, tasks = set(), []
        for t in pending:
            if t not in seen:
                seen.add(t)
                tasks.append(t)
        return tasks

    # ---------- 原理1/3/5：super-step 主循环 ----------
    def invoke(self, inputs: Dict, recursion_limit: int = 25) -> Dict:
        state = dict(inputs)
        tasks = [START]  # START 是伪节点：只作为边的起点，不执行逻辑
        step = 0
        while tasks:
            step += 1
            if step > recursion_limit:
                raise GraphRecursionError(
                    f"递归超过 {recursion_limit} 步（疑似死循环或未收敛环）"
                )
            writes: List[Tuple[str, Dict]] = []
            # 本超步所有就绪节点并行执行（同步版顺序跑，语义等价：都读同一份 state）
            for name in tasks:
                if name == START:
                    continue
                writes.append((name, self.graph.nodes[name](state)))
            executed = tasks
            # 原理5：Send 动态分支 —— 在本超步内派发子任务并立即执行，各自带 arg
            sends: List[Send] = []
            for _, update in writes:
                if update and "__send__" in update:
                    sends.extend(update["__send__"])
            for s in sends:
                sub = dict(state)
                if isinstance(s.arg, dict):
                    sub.update(s.arg)  # Send 携带的参数并入分支状态
                writes.append((s.node, self.graph.nodes[s.node](sub)))
                executed = executed + [s.node]
            # 原理3/4：推进 state
            state = self._apply_writes(state, writes)
            # 原理2：解析下一超步
            tasks = self._next_tasks(state, executed)
        return state


# ============================ 示例 ============================

def demo_conditional_routing():
    """演示：条件路由 + 状态推进。模拟“内容审核”流程。"""
    def append_log(a, b):
        return a + b  # channel 级 reducer：日志累加

    builder = StateGraph({"log": append_log, "verdict": overwrite})
    builder.add_node("analyze", lambda s: {"verdict": s["draft"], "log": ["analyzed"]})
    builder.add_node("publish", lambda s: {"status": "published", "log": ["published"]})
    builder.add_node("revise", lambda s: {"status": "revised", "log": ["revised"]})
    builder.add_edge(START, "analyze")

    def route(state):
        return "ok" if len(state["draft"]) >= 5 else "short"

    builder.add_conditional_edges("analyze", route, {"ok": "publish", "short": "revise"})
    builder.add_edge("publish", END)
    builder.add_edge("revise", END)

    graph = builder.compile()
    out = graph.invoke({"draft": "hello world"})
    print("长文本路径:", out)
    out2 = graph.invoke({"draft": "hi"})
    print("短文本路径:", out2)


def demo_map_reduce_send():
    """演示：Send 动态并行分支（map-reduce）+ reducer 聚合。"""
    def concat(a, b):
        return a + b  # 把多个 reviewer 的分数拼成一个 list

    builder = StateGraph({"scores": concat})
    builder.add_node("fan_out", lambda s: {"__send__": [
        Send("reviewer", {"score": 3}),
        Send("reviewer", {"score": 5}),
        Send("reviewer", {"score": 4}),
    ]})
    builder.add_node("reviewer", lambda s: {"scores": [s["score"]]})
    builder.add_node("aggregate", lambda s: {"total": sum(s["scores"])})
    builder.add_edge(START, "fan_out")
    builder.add_edge("fan_out", "aggregate")   # 与 reviewer->aggregate 去重，只跑一次
    builder.add_edge("reviewer", "aggregate")
    builder.add_edge("aggregate", END)

    graph = builder.compile()
    out = graph.invoke({"scores": []})
    print("map-reduce 结果:", out)


if __name__ == "__main__":
    demo_conditional_routing()
    demo_map_reduce_send()
#（注：内容由AI生成）
