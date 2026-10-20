"""
§2 控制流原语
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class Command:
    """节点直接决定下一跳，可顺带提交 update（LangGraph 的 Command）。"""

    # 下一跳目标。三种形态都允许：
    #   - str            单个节点名（最常见）
    #   - Sequence[str]  一组节点名，同超步里都会执行（"从一处扇出"）
    #   - None           不指定下一跳，走默认静态边
    # 与 add_conditional_edges 是替代关系：当路由逻辑需要看节点内部状态
    # 而不是合并后的状态，或者要把"路由 + update"原子地一起提交时，用 Command 更顺手。
    goto: str | Sequence[str] | None = None

    # 顺带提交的 state 增量（与节点返回 dict 时语义相同：经过 reducer 合并）。
    # 允许在这里直接写 update，不必另外返回 dict；典型用法是路由到错误分支
    # 的同时记一笔 retry 计数。
    update: dict[str, Any] | None = None

    # 用于把 resume 值带回图内（与 LangGraph 对齐的 API 字段）。
    # 当前 mini 版的 resume 实际通过 config={'resume': ...} 在 invoke 边界注入；
    # 此字段保留是为兼容 langgraph 的"从图内部 resume"的写法，
    # runtime 对它的处理可在后续补全。
    resume: Any = None


@dataclass
class Send:
    """动态扇出：把一份独立输入派给某节点，构造并行任务。"""

    # 接收 Send 的节点名。runtime._execute_batch 会按这个名字查 graph.nodes
    # 取函数，**和普通任务共用节点函数池**——所以扇出的 worker 节点本身
    # 不需要"特殊声明"，只要用 add_node 加进图就行。
    node: str

    # 派给该节点的入参。两类形态：
    #   - Mapping（最常见）：会被合并进主状态再交给节点函数
    #     （runtime._wrap_send），节点能直接读 arg 里的 key。
    #   - 任意对象：用 _SendView 包一层，主状态挂在字典底层，
    #     但 __getattr__ 优先返回 arg 的属性——用于 Send(worker, obj)
    #     这种"以对象而非 dict 入参"的写法。
    arg: Any