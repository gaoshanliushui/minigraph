"""
§1 状态：字段 -> reducer

这是 LangGraph 最本质的设计：
  节点是"无状态函数"，状态只存在于图这一层；
  节点返回的不是新状态，而是一组"增量（update）"，
  由 reducer 决定如何合并回主状态。
默认 reducer 是 last-write-wins；因此同一超步里两个并行节点
写同一个字段会直接报错（防止静默覆盖），这是很多人第一次
用并行分支时踩的坑。
"""

from __future__ import annotations

from typing import Any, Mapping

from .constants import MISSING, Reducer


# 默认 reducer：last-write-wins
LAST_VALUE: Reducer = lambda old, new: new


# 消息列表 reducer：按 id 追加/更新，语义对齐 langgraph 的 add_messages。
#
# 什么时候调用：
#   每个超步结束后由 runtime._merge_writes 调用。
#   节点若在 messages 字段上写了 N 条新消息，会逐条进入这里合并。
#   同超步里多个节点各写各的消息也会被合并在一起。
#
# 为什么这么设计：
#   - 不直接 list.__add__ 是因为我们要按 id 去重/覆盖，不是按顺序拼接。
#     langgraph 的 messages 通道在节点重跑 / checkpoint 恢复时，旧消息
#     可能带着旧 id 再次被 emit；如果用 + 拼接就会重复。add_messages 让
#     "同 id 的最新版本胜出"，从而节点即使被重跑，结果也是幂等的。
#   - msg 兼容 dict 和对象两种形态（`Mapping` 协议 vs `getattr(id)`），
#     是因为 langgraph 允许用户用 dataclass / pydantic 类做消息，
#     不强制必须是 dict。
def add_messages(old, new):
    """消息列表 reducer：按 id 追加/更新，语义对齐 langgraph 的 add_messages。"""
    old = [] if old is MISSING else old
    new = new if isinstance(new, (list, tuple)) else [new]
    out = list(old)
    for msg in new:
        mid = msg.get("id") if isinstance(msg, Mapping) else getattr(msg, "id", None)
        idx = next((i for i, m in enumerate(out)
                    if (m.get("id") if isinstance(m, Mapping) else getattr(m, "id", None)) == mid), None
                   ) if mid else None
        if idx is None:
            out.append(msg)
        else:
            out[idx] = msg
    return out


# Dict 浅合并 reducer。
#
# 什么时候调用：
#   与 add_messages 同：被 _merge_writes 在合并阶段调用。
#   典型场景——多个并行节点各写一个 dict 字段的不同子键
#   （{"meta": {"a": 1}} 与 {"meta": {"b": 2}}），需要在字段层把它们的
#   子键并集起来。
#
# 为什么这么设计：
#   - 缺省 last-write-wins 在并行写同字段时会丢兄弟分支的键；
#     merge_dict 提供 dict 字段的"并行安全"语义，与 add_messages / operator.add
#     一脉相承。
#   - 只做浅合并：深合并在 metadata 这类场景里很容易把同一 key 下嵌套结构
#     互相覆盖，反而更危险；浅合并让作者主动决定要不要再嵌一层。
#   - 与 add_messages 不同，它**不会**改变"节点必须返回增量"的契约——
#     节点依然只写自己关心的子键，框架负责合并。
def merge_dict(old, new):
    if old is MISSING:
        return dict(new or {})
    merged = dict(old)
    merged.update(new or {})
    return merged


class State(dict):
    """
    声明式状态。用法：

        class S(State):
            messages: Annotated[list, add_messages]   # 自定义合并
            total:    Annotated[int, operator.add]    # 累加（并行安全）
            query:    str                             # 缺省 last-write-wins
    """

    reducers: dict[str, Reducer] = {}

    "父类在被子类继承的瞬间自动触发**，不是实例化对象的时候调用"
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        reducers: dict[str, Reducer] = {}
        for base in reversed(cls.__mro__[1:]):
            reducers.update(getattr(base, "reducers", {}))
        for name, ann in cls.__dict__.get("__annotations__", {}).items():
            if name.startswith("_") or name == "reducers":
                continue
            reducers[name] = _extract_reducer(ann) or LAST_VALUE
        cls.reducers = reducers

    @classmethod
    def schema(cls) -> dict[str, Reducer]:
        return dict(cls.reducers)


# 从 `Annotated[T, ...]` 中取出 reducer。
#
# 什么时候调用：
#   State.__init_subclass__ 在子类声明完成后遍历每条字段注解时调用；
#   是"声明式 reducer"机制的唯一入口。如果提取出 None，表示这条字段
#   没有 reducer，调用方会退化为 LAST_VALUE。
#
# 为什么这么设计：
#   typing.Annotated[T, x, y] 把 T 之外的 x/y 放进 `__metadata__` 元组。
#   历史上 langgraph 出现过两种写法：
#     1. Annotated[list, add_messages]           —— 元数据本身就是 callable
#     2. Annotated[list, (operator.add, ...)]    —— 元数据是 (callable, ...)
#   我们两种都接受（callable 直接用，否则看 first 是不是 tuple 且首元素
#   callable），把语法选择权留给用户。
def _extract_reducer(ann) -> Reducer | None:
    meta = getattr(ann, "__metadata__", None)
    if not meta:
        return None
    first = meta[0]
    if callable(first):
        return first
    if isinstance(first, tuple) and first and callable(first[0]):
        return first[0]
    return None


# 把 StateGraph.__init__ 收到的 state_schema 参数归一化成 {字段: reducer}。
#
# 什么时候调用：
#   StateGraph.__init__(state_schema=...) 内部调一次。
#   是构造期最早期的一次"用户输入 -> 内部表示"转换。
#
# 为什么这么设计：
#   用户可以用三种形态之一：
#     - 缺省 / None：声明 schema 全用 last-write-wins（最简形态）
#     - State 子类：从 cls.schema()（也就是 __init_subclass__ 算出来的
#       reducers）读出来
#     - 裸 dict {field: reducer}：函数式声明，最轻
#   内部统一收敛到 dict 形态，下游（_merge_writes / _apply）就只面对一种
#   数据结构，避免在热路径里反复 isinstance / issubclass。
def _schema(obj: Mapping[str, Reducer] | type[State] | None) -> dict[str, Reducer]:
    if obj is None:
        return {}
    if isinstance(obj, type) and issubclass(obj, State):
        return obj.schema()
    return dict(obj)