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


def _schema(obj: Mapping[str, Reducer] | type[State] | None) -> dict[str, Reducer]:
    if obj is None:
        return {}
    if isinstance(obj, type) and issubclass(obj, State):
        return obj.schema()
    return dict(obj)