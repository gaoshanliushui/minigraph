"""
`from __future__ import annotations` 下的 schema 提取。

这个文件**故意**开启 future annotations —— 于是本文件里 State 子类的
注解全是**字符串**，而不是 `Annotated[...]` 对象。

如果不把字符串解析回真实类型，reducer 会被**静默**丢掉：
`Annotated[int, operator.add]` 退化成 last-write-wins，
表现是并行分支写同一字段直接报并发冲突，而不是累加。
"""

from __future__ import annotations

import operator
from typing import Annotated

from minigraph.builder import StateGraph
from minigraph.constants import END, START
from minigraph.state import LAST_VALUE, State, add_messages, merge_dict


class S(State):
    n: Annotated[int, operator.add]
    msgs: Annotated[list, add_messages]
    meta: Annotated[dict, merge_dict]
    plain: str


def test_reducers_survive_future_annotations():
    assert S.reducers["n"] is operator.add
    assert S.reducers["msgs"] is add_messages
    assert S.reducers["meta"] is merge_dict
    assert S.reducers["plain"] is LAST_VALUE


def test_reducer_actually_applies():
    """不是只看 schema，还要确认合并语义真的生效（并行分支累加而不是报错）。"""

    class S2(State):
        n: Annotated[int, operator.add]

    g = StateGraph(S2)
    g.add_node("a", lambda s: {"n": 1})
    g.add_node("b", lambda s: {"n": 2})
    g.add_edge(START, "a")
    g.add_edge(START, "b")
    g.add_edge("a", END)
    g.add_edge("b", END)

    out = g.compile().invoke({"n": 0})
    assert out["n"] == 3          # 1 + 2，而不是并发写冲突


def test_unresolvable_annotation_falls_back_gracefully():
    """引用了还没定义的名字时，不该把整个类的 schema 拖垮。"""

    class S3(State):
        ok: Annotated[int, operator.add]
        later: "NotDefinedYet"  # noqa: F821

    assert S3.reducers["ok"] is operator.add
    assert "later" in S3.reducers
