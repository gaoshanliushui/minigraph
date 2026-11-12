"""
§0 哨兵与类型别名

包内最底层的符号；其他模块都从这里 import。
"""

from __future__ import annotations

from typing import Any, Callable

START = "__start__"
END = "__end__"

Reducer = Callable[[Any, Any], Any]  # (旧值, 新值) -> 合并后的值

# ---------------------------------------------------------------------------
# 通道名模板（对齐真实 LangGraph 的命名）
# ---------------------------------------------------------------------------
# "边不是数据结构，是一次 channel 写入" —— 静态边 add_edge(A,B) 编译成
# A 写 "branch:to:B"，B 订阅 "branch:to:B"。
BRANCH_TO = "branch:to:{}"

# 多源汇聚 add_edge(["a","b"], "c") 编译成屏障通道 "join:a+b:c"。
JOIN = "join:{}"

# Send 队列（PUSH 轨）。名字与 LangGraph 一致，但不对外承诺。
TASKS = "__pregel_tasks"


class _Missing:
    def __repr__(self) -> str:  # pragma: no cover
        return "<missing>"

    def __copy__(self) -> "_Missing":
        return self

    def __deepcopy__(self, memo: dict) -> "_Missing":
        # 通道快照用 MISSING 表示"空"。深拷贝 checkpoint 时若复制出另一个
        # _Missing 实例，from_checkpoint 的 `is not MISSING` 判断就会失效，
        # 把空通道误恢复成有值通道。
        return self


MISSING = _Missing()