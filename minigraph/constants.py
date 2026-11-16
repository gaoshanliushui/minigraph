"""
§0 哨兵与类型别名

包内最底层的符号；其他模块都从这里 import。
"""

from __future__ import annotations

from typing import Any, Callable

START = "__start__"
END = "__end__"

Reducer = Callable[[Any, Any], Any]  # (旧值, 新值) -> 合并后的值


# 默认 reducer：last-write-wins。
#
# 定义在 constants 里而不是 state.py —— state.py 要 import channels.py
# （认 DeltaChannel），channels.py 又要 import LAST_VALUE，两边直接
# import 会成环。constants 是两边的公共底座，放这儿正好。
# state.py 会把它再导出一次（`from .state import LAST_VALUE` 仍然可用）。
LAST_VALUE: Reducer = lambda old, new: new

# ---------------------------------------------------------------------------
# 通道名模板（对齐真实 LangGraph 的命名）
# ---------------------------------------------------------------------------
# "边不是数据结构，是一次 channel 写入" —— 静态边 add_edge(A,B) 编译成
# A 写 "branch:to:B"，B 订阅 "branch:to:B"。
BRANCH_TO = "branch:to:{}"

# 多源汇聚 add_edge(["a","b"], "c") 编译成屏障通道 "join:a+b:c"。
JOIN = "join:{}"

# Send 队列（PUSH 轨）。名字与 LangGraph 一致，但对外不承诺。
TASKS = "__pregel_tasks"

# ---------------------------------------------------------------------------
# DeltaChannel 的快照节奏
# ---------------------------------------------------------------------------
# 距上次快照最多隔多少个超步，就必须再落一份完整快照。
#
# 为什么需要它：DeltaChannel 的重放深度由"最近一次快照之后积累了多少
# 增量"决定。只按"写入次数"（snapshot_frequency）计数是不够的 ——
# 某个通道可能很久没被写，但每份 checkpoint 都在为它记录重放链，
# 链条会一直变长。这个上界保证重放深度有天花板。
#
# 对齐 LangGraph：LANGGRAPH_DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT，默认 5000。
import os

DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT = int(
    os.getenv("MINIGRAPH_DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT", "5000")
)


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