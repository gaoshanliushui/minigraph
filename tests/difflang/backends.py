"""
两套后端的适配层。

minigraph 和 LangGraph 的 API 几乎同名，差异集中在四处：
    1. 状态 schema 的形态（State 子类 vs TypedDict）
    2. Send / Command 的来源模块
    3. START / END 常量的来源模块
    4. checkpoint 的 config 形态（{"thread_id":...} vs {"configurable":{...}}）

Backend 把这些差异收敛掉，scenario 里就只剩"图长什么样"。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# minigraph 侧
# ---------------------------------------------------------------------------
from minigraph.builder import StateGraph as MiniStateGraph
from minigraph.constants import END as MINI_END
from minigraph.constants import START as MINI_START
from minigraph.control import Command as MiniCommand
from minigraph.control import Send as MiniSend
from minigraph.state import State as MiniState

# ---------------------------------------------------------------------------
# LangGraph 侧
# ---------------------------------------------------------------------------
try:
    from langgraph.graph import END as LG_END
    from langgraph.graph import START as LG_START
    from langgraph.graph import StateGraph as LGStateGraph
    from langgraph.types import Command as LGCommand
    from langgraph.types import Send as LGSend
    from typing import TypedDict

    HAS_LANGGRAPH = True
except ImportError:  # pragma: no cover
    HAS_LANGGRAPH = False


@dataclass
class Backend:
    """一套后端需要的全部符号。"""

    name: str
    state_graph_cls: Any
    state_base: Any
    Send: Any
    Command: Any
    START: str
    END: str

    def make_state(self, annotations: dict[str, Any]) -> Any:
        """
        用同一份注解造出**这一套后端**能用的状态 schema。

        两边都读同一份 Annotated[...] 元数据，所以一份注解生成两个 schema，
        保证 reducer 语义完全一致。
        """
        if self.name == "minigraph":
            # minigraph 的 State.__init_subclass__ 读 cls.__dict__["__annotations__"]
            return type("S", (self.state_base,), {"__annotations__": dict(annotations)})
        # LangGraph 用 TypedDict，get_type_hints(include_extras=True) 读注解
        return TypedDict("S", dict(annotations))  # type: ignore[misc]


MINIGRAPH = Backend(
    name="minigraph",
    state_graph_cls=MiniStateGraph,
    state_base=MiniState,
    Send=MiniSend,
    Command=MiniCommand,
    START=MINI_START,
    END=MINI_END,
)

LANGGRAPH = Backend(
    name="langgraph",
    state_graph_cls=LGStateGraph if HAS_LANGGRAPH else None,
    state_base=None,
    Send=LGSend if HAS_LANGGRAPH else None,
    Command=LGCommand if HAS_LANGGRAPH else None,
    START=LG_START if HAS_LANGGRAPH else "",
    END=LG_END if HAS_LANGGRAPH else "",
)
