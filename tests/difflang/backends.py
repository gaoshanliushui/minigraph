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
from minigraph.checkpoint import InMemorySaver as MiniSaver
from minigraph.constants import END as MINI_END
from minigraph.constants import START as MINI_START
from minigraph.context import interrupt as mini_interrupt
from minigraph.control import Command as MiniCommand
from minigraph.control import Send as MiniSend
from minigraph.state import State as MiniState

# ---------------------------------------------------------------------------
# LangGraph 侧
# ---------------------------------------------------------------------------
try:
    from langgraph.checkpoint.memory import InMemorySaver as LGSaver
    from langgraph.graph import END as LG_END
    from langgraph.graph import START as LG_START
    from langgraph.graph import StateGraph as LGStateGraph
    from langgraph.types import Command as LGCommand
    from langgraph.types import Send as LGSend
    from langgraph.types import interrupt as lg_interrupt
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
    # 检查点相关： saver 类型 + interrupt 的来源模块
    checkpointer_cls: Any = None
    interrupt: Any = None

    def make_state(self, annotations: dict[str, Any]) -> Any:
        """
        用同一份注解造出**这一套后端**能用的状态 schema。

        两边都读同一份 Annotated[...] 元数据，所以一份注解生成两个 schema，
        保证 reducer 语义完全一致。
        """
        if self.name == "minigraph":
            # minigraph 的 State.__init_subclass__ 读 cls.__annotations__
            # （3.14 起注解是惰性的，类字典里没有 "__annotations__" 这一项）
            return type("S", (self.state_base,), {"__annotations__": dict(annotations)})
        # LangGraph 用 TypedDict，get_type_hints(include_extras=True) 读注解
        return TypedDict("S", dict(annotations))  # type: ignore[misc]

    # -- 检查点相关的形状差异 ---------------------------------------------
    def config(self, thread_id: str, **extra) -> dict:
        """
        造出这一套后端的 config。

        minigraph 把 thread_id 放在顶层；LangGraph 放在 configurable 里。
        """
        if self.name == "minigraph":
            return {"thread_id": thread_id, **extra}
        return {"configurable": {"thread_id": thread_id}, **extra}

    def resume(self, app: Any, thread_id: str, value: Any) -> Any:
        """
        答复挂起的中断，让图继续跑。

        minigraph 从 config['resume'] 取值；LangGraph 从 Command(resume=) 取。
        """
        if self.name == "minigraph":
            return app.invoke(None, {"thread_id": thread_id, "resume": value})
        return app.invoke(self.Command(resume=value),
                          {"configurable": {"thread_id": thread_id}})

    def history(self, app: Any, cfg: dict) -> list[tuple[int, dict]]:
        """
        历史快照统一成 [(step, 可直接喂回 update_state 的 config)]。

        两边形状不同：minigraph 的快照是 Checkpoint 对象、config 形如
        {"thread_id":..., "checkpoint_id":...}；LangGraph 的是 StateSnapshot，
        config 藏在 snapshot.config["configurable"] 里。
        """
        if self.name == "minigraph":
            tid = cfg["thread_id"]
            return [(c.step, {"thread_id": tid, "checkpoint_id": c.id})
                    for c in app.get_state_history(cfg)]
        out: list[tuple[int, dict]] = []
        for snap in app.get_state_history(cfg):
            conf = dict(snap.config["configurable"])
            out.append((snap.metadata["step"], {"configurable": conf}))
        return out


MINIGRAPH = Backend(
    name="minigraph",
    state_graph_cls=MiniStateGraph,
    state_base=MiniState,
    Send=MiniSend,
    Command=MiniCommand,
    START=MINI_START,
    END=MINI_END,
    checkpointer_cls=MiniSaver,
    interrupt=mini_interrupt,
)

LANGGRAPH = Backend(
    name="langgraph",
    state_graph_cls=LGStateGraph if HAS_LANGGRAPH else None,
    state_base=None,
    Send=LGSend if HAS_LANGGRAPH else None,
    Command=LGCommand if HAS_LANGGRAPH else None,
    START=LG_START if HAS_LANGGRAPH else "",
    END=LG_END if HAS_LANGGRAPH else "",
    checkpointer_cls=LGSaver if HAS_LANGGRAPH else None,
    interrupt=lg_interrupt if HAS_LANGGRAPH else None,
)
