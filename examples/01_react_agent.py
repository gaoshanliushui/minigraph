"""
示例 1：最小 ReAct Agent —— LangGraph 最经典的图结构

        ┌──────────┐
        │  agent   │◄────────┐
        └────┬─────┘         │
             │ 有 tool_calls? │
        ┌────▼─────┐         │
        │  tools   │─────────┘
        └────┬─────┘
             │ 无
             ▼
            END

这里把"LLM"换成一个规则函数，好处是：你能完整看到控制流，
而不被模型的不确定性干扰。换成真的 ChatModel 只需修改 `call_model`。

运行：python examples/01_react_agent.py
"""

import operator
from pathlib import Path
from typing import Annotated

from minigraph.builder import StateGraph
from minigraph.checkpoint import InMemorySaver
from minigraph.checkpoint_sqlite import SqliteSaver
from minigraph.constants import END, START
from minigraph.state import State, add_messages

# checkpoint 落盘位置跟着**文件**走，而不是跟着当前工作目录 ——
# 这样 `python examples/01_react_agent.py`（从仓库根跑）和
# `cd examples && python 01_react_agent.py` 写的是同一个文件。
_DB_PATH = Path(__file__).resolve().parent.parent / "minigraph.db"

# ---------------------------------------------------------------- 假工具
WEATHER = {"北京": "晴 18°C", "上海": "小雨 22°C"}


def get_weather(city: str) -> str:
    return WEATHER.get(city, f"{city}：数据缺失")


TOOLS = {"get_weather": get_weather}


class AgentState(State):
    messages: Annotated[list, add_messages]   # 对话历史，按 id 追加/更新
    steps: Annotated[int, operator.add]       # 循环计数，并发安全


# ---------------------------------------------------------------- 假 LLM
def call_model(state: AgentState) -> dict:
    """规则版"LLM"：没有工具结果就发工具调用，有了就直接回答。"""
    last = state["messages"][-1]
    # id 必须**每条消息都不同** —— add_messages 是"按 id 追加/更新"，
    # id 撞了就会把已有那条改掉，历史就串了。这里用当前消息数当序号。
    n = len(state["messages"])

    if last.get("role") == "tool":
        return {"steps": 1, "messages": [{
            "id": f"ai-{n}",
            "role": "assistant",
            "content": f"根据工具结果，答案是：{last['content']}",
        }]}

    if last.get("role") == "human":
        return {"steps": 1, "messages": [{
            "id": f"ai-{n}",
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "get_weather", "args": {"city": last["content"]}}],
        }]}

    return {"steps": 1, "messages": []}


def _last_assistant_with_calls(state: AgentState) -> dict:
    """找最近一条带 tool_calls 的 assistant 消息。

    不能直接看 messages[-1] —— 多轮对话时最后一条可能是 tool 结果，
    真正的工具调用在它前面几条。
    """
    for msg in reversed(state["messages"]):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            return msg
    return {}


def tool_node(state: AgentState) -> dict:
    """执行最后一条 assistant 消息里的所有 tool_calls（真实场景是 ToolNode）。"""
    calls = _last_assistant_with_calls(state).get("tool_calls", [])
    outs = []
    for i, call in enumerate(calls):
        fn = TOOLS[call["name"]]
        outs.append({
            "id": f"tool-{state['steps']}-{i}",
            "role": "tool",
            "name": call["name"],
            "content": fn(**call["args"]),
        })
    return {"messages": outs}


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if last.get("tool_calls") else END


# ---------------------------------------------------------------- 构图
def build():
    g = StateGraph(AgentState)
    g.add_node("agent", call_model)
    g.add_node("tools", tool_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")      # 关键：工具执行完回到模型，形成循环
    # checkpointer = InMemorySaver()
    checkpointer =  SqliteSaver(str(_DB_PATH))
    return g.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    app = build()
    cfg = {"thread_id": "demo", "debug": True}

    # 同一个 thread 连问两轮。
    #
    # 每轮 human 消息的 id 必须**整个 thread 内唯一**：add_messages 是
    # "按 id 追加/更新"，id 撞了就会把已有那条**改掉**（这是对齐
    # LangGraph 的语义，不是 bug）。所以这里用"当前历史条数"当序号，
    # 保证反复运行本脚本也不会撞 id、历史能一路累积。
    for turn, city in enumerate(["北京", "上海"], start=1):
        try:
            n = len(app.get_state(cfg).values.get("messages", []))
        except ValueError:
            n = 0   # thread 还没有 checkpoint
        mid = f"h{n}"

        print("=" * 60)
        print(f"第 {turn}轮  Q: {city}   (id={mid})")
        print("=" * 60)
        for node, step, update in app.stream(
            {"messages": [{"id": mid, "role": "human", "content": city}],
             "steps": 0},
            config=cfg,
        ):
            msgs = update.get("messages", [])
            newest = msgs[-1] if msgs else {}
            brief = newest.get("content") or newest.get("tool_calls") or ""
            print(f"  step {step} | {node:<6} | 追加消息: {brief}")

    final = app.get_state(cfg).values
    print("\n最终消息（两轮累积在一起）：")
    for m in final["messages"]:
        print(f"  {m['role']:10} {m['content'] or m.get('tool_calls')}")

    print("\n超步数 =", final["steps"], "| checkpoint 数 =", len(app.get_state_history(cfg)))

