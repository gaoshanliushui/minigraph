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
from typing import Annotated

from minigraph.graph import END, START, InMemorySaver, State, StateGraph, add_messages

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

    if last.get("role") == "tool":
        return {"steps": 1, "messages": [{
            "id": f"ai-{len(state['messages'])}",
            "role": "assistant",
            "content": f"根据工具结果，答案是：{last['content']}",
        }]}

    if last.get("role") == "human":
        return {"steps": 1, "messages": [{
            "id": "ai-0",
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "get_weather", "args": {"city": last["content"]}}],
        }]}

    return {"steps": 1, "messages": []}


def tool_node(state: AgentState) -> dict:
    """执行最后一条消息里的所有 tool_calls（真实场景是 ToolNode）。"""
    calls = state["messages"][-1].get("tool_calls", [])
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
    return g.compile(checkpointer=InMemorySaver())


if __name__ == "__main__":
    app = build()
    cfg = {"thread_id": "demo"}

    print("=" * 60)
    print("Q: 北京")
    print("=" * 60)
    for node, step, update in app.stream(
        {"messages": [{"id": "h1", "role": "human", "content": "北京"}], "steps": 0},
        config=cfg,
    ):
        msgs = update.get("messages", [])
        newest = msgs[-1] if msgs else {}
        brief = newest.get("content") or newest.get("tool_calls") or ""
        print(f"  step {step} | {node:<6} | 追加消息: {brief}")

    final = app.get_state(cfg).values
    print("\n最终消息：")
    for m in final["messages"]:
        print(f"  {m['role']:10} {m['content'] or m.get('tool_calls')}")

    print("\n超步数 =", final["steps"], "| checkpoint 数 =", len(app.get_state_history(cfg)))
