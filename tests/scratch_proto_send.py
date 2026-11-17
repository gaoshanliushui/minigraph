"""原型：验证 Send 扇出 + task_writes 复用 能做到"每个工具恰好执行一次"。"""
import operator
import time
from typing import Annotated

from minigraph.builder import StateGraph
from minigraph.constants import END, START
from minigraph.context import interrupt
from minigraph.control import Send
from minigraph.state import State, add_messages


class S(State):
    messages: Annotated[list, add_messages]
    results: Annotated[list, operator.add]
    calls: Annotated[list, operator.add]


executed = []


def dispatch(state):
    calls = state.get("calls") or []
    print("  >> dispatch sees calls =", calls, " state keys =", sorted(state))
    return [Send("worker", {"call": c}) for c in calls]


def worker(state):
    call = state["call"]
    answer = interrupt({"tool_call_id": call["id"], "name": "risky"})
    # 让后面的调用先完成，检验合并顺序是否按**调用顺序**而不是完成顺序
    time.sleep(0.1 if call["id"] == "c1" else 0.0)
    executed.append(call["id"])
    return {"results": [{"id": call["id"], "answer": answer}]}


def join(state):
    rs = state.get("results") or []
    return {"messages": [{"id": "join", "role": "user", "content": rs}],
            "results": [], "calls": []}


def build():
    from minigraph.checkpoint import InMemorySaver
    g = StateGraph(S)
    g.add_node("dispatch", dispatch)
    g.add_node("worker", worker)
    g.add_node("join", join)
    g.add_edge(START, "dispatch")
    g.add_edge("worker", "join")
    g.add_edge("join", END)
    return g.compile(checkpointer=InMemorySaver())


if __name__ == "__main__":
    app = build()
    out = app.invoke({"messages": [{"id": "h1", "role": "user", "content": "go"}],
                      "results": [], "calls": [{"id": "c1"}, {"id": "c2"}]},
                     config={"thread_id": "t1"}, debug=True)
    print("interrupted:", app.interrupted)
    print("executed:", executed)
    ck = app.get_state({"thread_id": "t1"})
    print("CKPT channel_values:", ck.channel_values)
    print("CKPT step:", ck.step, "values:", ck.values)

    out2 = app.invoke(None, config={"thread_id": "t1",
                                    "resume": {"type": "approve"}}, debug=True)
    print("after resume1: interrupted =", app.interrupted)
    print("executed:", executed)
    out3 = app.invoke(None, config={"thread_id": "t1",
                                    "resume": {"type": "approve"}}, debug=True)
    print("after resume2: interrupted =", app.interrupted)
    print("executed:", executed)
    print("messages:", out3.get("messages"))
