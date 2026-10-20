"""
示例 2：两个进阶机制

A. Human-in-the-loop：写稿件 -> 人工审批(interrupt) -> 通过/打回
   展示：挂起、进程级"停机"、从 checkpoint 恢复、以及 update_state 纠偏。

B. Map-Reduce 扇出：Send 把 N 个子任务并行派发，reducer 自动汇总。

运行：python examples/02_hitl_and_fanout.py
"""

import operator
import time
from typing import Annotated

from minigraph.builder import StateGraph
from minigraph.checkpoint import InMemorySaver
from minigraph.constants import END, START
from minigraph.context import interrupt
from minigraph.control import Send
from minigraph.state import State


# ================================================================ A. 人工审批
class DraftState(State):
    topic: str
    draft: str
    feedback: Annotated[list, operator.add]
    status: str


def plan(state):
    time.sleep(0.01)                          # 假装在调模型
    return {
        "draft": f"《{state['topic']}》初稿：这是一段自动生成的正文……",
        "status": "drafted",
    }


def human_review(state):
    """
    interrupt() 在这里"冻结"整张图。
    返回值 = 恢复时 config={"resume": ...} 传入的内容。
    注意：节点会从头重跑，所以 interrupt 之前的代码必须幂等。
    """
    decision = interrupt({
        "节点": "human_review",
        "草稿": state["draft"],
        "请回复": {"action": "approve | reject", "note": "可选意见"},
    })
    action = decision.get("action")
    return {
        "status": action,
        "feedback": [f"{action}: {decision.get('note', '')}".strip(": ")],
    }


def revise(state):
    return {"draft": state["draft"] + "（已按意见修订）", "status": "revised"}


def publish(state):
    return {"status": "published", "feedback": ["已发布"]}


def after_review(state):
    return "publish" if state["status"] == "approve" else "revise"


def build_draft_graph():
    g = StateGraph(DraftState)
    g.add_node("plan", plan)
    g.add_node("human_review", human_review)
    g.add_node("revise", revise)
    g.add_node("publish", publish)
    g.add_edge(START, "plan")
    g.add_edge("plan", "human_review")
    g.add_conditional_edges(
        "human_review", after_review, {"publish": "publish", "revise": "revise"}
    )
    g.add_edge("revise", "human_review")     # 打回后再走一次审批
    g.add_edge("publish", END)
    return g.compile(checkpointer=InMemorySaver())


def demo_hitl():
    print("=" * 64)
    print("A. Human-in-the-loop")
    print("=" * 64)
    app = build_draft_graph()
    cfg = {"thread_id": "article-42"}

    s = app.invoke({"topic": "硬核财经", "draft": "", "feedback": [], "status": ""}, cfg)
    print(f"第一次运行后 status={s['status']!r}，图已冻结：{app.interrupted}")
    print("  checkpoint 数 =", len(app.get_state_history(cfg)))

    # —— 此处可以关掉进程、等真人几小时后回复，checkpoint 仍在 ——
    s = app.invoke(None, {**cfg, "resume": {"action": "reject", "note": "结论不够锐利"}})
    print(f"人工打回后 status={s['status']!r}，再次冻结：{app.interrupted}")

    s = app.invoke(None, {**cfg, "resume": {"action": "approve"}})
    print(f"人工通过后 status={s['status']!r}")
    print("feedback:", s["feedback"])
    print("最终草稿:", s["draft"])


# ================================================================ B. 扇出
class ResearchState(State):
    question: str
    subtasks: Annotated[list, operator.add]
    findings: Annotated[list, operator.add]
    report: str


def split(state):
    return {"subtasks": ["市场规模", "竞争格局", "监管风险"]}


def dispatch(state):
    """返回 list[Send]：每个子任务一份独立输入，下一超步并行执行。"""
    return [Send("research_one", {"question": state["question"], "aspect": t})
            for t in state["subtasks"]]


def research_one(state):
    # Send 的 arg 会被合并进入参，所以这里能直接读到 aspect
    time.sleep(0.05)
    return {"findings": [f"[{state['aspect']}] 关于「{state['question']}」的调研结论"]}


def reduce_report(state):
    body = "\n".join(f"- {f}" for f in sorted(state["findings"]))
    return {"report": f"# {state['question']}\n\n{body}"}


def build_map_reduce():
    g = StateGraph(ResearchState)
    g.add_node("split", split)
    g.add_node("dispatch", dispatch)
    g.add_node("research_one", research_one)
    g.add_node("report", reduce_report)
    g.add_edge(START, "split")
    g.add_edge("split", "dispatch")
    g.add_edge("dispatch", "research_one")
    g.add_edge("research_one", "report")
    g.add_edge("report", END)
    return g.compile()


def demo_fanout():
    print()
    print("=" * 64)
    print("B. Map-Reduce 扇出（3 个子任务并行）")
    print("=" * 64)
    app = build_map_reduce()
    t0 = time.time()
    out = app.invoke({"question": "固态电池产业", "subtasks": [], "findings": [], "report": ""})
    cost = time.time() - t0
    print(out["report"])
    print(f"\n总耗时 {cost*1000:.0f} ms（串行应为 ~150ms，说明 3 个子任务在同一超步并发）")


if __name__ == "__main__":
    demo_hitl()
    demo_fanout()
