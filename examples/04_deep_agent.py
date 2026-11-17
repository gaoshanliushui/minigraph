"""
示例 4：create_deep_agent —— 模仿 deepagents 的深度代理

在示例 3 的同一张 ReAct 回环上，用工厂方法一键组装出带**规划、真实文件
backend、子代理和自动摘要**能力的深度代理（deep_agent/deep_agent.py）：

    backend = FilesystemBackend(_ROOT / ".deep_agent_workspace")
    app = create_deep_agent(llm, tools=[get_weather], backend=backend,
                            system_prompt="你是旅行规划助手。",
                            interrupt_on={"write_file": True},
                            checkpointer=SqliteSaver(...))

    内置工具：write_todos（待办清单 -> state["todos"]）
              ls/read_file/write_file/edit_file/glob/grep/delete（真实 backend）
              task（派子代理跑独立 ReAct 回环，子代理共享 backend）

运行（复用示例 3 的 .env 多 provider 配置）：

    # 不传问题：进入连续对话；输入 exit / quit 或按 Ctrl-D 退出。
    uv run --extra llm python examples/04_deep_agent.py

    # 传入问题：保持单次问答模式。
    uv run --extra llm python examples/04_deep_agent.py "把调研结果整理成报告写进 /report.md"

连续对话始终使用同一个 thread_id。每轮只传入新的 user 消息，之前的问答由
SqliteSaver 恢复，不需要手动把完整 messages 再传一遍。

观察点：
  - stream 输出里能看到模型先 write_todos 拆任务，再逐步执行、更新清单；
  - 中间产物通过 FilesystemMiddleware 写进 .deep_agent_workspace，最后打印文件清单；
  - 独立子任务通过 task 派给 general-purpose 子代理，并共享同一个 backend；
  - write_file 执行前由 HumanInTheLoopMiddleware 挂起，终端批准后自动 resume；
  - 达到上下文阈值时 SummarizationMiddleware 自动压缩模型输入并转存历史。
"""

from __future__ import annotations

import importlib.util
import json
import sys
import uuid
from pathlib import Path

from deep_agent import create_deep_agent
from deep_agent.backends import FilesystemBackend
from deep_agent.middleware import ModelRequest
from minigraph.checkpoint_sqlite import SqliteSaver
from minigraph.tools import tool
from minigraph.utils import print_graph

_ROOT = Path(__file__).resolve().parent.parent


def _load_ex03():
    """复用示例 3 的 provider 配置 / client / _chat_once（.env 不用配第二份）。"""
    spec = importlib.util.spec_from_file_location(
        "ex03_real_llm_agent", Path(__file__).with_name("03_real_llm_agent.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m          # State 字符串注解解析需要（见 tests）
    spec.loader.exec_module(m)
    return m


ex03 = _load_ex03()


def llm(request: ModelRequest) -> list[dict]:
    """把 03 的多 provider 配置适配成 create_deep_agent 的 model 协议。"""
    request.model = ex03._get_llm().model
    return ex03._chat_once(request)


@tool
def get_weather(city: str) -> str:
    """查询中国城市的天气（演示数据，只有北京和上海）。

    Args:
        city: 城市名，如 北京、上海
    """
    return {"北京": "晴 18°C", "上海": "小雨 22°C"}.get(city, f"{city}：数据缺失")


def run_turn(app, cfg: dict, question: str) -> dict:
    """完成一轮问答；若发生 HITL 中断，审批后原地恢复本轮。"""
    run_input = {"messages": [{
        "id": f"human-{uuid.uuid4().hex[:12]}",
        "role": "user",
        "content": question,
    }]}
    run_cfg = cfg
    while True:
        # 双模式流式：custom 拿模型 token（实时打印），updates 拿节点更新。
        streaming = False              # 是否正在一段 token 输出里
        for mode, payload in app.stream(
            run_input, config=run_cfg, stream_mode=["updates", "custom"],
        ):
            if mode == "custom":
                if not streaming:
                    print("  模型 | ", end="", flush=True)
                    streaming = True
                print(payload.get("text", ""), end="", flush=True)
                continue
            if streaming:               # token 段落结束，补换行
                print()
                streaming = False
            node, step, update = payload
            for message in update.get("messages", []):
                # token 已经实时打印过了，这里只报工具调用 / 工具结果，
                # 避免把 assistant 正文再打印一遍。
                for b in (message.get("content") or []):
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "tool_use":
                        print(f"  step {step} | {node:<28} | "
                              f"[调用 {b['name']}({json.dumps(b.get('input', {}), ensure_ascii=False)})]")
                    elif b.get("type") == "tool_result":
                        body = b.get("content")
                        text = body if isinstance(body, str) else json.dumps(
                            body, ensure_ascii=False, default=str)
                        # print(f"  step {step} | {node:<28} | "
                        #       f"[工具结果 {len(text)} 字符]")
            if "todos" in update:
                for todo in update["todos"]:
                    print(f"           todo [{todo.get('status')}] "
                          f"{todo.get('content')}")
        if streaming:
            print()

        pending = app.interrupted
        if pending is None:
            break
        print("\n工具调用等待人工审批：")
        print(json.dumps(pending, ensure_ascii=False, indent=2, default=str))
        decisions = []
        for action in pending["action_requests"]:
            answer = input(
                f"批准执行 {action['name']} {action['args']}？[y/N] "
            ).strip().lower()
            decisions.append(
                {"type": "approve"} if answer in {"y", "yes"}
                else {"type": "reject", "message": "用户未批准这次文件写入"})
        # resume 只继续当前中断，不会开启新的用户轮次。
        run_input = None
        run_cfg = {**cfg, "resume": {"decisions": decisions}}

    return app.get_state(cfg).values


if __name__ == "__main__":
    try:
        info = ex03._get_llm()
    except RuntimeError as e:
        sys.exit(str(e))

    backend = FilesystemBackend(_ROOT / ".deep_agent_workspace")
    app = create_deep_agent(
        llm,
        tools=[get_weather],
        backend=backend,
        system_prompt="你是旅行规划助手，回答用中文。",
        interrupt_on={"write_file": {
            "allowed_decisions": ["approve", "reject"]}},
        checkpointer=SqliteSaver(str(_ROOT / "minigraph.db")),
    )

    # print_graph(app)

    cfg = {"thread_id": "deep-demo6"}
    one_shot = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"provider={info.provider}, model={info.model}, thread={cfg['thread_id']}")
    if one_shot is None:
        print("连续对话模式：输入 exit / quit 或按 Ctrl-D 退出。")

    while True:
        try:
            if one_shot is not None:
                question = one_shot
                print(f"\n你> {question}")
            else:
                question = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not question:
            if one_shot is not None:
                break
            continue
        if question.strip().lower() in {"exit", "quit", "退出"}:
            print("再见！")
            break

        values = run_turn(app, cfg, question)
        # print("\n助手：", ex03._brief(values["messages"][-1]))
        # print(f"[本 thread 已保存 {len(values['messages'])} 条消息]")

        if one_shot is not None:
            break

    files = sorted(item["path"] for item in (backend.glob("*").matches or []))
    print(f"workspace 文件：{files}")
