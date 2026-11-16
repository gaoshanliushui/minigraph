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

    uv run --extra llm python examples/04_deep_agent.py
    uv run --extra llm python examples/04_deep_agent.py "把调研结果整理成报告写进 /report.md"

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
from pathlib import Path

from deep_agent import create_deep_agent
from deep_agent.backends import FilesystemBackend
from deep_agent.middleware import ModelRequest
from minigraph.checkpoint_sqlite import SqliteSaver
from minigraph.tools import tool

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
    cfg = {"thread_id": "deep-demo"}
    question = sys.argv[1] if len(sys.argv) > 1 else \
        "帮我规划北京两日游：先查天气，用 write_todos 拆任务，把每日安排写进 /plan.md，最后总结。"
    print(f"Q: {question}   (provider={info.provider}, model={info.model})\n")

    run_input = {"messages": [{"role": "user", "content": question}]}
    run_cfg = cfg
    while True:
        for node, step, update in app.stream(run_input, config=run_cfg):
            for m in update.get("messages", []):
                print(f"  step {step} | {node:<28} | {ex03._brief(m)}")
            if "todos" in update:
                for t in update["todos"]:
                    print(f"           todo [{t.get('status')}] {t.get('content')}")

        pending = app.interrupted
        if pending is None:
            break
        print("\n工具调用等待人工审批：")
        print(json.dumps(pending, ensure_ascii=False, indent=2, default=str))
        decisions = []
        for action in pending["action_requests"]:
            answer = input(
                f"批准执行 {action['name']} {action['args']}？[y/N] ").strip().lower()
            decisions.append(
                {"type": "approve"} if answer in {"y", "yes"}
                else {"type": "reject", "message": "用户未批准这次文件写入"})
        run_input = None
        run_cfg = {**cfg, "resume": {"decisions": decisions}}

    values = app.get_state(cfg).values
    files = sorted(item["path"] for item in (backend.glob("*").matches or []))
    print(f"\n消息 {len(values['messages'])} 条；待办 {len(values.get('todos') or [])} 项；"
          f"workspace 文件 {files}")
    print("最终回答：", ex03._brief(values["messages"][-1]))
