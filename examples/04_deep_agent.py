"""
示例 4：create_deep_agent —— 模仿 deepagents 的深度代理

在示例 3 的同一张 ReAct 回环上，用工厂方法一键组装出带**规划、虚拟文件
系统、子代理**三大能力的深度代理（minigraph/deep_agent.py）：

    app = create_deep_agent(llm, tools=[get_weather],
                            system_prompt="你是旅行规划助手。",
                            checkpointer=SqliteSaver(...))

    内置工具：write_todos（待办清单 -> state["todos"]）
              ls/read_file/write_file/edit_file（虚拟文件系统 -> state["files"]）
              task（派子代理跑独立 ReAct 回环，主对话只拿结论）

运行（复用示例 3 的 .env 多 provider 配置）：

    uv run --extra llm python examples/04_deep_agent.py
    uv run --extra llm python examples/04_deep_agent.py "把调研结果整理成报告写进 /report.md"

观察点：
  - stream 输出里能看到模型先 write_todos 拆任务，再逐步执行、更新清单；
  - 中间产物被写进虚拟文件（state["files"]），最后打印文件清单；
  - 独立子任务通过 task 派给 general-purpose 子代理。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from minigraph.checkpoint_sqlite import SqliteSaver
from minigraph.deep_agent import create_deep_agent
from minigraph.middleware import ModelRequest
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

    app = create_deep_agent(
        llm,
        tools=[get_weather],
        system_prompt="你是旅行规划助手，回答用中文。",
        checkpointer=SqliteSaver(str(_ROOT / "minigraph_deep.db")),
    )
    cfg = {"thread_id": "deep-demo"}
    question = sys.argv[1] if len(sys.argv) > 1 else \
        "帮我规划北京两日游：先查天气，用 write_todos 拆任务，把每日安排写进 /plan.md，最后总结。"
    print(f"Q: {question}   (provider={info.provider}, model={info.model})\n")

    for node, step, update in app.stream(
        {"messages": [{"role": "user", "content": question}]},
        config=cfg,
    ):
        for m in update.get("messages", []):
            print(f"  step {step} | {node:<28} | {ex03._brief(m)}")
        if "todos" in update:
            for t in update["todos"]:
                print(f"           todo [{t.get('status')}] {t.get('content')}")
        if "files" in update:
            print(f"           files 更新: {sorted(update['files'])}")

    values = app.get_state(cfg).values
    print(f"\n消息 {len(values['messages'])} 条；待办 {len(values.get('todos') or [])} 项；"
          f"文件 {sorted((values.get('files') or {}))}")
    print("最终回答：", ex03._brief(values["messages"][-1]))
