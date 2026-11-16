"""
示例 3：真正的大模型调用 —— 把 01 的规则版 "LLM" 换成真模型（多 provider）

图结构和 01 **完全一样**（agent ⇄ tools 的 ReAct 回环），唯一区别是
`call_model` 里真的调了一次模型：要不要调工具、调哪个、参数是什么，
都由模型自己决定，不再是写死的规则。

    ┌──────────┐  含 tool_use?   ┌─────────┐
    │  agent   │ ──────────────► │  tools  │
    │ (真 LLM) │ ◄────────────── │ (本地)  │
    └────┬─────┘   tool_result   └─────────┘
         │ 纯文本回答
         ▼
        END

支持多家模型，配置全部在 **.env** 里（模板见 .env.example）：

    cp .env.example .env        # 填 key，选 LLM_PROVIDER
    uv sync --extra llm
    uv run --extra llm python examples/03_real_llm_agent.py "北京今天天气怎么样？"

内置 provider：anthropic（官方）、minimax、volcengine（火山引擎·方舟）。
三家都讲 **Anthropic Messages 协议**（MiniMax / 方舟提供 Anthropic 兼容
端点，Claude Code 就是走它接入的），所以一份消息格式、一个官方
anthropic SDK 就够，不需要为每家写转换层。端点/模型名如有调整，
改 .env 里的 LLM_BASE_URL / LLM_MODEL 即可，不用动代码。

依赖只有 anthropic + python-dotenv（llm extra；minigraph 内核零依赖）。
测试不需要 key —— tests/test_real_llm_example.py 用 fake client
跑通整个回环，并验证 SDK 响应落 checkpoint 的序列化路径。

消息格式：内部直接用 **Anthropic 原生消息**（role + content blocks），
只在消息级多挂一个 `id` 给 add_messages（按 id 追加/更新）；调 API 前
把消息级 id 剥掉。tool_use / tool_result 靠 block 里的 id 配对，原样回传。
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import time
from pathlib import Path
from typing import Annotated, NamedTuple

from anthropic import Anthropic
from dotenv import load_dotenv

from minigraph.builder import StateGraph
from minigraph.checkpoint_sqlite import SqliteSaver
from minigraph.constants import END, START
from minigraph.state import DeltaChannel, State, add_messages

_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _ROOT / ".env"


# ---------------------------------------------------------------- provider 配置
# 每家只需四项：Anthropic 兼容端点、key 的环境变量名、默认模型、认证头方式。
#   auth = "bearer"    -> Authorization: Bearer <key>（MiniMax / 方舟的 Claude Code 接法）
#   auth = "x-api-key" -> x-api-key: <key>（Anthropic 官方）
PROVIDERS = {
    "anthropic": {
        "base_url": None,                    # SDK 默认 https://api.anthropic.com
        "key_envs": ("ANTHROPIC_API_KEY",),
        "model": "claude-fable-5-2",
        "auth": "x-api-key",
    },
    "minimax": {
        "base_url": "https://api.minimaxi.com/anthropic",
        "key_envs": ("MINIMAX_API_KEY",),
        "model": "MiniMax-M3",
        "auth": "bearer",
    },
    "volcengine": {                          # 火山引擎·方舟
        "base_url": "https://ark.cn-beijing.volces.com/api/coding",
        "key_envs": ("ARK_API_KEY", "VOLCENGINE_API_KEY"),
        "model": "ark-code-latest",   # 也可填推理接入点 ep-xxx
        "auth": "bearer",
    },
}


def _resolve(env) -> dict:
    """
    从环境变量（含 .env，已由 load_dotenv 并入）解析出本次调用配置。

    优先级：LLM_API_KEY / LLM_BASE_URL / LLM_MODEL / LLM_AUTH（通用覆盖）
    > provider 自己的 key 变量 > PROVIDERS 里的默认值。
    """
    name = env.get("LLM_PROVIDER", "anthropic").strip().lower()
    if name not in PROVIDERS:
        raise RuntimeError(
            f"未知 LLM_PROVIDER={name!r}，可选：{', '.join(PROVIDERS)}"
        )
    p = PROVIDERS[name]
    api_key = env.get("LLM_API_KEY") or next(
        (env[k] for k in p["key_envs"] if env.get(k)), None
    )
    if not api_key:
        raise RuntimeError(
            f"provider {name} 缺少 API key：cp .env.example .env 后在 .env 里填 "
            f"{' 或 '.join(p['key_envs'])}（也可用 LLM_API_KEY 通用覆盖）"
        )
    return {
        "provider": name,
        "api_key": api_key,
        "base_url": env.get("LLM_BASE_URL") or p["base_url"],
        "model": env.get("LLM_MODEL") or p["model"],
        "auth": env.get("LLM_AUTH") or p["auth"],
    }


class _LLM(NamedTuple):
    client: Anthropic
    model: str
    provider: str


_llm: _LLM | None = None


def _get_llm() -> _LLM:
    """懒创建 client：import 本模块（比如测试）不应要求 .env / API key。"""
    global _llm
    if _llm is None:
        load_dotenv(_ENV_PATH)               # 不覆盖已存在的环境变量
        cfg = _resolve(os.environ)
        kwargs: dict = {}
        if cfg["base_url"]:
            kwargs["base_url"] = cfg["base_url"]
        if cfg["auth"] == "bearer":
            kwargs["auth_token"] = cfg["api_key"]
        else:
            kwargs["api_key"] = cfg["api_key"]
        _llm = _LLM(Anthropic(**kwargs), cfg["model"], cfg["provider"])
    return _llm


# ---------------------------------------------------------------- LLM 调用
def chat(messages: list[dict], tools: list[dict], *, max_tokens: int = 1024) -> list[dict]:
    """
    一次 Messages API 调用，返回 assistant 的 content blocks（dict 列表）。

    messages 是 Anthropic 原生格式：[{"role": "user"/"assistant", "content": ...}]。
    错误（key 无效、模型名错、额度不足）由 SDK 抛 AnthropicError 子类，
    报错信息原样带出，这里不再包一层。

    为什么把 SDK 的 block 对象 model_dump() 成 dict：
      - checkpoint 是 JSON 序列化的（SqliteSaver 不支持 pydantic 对象）；
      - 下一轮要把 assistant 消息原样回传给 API，dict 直接可用；
      - add_messages 按消息级 id 合并，dict 就好，不需要类实例。
    """
    llm = _get_llm()
    resp = llm.client.messages.create(
        model=llm.model,
        max_tokens=max_tokens,
        messages=messages,
        tools=tools,
    )
    return [block.model_dump() for block in resp.content]


# ---------------------------------------------------------------- 本地工具
WEATHER = {"北京": "晴 18°C", "上海": "小雨 22°C"}


def get_weather(city: str) -> str:
    return WEATHER.get(city, f"{city}：数据缺失（演示工具只有北京/上海）")


TOOLS = {"get_weather": get_weather}

TOOL_SCHEMAS = [
    {
        "name": "get_weather",
        "description": "查询中国城市的天气（演示数据，只有北京和上海）",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名，如 北京、上海"},
            },
            "required": ["city"],
        },
    }
]


# ---------------------------------------------------------------- 状态
class AgentState(State):
    # 消息 = Anthropic 原生格式 + 消息级 id。
    # DeltaChannel：checkpoint 里只存每步的增量消息，真实长对话下
    # checkpoint 体积随轮数线性增长，而不是平方（见 README §6.5）。
    messages: Annotated[list, DeltaChannel(add_messages, list)]


_ids = itertools.count(1)


def _new_id(prefix: str) -> str:
    # 带上进程启动时刻，避免多次运行写同一 thread 时和历史消息撞 id
    # （add_messages 按 id 覆盖，撞了就把老消息改掉了）。
    return f"{prefix}-{int(time.time())}-{next(_ids)}"


# ---------------------------------------------------------------- 节点
def call_model(state: AgentState) -> dict:
    """真 LLM：把历史原样发给 API（只剥掉消息级 id），存回 assistant 消息。"""
    api_messages = [
        {k: v for k, v in m.items() if k != "id"} for m in state["messages"]
    ]
    content = chat(api_messages, TOOL_SCHEMAS)
    return {"messages": [{
        "id": _new_id("ai"),
        "role": "assistant",
        "content": content,               # block 列表：text / tool_use ...
    }]}


def tool_node(state: AgentState) -> dict:
    """
    执行最后一条 assistant 消息里的所有 tool_use。

    结果打包成**一条** user 消息（content = tool_result blocks）——
    这是 Messages API 的要求：assistant 的 tool_use 之后必须紧跟
    user 角色的 tool_result，且 tool_use_id 一一对应。
    """
    last = state["messages"][-1]
    calls = [b for b in last["content"] if b.get("type") == "tool_use"]
    results = []
    for c in calls:
        try:
            out = TOOLS[c["name"]](**c.get("input", {}))
        except Exception as e:                       # 工具挂了就告诉模型
            out = f"工具执行失败: {e}"
        results.append({
            "type": "tool_result",
            "tool_use_id": c["id"],
            "content": str(out),
        })
    return {"messages": [{"id": _new_id("tool"), "role": "user", "content": results}]}


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    has_call = last["role"] == "assistant" and any(
        b.get("type") == "tool_use" for b in last["content"]
    )
    return "tools" if has_call else END


# ---------------------------------------------------------------- 构图
def build(checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("agent", call_model)
    g.add_node("tools", tool_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------- 展示
def _brief(msg: dict) -> str:
    """把一条消息的 content 压成一行可读文本。"""
    content = msg["content"]
    if isinstance(content, str):
        return content
    parts = []
    for b in content:
        t = b.get("type")
        if t == "text":
            parts.append(b.get("text", ""))
        elif t == "tool_use":
            parts.append(f"[调用 {b['name']}({json.dumps(b.get('input', {}), ensure_ascii=False)})]")
        elif b.get("type") == "tool_result":
            parts.append(f"[工具结果 {b.get('content')}]")
    return " | ".join(p for p in parts if p)


if __name__ == "__main__":
    try:
        llm = _get_llm()
    except RuntimeError as e:
        sys.exit(str(e))

    db = _ROOT / "minigraph.db"
    app = build(SqliteSaver(str(db)))
    cfg = {"thread_id": "real-demo"}

    question = sys.argv[1] if len(sys.argv) > 1 else "北京今天天气怎么样？"
    print(f"Q: {question}   (provider={llm.provider}, model={llm.model}, db={db.name})")

    for node, step, update in app.stream(
        {"messages": [{"id": _new_id("h"), "role": "user", "content": question}]},
        config=cfg,
    ):
        for m in update.get("messages", []):
            print(f"  step {step} | {node:<6} | {_brief(m)}")

    final = app.get_state(cfg).values["messages"]
    print(f"\n本 thread 共 {len(final)} 条消息；最后一条：")
    print(" ", _brief(final[-1]))
