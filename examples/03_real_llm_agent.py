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

工具用 **@tool** 声明（minigraph/tools.py）：从函数签名 + docstring
自动生成给模型看的 JSON Schema，不再手写 TOOL_SCHEMAS。
模型调用和工具调用都走**中间件链**（minigraph/middleware.py）：

    LogCalls（@middleware 类式，日志）
      └─ retry_model（@wrap_model_call，失败重试）   -> 真实模型调用
      └─ tool_guard（@wrap_tool_call，异常兜底）     -> 真实工具执行

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
from minigraph.middleware import (
    MiddlewareStack,
    ModelRequest,
    ToolRequest,
    add_hook_nodes,
    middleware,
    wrap_model_call,
    wrap_tool_call,
)
from minigraph.state import DeltaChannel, State, add_messages
from minigraph.tools import tool

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


# ---------------------------------------------------------------- 中间件
@middleware
class LogCalls:
    """类式中间件（@middleware）：状态钩子 + around 钩子，只打印不干预。"""

    def before_agent(self, state):
        print("  [mw] ===== agent 开始 =====")

    def after_agent(self, state):
        print(f"  [mw] ===== agent 结束：共 {len(state.get('messages', []))} 条消息 =====")

    def before_model(self, state):
        print(f"    [mw] before_model：当前历史 {len(state.get('messages', []))} 条")

    def after_model(self, state):
        print(f"    [mw] after_model：模型已返回，历史 {len(state.get('messages', []))} 条")

    def wrap_model_call(self, request, handler):
        print(f"    [mw] model -> {request.model}（{len(request.messages)} 条消息）")
        result = handler(request)
        n = sum(b.get("type") == "tool_use" for b in result)
        print(f"    [mw] model <- {len(result)} 个 block（含 {n} 个 tool_use）")
        return result

    def wrap_tool_call(self, request, handler):
        result = handler(request)
        print(f"    [mw] tool {request.name}({request.args}) -> {result}")
        return result


@wrap_model_call
def retry_model(request, handler):
    """函数式中间件：网络错误/5xx 最多试 3 次；4xx（key、模型名错）直接抛。"""
    for attempt in range(3):
        try:
            return handler(request)
        except Exception as e:
            status = getattr(e, "status_code", None)
            if (status is not None and 400 <= status < 500) or attempt == 2:
                raise
            wait = 2 ** attempt
            print(f"    [mw] 模型调用失败（{e}），{wait}s 后第 {attempt + 2} 次尝试")
            time.sleep(wait)


@wrap_tool_call
def tool_guard(request, handler):
    """工具异常 -> 错误文本回给模型，而不是炸掉整张图。"""
    try:
        return handler(request)
    except Exception as e:
        return f"工具执行失败: {e}"


# 洋葱序：LogCalls 最外层 -> retry/guard -> 真实调用
STACK = MiddlewareStack([LogCalls, retry_model, tool_guard])


# ---------------------------------------------------------------- LLM 调用
def _chat_once(request: ModelRequest) -> list[dict]:
    """真正发 HTTP 的一层（中间件链的最内层 handler）。"""
    resp = _get_llm().client.messages.create(
        model=request.model,
        max_tokens=request.max_tokens,
        messages=request.messages,
        tools=request.tools,
    )
    # block 对象 model_dump() 成 dict：checkpoint 走 JSON 序列化（存不了
    # pydantic 对象），下一轮回传 API 和 add_messages 合并也都用 dict。
    return [block.model_dump() for block in resp.content]


def chat(messages: list[dict], tools: list[dict], *, max_tokens: int = 1024) -> list[dict]:
    """一次带中间件链的模型调用：日志 -> 重试 -> 真实请求，返回 content blocks。"""
    req = ModelRequest(model=_get_llm().model, messages=list(messages),
                       tools=list(tools), max_tokens=max_tokens)
    return STACK.model_call(req, _chat_once)


# ---------------------------------------------------------------- 本地工具
WEATHER = {"北京": "晴 18°C", "上海": "小雨 22°C"}


@tool
def get_weather(city: str) -> str:
    """查询中国城市的天气（演示数据，只有北京和上海）。

    Args:
        city: 城市名，如 北京、上海
    """
    return WEATHER.get(city, f"{city}：数据缺失（演示工具只有北京/上海）")


# @tool 从签名 + docstring 自动生成给模型看的 JSON Schema，不再手写
TOOLS = {get_weather.name: get_weather}
TOOL_SCHEMAS = [get_weather.schema()]


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
    """真 LLM：把历史原样发给 API（只剥掉消息级 id），存回 assistant 消息。

    before_model/after_model 不在函数体里调用 —— 它们是**独立的图节点**
    （见 build()），执行时机由图拓扑表达。
    """
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
        # 工具执行也走中间件链：日志 + 异常兜底（tool_guard）
        out = STACK.tool_call(ToolRequest(c), _run_tool)
        results.append({
            "type": "tool_result",
            "tool_use_id": c["id"],
            "content": out,
        })
    return {"messages": [{"id": _new_id("tool"), "role": "user", "content": results}]}


def _run_tool(request: ToolRequest) -> str:
    """真正执行工具的一层（中间件链的最内层 handler）。"""
    return str(TOOLS[request.name].invoke(request.args))


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    has_call = last["role"] == "assistant" and any(
        b.get("type") == "tool_use" for b in last["content"]
    )
    return "tools" if has_call else END


# ---------------------------------------------------------------- 构图
def _chain(g, src: str, names: list[str]) -> str:
    """src -> n1 -> n2 -> ...，返回链尾（names 为空时返回 src）。"""
    prev = src
    for n in names:
        g.add_edge(prev, n)
        prev = n
    return prev


def build(checkpointer=None):
    """
    对齐 LangChain create_agent（agents/factory.py）的接线：
    实现了状态钩子的中间件**各自生成一个节点**，执行时机由图拓扑表达：

        START -> LogCalls.before_agent -> LogCalls.before_model -> agent
        tools -> LogCalls.before_model -> agent          （回环也过 before_model）
        agent -> LogCalls.after_model -> 条件路由
        路由的 END 分支 -> LogCalls.after_agent -> END   （只跑一次）
    """
    g = StateGraph(AgentState)
    g.add_node("model", call_model)
    g.add_node("tools", tool_node)
    hooks = add_hook_nodes(g, STACK)     # 状态钩子中间件 -> 节点

    # 入口：START -> before_agent* -> before_model* -> agent
    g.add_edge(_chain(g, START, hooks["before_agent"] + hooks["before_model"]), "model")
    # 回环：tools -> before_model* -> agent（每轮 ReAct 都过 before_model）
    g.add_edge(_chain(g, "tools", hooks["before_model"]), "model")
    # 出口：agent -> after_model*（逆序）-> 条件路由
    route_from = _chain(g, "model", hooks["after_model"])
    # 路由的 END 分支先过 after_agent*（逆序），再到 END
    exits = hooks["after_agent"]
    if exits:
        g.add_edge(_chain(g, exits[0], exits[1:]), END)
    g.add_conditional_edges(route_from, should_continue,
                            {"tools": "tools", END: exits[0] if exits else END})
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

    user_input = {"messages": [{"id": _new_id("h"), "role": "user", "content": question}]}

    for node, step, update in app.stream(user_input, config=cfg):
        for m in update.get("messages", []):
            print(f"  step {step} | {node:<6} | {_brief(m)}")

    final = app.get_state(cfg).values["messages"]
    print(f"\n本 thread 共 {len(final)} 条消息；最后一条：")
    print(" ", _brief(final[-1]))
