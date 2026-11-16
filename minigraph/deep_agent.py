"""
§9 create_deep_agent —— 模仿 deepagents 的深度代理工厂

对照真实实现：deepagents/libs/deepagents/deepagents/graph.py 的
`create_deep_agent`。它不是一个新的图结构，而是"ReAct 回环 + 一组内置
能力"的**产品化组装**。本函数复刻其核心子集：

    create_deep_agent(model, tools, *, system_prompt, middleware,
                      subagents, checkpointer) -> CompiledGraph

  - model        模型适配器：callable(ModelRequest) -> content blocks。
                 request.model 初始为空，由适配器填真实模型名（这样内核
                 不绑定任何 SDK / provider；示例 04 用 03 的配置适配）。
  - tools        额外工具：@tool 产物或裸函数（自动包装），与内置工具合并。
  - system_prompt 用户指令，拼在内置 BASE 提示**之前**（USER -> BASE，
                 对齐真实实现的提示组装顺序）。
  - middleware   额外中间件。状态钩子自动生成节点（add_hook_nodes，
                 create_agent 拓扑）；around 钩子包模型/工具调用。
  - subagents    子代理规格：[{"name", "description", "prompt"?}]。
                 默认带一个 general-purpose 子代理（与主代理同工具、无 task）。
  - checkpointer 与 compile(checkpointer=...) 相同。

内置工具（对齐 deepagents 的三大能力）：

  规划      write_todos(todos)              待办清单，全量替换，存 state["todos"]
  文件系统  ls / read_file / write_file / edit_file
            **虚拟**文件系统，存 state["files"]（{路径: 内容}）—— 不碰真实
            磁盘，可随 checkpoint 持久化、可时间旅行；用来卸载中间结果，
            避免长上下文膨胀（deepagents 文件系统的核心动机）
  子代理    task(description, prompt, subagent_type)
            派一个独立子代理跑完整 ReAct 回环（独立消息历史），
            主对话只拿到它的最终结论

与真实 deepagents 的差异（刻意精简）：无 skills/memory/response_format/
summarization；无 execute（shell 需要沙箱后端）；文件系统是 state 内的
虚拟实现而非可插拔 backend；子代理不支持 async/远程、不可再嵌套 task。

模型/工具调用默认带一个最内层的异常兜底中间件（工具异常 -> 错误文本），
用户 middleware 在它外层。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Annotated, Any, Callable

from .builder import StateGraph
from .constants import END, START
from .middleware import (
    MiddlewareStack,
    ModelRequest,
    ToolRequest,
    add_hook_nodes,
    wrap_tool_call,
)
from .state import DeltaChannel, State, add_messages, merge_dict
from .tools import Tool, tool

BASE_PROMPT = """\
你是一个深度代理（deep agent），擅长把复杂的多步骤任务拆解并推进到底。

- 规划：收到复杂任务先用 write_todos 拆成待办清单，进行中及时把各项标记为
  in_progress / completed；清单是全量替换，每次写完整版本。
- 文件系统：你有一个虚拟文件系统（ls/read_file/write_file/edit_file）。
  把中间结果、笔记、长文本写进文件，需要时再读回，避免上下文膨胀；
  最终交付物（计划、报告、代码）也应该落成文件。
- 子代理：独立、无需看主对话上下文的子任务，用 task 派给子代理，
  主对话只保留它的结论。派工时把必要背景写进 prompt。
- 完成后用简洁的文字总结：做了什么、结果文件在哪里。"""


class DeepAgentState(State):
    messages: Annotated[list, DeltaChannel(add_messages, list)]
    todos: list                              # write_todos 全量替换（last-write-wins）
    files: Annotated[dict, merge_dict]       # 虚拟文件系统：路径 -> 内容（浅合并）


def _mid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _texts(content) -> str:
    """从一条消息的 content（str 或 blocks）里拼出纯文本。"""
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "") for b in content
                     if isinstance(b, dict) and b.get("type") == "text")


# ---------------------------------------------------------------------------
# state 工具：需要读写 agent state 的内置工具
# ---------------------------------------------------------------------------

@dataclass
class _StateTool(Tool):
    """state_fn(state, args) -> (给模型看的文本, state 更新 dict 或 None)。"""

    state_fn: Callable[[dict, dict], tuple[str, dict | None]] | None = None


def _builtin_tools() -> list[Tool]:
    def write_todos(state, args):
        todos = args.get("todos", [])
        lines = [f"- [{t.get('status', 'pending')}] {t.get('content', '')}"
                 for t in todos]
        return "待办清单已更新：\n" + "\n".join(lines), {"todos": todos}

    def ls(state, args):
        files = state.get("files") or {}
        return ("\n".join(sorted(files)) if files else "（文件系统为空）"), None

    def read_file(state, args):
        files = state.get("files") or {}
        path = args["path"]
        if path not in files:
            return f"错误：文件不存在：{path}", None
        return files[path], None

    def write_file(state, args):
        path, content = args["path"], args["content"]
        return f"已写入 {path}（{len(content)} 字符）", {"files": {path: content}}

    def edit_file(state, args):
        files = state.get("files") or {}
        path, old, new = args["path"], args["old_string"], args["new_string"]
        if path not in files:
            return f"错误：文件不存在：{path}", None
        if old not in files[path]:
            return f"错误：{path} 里找不到要替换的文本：{old!r}", None
        content = files[path].replace(old, new, 1)
        return f"已更新 {path}", {"files": {path: content}}

    def st(name, description, schema, fn) -> _StateTool:
        return _StateTool(name=name, description=description,
                          input_schema=schema, func=None, state_fn=fn)

    return [
        st("write_todos",
           "把复杂任务拆成待办清单并随时更新进度。todos 是全量替换，"
           "每项 {content: 任务描述, status: pending|in_progress|completed}",
           {"type": "object",
            "properties": {"todos": {"type": "array", "items": {"type": "object"}}},
            "required": ["todos"]},
           write_todos),
        st("ls", "列出虚拟文件系统里的全部文件路径",
           {"type": "object", "properties": {}, "required": []}, ls),
        st("read_file", "读取文件内容",
           {"type": "object",
            "properties": {"path": {"type": "string", "description": "文件路径，如 /notes.md"}},
            "required": ["path"]},
           read_file),
        st("write_file", "写入（新建或覆盖）一个文件",
           {"type": "object",
            "properties": {"path": {"type": "string"},
                           "content": {"type": "string", "description": "完整文件内容"}},
            "required": ["path", "content"]},
           write_file),
        st("edit_file", "把文件里的 old_string 替换为 new_string（只替换第一处）",
           {"type": "object",
            "properties": {"path": {"type": "string"},
                           "old_string": {"type": "string"},
                           "new_string": {"type": "string"}},
            "required": ["path", "old_string", "new_string"]},
           edit_file),
    ]


@wrap_tool_call
def _tool_guard(request, handler):
    """默认最内层兜底：工具异常变成错误文本回给模型。"""
    try:
        return handler(request)
    except Exception as e:
        return f"工具执行失败: {e}"


def _chain(g, src: str, names: list[str]) -> str:
    """src -> n1 -> n2 -> ...，返回链尾（names 为空时返回 src）。"""
    prev = src
    for n in names:
        g.add_edge(prev, n)
        prev = n
    return prev


# ---------------------------------------------------------------------------
# 组装
# ---------------------------------------------------------------------------

def _build_agent(model, registry: dict[str, Tool], prompt: str,
                 middleware, subapps: dict[str, Any] | None,
                 subagent_specs: list[dict], checkpointer):
    registry = dict(registry)
    if subapps is not None:
        desc = "；".join(f"{s['name']}（{s.get('description', '')}）"
                         for s in subagent_specs)

        @tool(name="task",
              description=f"把独立子任务派给子代理执行，返回它的最终结论。"
                          f"可用子代理：{desc}")
        def task(description: str, prompt: str,
                 subagent_type: str = "general-purpose") -> str:
            """派发子任务。

            Args:
                description: 一句话说明要做什么（用于日志）
                prompt: 给子代理的完整指令（它看不到主对话，背景要写全）
                subagent_type: 子代理名，默认 general-purpose
            """
            app = subapps.get(subagent_type)
            if app is None:
                return f"错误：未知子代理 {subagent_type!r}，可用：{sorted(subapps)}"
            out = app.invoke(
                {"messages": [{"id": _mid("h"), "role": "user", "content": prompt}]},
                config={"thread_id": f"sub-{_mid('t')}"},
            )
            return _texts(out["messages"][-1]["content"]) or "（子代理没有产出文本）"

        registry["task"] = task

    stack = MiddlewareStack([*middleware, _tool_guard])

    def call_model(state: DeepAgentState) -> dict:
        api_messages = [
            {k: v for k, v in m.items() if k != "id"} for m in state["messages"]
        ]
        req = ModelRequest(model="", messages=api_messages,
                           tools=[t.schema() for t in registry.values()],
                           system=prompt)
        content = stack.model_call(req, model)
        return {"messages": [{"id": _mid("ai"), "role": "assistant",
                              "content": content}]}

    def tool_node(state: DeepAgentState) -> dict:
        last = state["messages"][-1]
        calls = [b for b in last["content"] if b.get("type") == "tool_use"]
        results, extra = [], {}
        for c in calls:
            t = registry.get(c.get("name"))

            def _exec(request: ToolRequest, _t=t) -> str:
                if _t is None:
                    return f"错误：未知工具 {request.name!r}"
                if isinstance(_t, _StateTool):
                    text, upd = _t.state_fn(state, request.args)
                    for k, v in (upd or {}).items():
                        if k == "files":
                            extra.setdefault("files", {}).update(v)
                        else:
                            extra[k] = v
                    return text
                return str(_t.invoke(request.args))

            out = stack.tool_call(ToolRequest(c), _exec)
            results.append({"type": "tool_result", "tool_use_id": c["id"],
                            "content": out})
        return {**extra, "messages": [{"id": _mid("tool"), "role": "user",
                                       "content": results}]}

    def should_continue(state: DeepAgentState) -> str:
        last = state["messages"][-1]
        has_call = last["role"] == "assistant" and any(
            b.get("type") == "tool_use" for b in last["content"])
        return "tools" if has_call else END

    # create_agent 拓扑：状态钩子中间件各自成节点（与 examples/03 一致）
    g = StateGraph(DeepAgentState)
    g.add_node("model", call_model)
    g.add_node("tools", tool_node)
    hooks = add_hook_nodes(g, stack)
    g.add_edge(_chain(g, START, hooks["before_agent"] + hooks["before_model"]), "model")
    g.add_edge(_chain(g, "tools", hooks["before_model"]), "model")
    route_from = _chain(g, "model", hooks["after_model"])
    exits = hooks["after_agent"]
    if exits:
        g.add_edge(_chain(g, exits[0], exits[1:]), END)
    g.add_conditional_edges(route_from, should_continue,
                            {"tools": "tools", END: exits[0] if exits else END})
    return g.compile(checkpointer=checkpointer)


def create_deep_agent(model: Callable[[ModelRequest], list[dict]],
                      tools=None, *,
                      system_prompt: str | None = None,
                      middleware=(),
                      subagents: list[dict] | None = None,
                      checkpointer=None):
    """组装一个深度代理（签名与语义对照 deepagents.create_deep_agent）。

    参数含义见模块 docstring。返回编译好的图，直接 invoke/stream 即可：

        app = create_deep_agent(my_model, tools=[my_tool],
                                system_prompt="你是调研助手")
        out = app.invoke({"messages": [{"role": "user", "content": "..."}]},
                         config={"thread_id": "t1"})
        out["todos"], out["files"]   # 规划清单与虚拟文件系统也在 state 里
    """
    registry: dict[str, Tool] = {t.name: t for t in _builtin_tools()}
    for t in tools or []:
        t = t if isinstance(t, Tool) else tool(t)     # 裸函数自动包装
        registry[t.name] = t

    specs = [{"name": "general-purpose",
              "description": "通用子代理：与主代理相同的工具（不含 task），"
                             "适合独立的多步子任务"}]
    specs += list(subagents or [])
    # 子代理 = 同一套组装，但没有 task（不可再嵌套）、无 checkpointer
    subapps = {
        s["name"]: _build_agent(model, registry,
                                "\n\n".join(p for p in
                                            [s.get("prompt") or system_prompt,
                                             BASE_PROMPT] if p),
                                (), None, [], None)
        for s in specs
    }
    prompt = "\n\n".join(p for p in [system_prompt, BASE_PROMPT] if p)
    return _build_agent(model, registry, prompt, middleware,
                        subapps, specs, checkpointer)
