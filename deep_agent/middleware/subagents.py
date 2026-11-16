"""通过 ``task`` 工具向主代理提供隔离的子代理。

对照 deepagents.middleware.subagents 的核心路径：

* ``SubAgent`` 是声明式规格，由本项目的 ``_build_agent`` 编译；
* ``CompiledSubAgent`` 直接提供带 ``invoke`` 的 runnable；
* 中间件通过 ``tools`` 注册 ``task``，并可用 ``wrap_model_call`` 注入说明；
* 子代理只收到本次任务，收不到主对话，也不能默认继续派发 ``task``；
* 主代理只收到子代理最后一条非空 assistant 文本。

与完整 deepagents 的边界：这里不实现 fork 上下文、异步/远程子代理、结构化
响应或子代理状态回写。嵌套 HITL 的恢复协议也尚未实现，因此规格里的
``interrupt_on`` 会被明确拒绝，而不是静默忽略。
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any, Literal, NotRequired, TypedDict

from minigraph.tools import Tool, tool

from .core import Middleware, ModelRequest


class SubAgent(TypedDict):
    """声明式子代理规格。

    直接交给 ``SubAgentMiddleware`` 时，``model`` 和 ``tools`` 必填，与参考
    实现一致。``create_deep_agent`` 会在调用前补齐这两项及默认中间件配置。

    ``prompt`` 是本项目早期 ``system_prompt`` 的兼容别名；新代码应使用后者。
    """

    name: str
    description: str
    model: NotRequired[Callable[[ModelRequest], list[dict]]]
    tools: NotRequired[Sequence[Tool | Callable]]
    middleware: NotRequired[Sequence[Any]]
    system_prompt: NotRequired[str]
    prompt: NotRequired[str]
    skills: NotRequired[list[str]]
    summarization: NotRequired[bool]
    mode: NotRequired[Literal["isolated", "handoff"]]


class CompiledSubAgent(TypedDict):
    """预编译子代理；runnable 的输出 state 必须含 ``messages``。"""

    name: str
    description: str
    runnable: Any
    mode: NotRequired[Literal["isolated", "handoff"]]


_SubAgentSpec = SubAgent | CompiledSubAgent

DEFAULT_SUBAGENT_PROMPT = """In order to complete the objective that the user asks of you, you have access to a number of standard tools.

The calling agent only sees your final assistant message, not your intermediate work, tool results, or status tracking. Ensure your final
response contains the complete answer."""

DEFAULT_GENERAL_PURPOSE_DESCRIPTION = (
    "General-purpose agent for researching complex questions, searching for "
    "files and content, and executing multi-step tasks. This agent has access "
    "to the main agent's non-task tools."
)

GENERAL_PURPOSE_SUBAGENT: SubAgent = {
    "name": "general-purpose",
    "description": DEFAULT_GENERAL_PURPOSE_DESCRIPTION,
    "system_prompt": DEFAULT_SUBAGENT_PROMPT,
}

TASK_TOOL_DESCRIPTION = """Launch an isolated subagent to handle a complex, multi-step task.

Available agent types:
{available_agents}

Specify subagent_type to select the agent. Each invocation is stateless: the
subagent sees only the delegated task and returns one final report. Put all
necessary context in description. The optional prompt field is a backwards-
compatible override for description; new calls only need description."""


def _validate_subagents(subagents: Sequence[_SubAgentSpec]) -> None:
    if not subagents:
        raise ValueError("At least one subagent must be specified")
    seen: set[str] = set()
    for spec in subagents:
        name = spec.get("name")
        description = spec.get("description")
        if not isinstance(name, str) or not name:
            raise ValueError("Each subagent must have a non-empty string name")
        if not isinstance(description, str) or not description:
            raise ValueError(f"SubAgent {name!r} must have a description")
        if name in seen:
            raise ValueError(
                f"Duplicate subagent name {name!r}; each name must be unique")
        seen.add(name)
        mode = spec.get("mode")
        if mode not in (None, "isolated", "handoff"):
            raise ValueError(
                f"SubAgent {name!r} mode {mode!r} is not supported; "
                "only isolated subagents are available")
        if "runnable" not in spec and spec.get("interrupt_on"):
            raise ValueError(
                f"SubAgent {name!r} cannot use interrupt_on yet: nested "
                "interrupt resume is not implemented")


def create_sub_agent(spec: SubAgent, *, backend=None):
    """把完整的声明式规格编译成 minigraph agent。

    编译放在调用时并延迟导入工厂，避免 ``deep_agent`` 导入中间件、中间件
    再导入工厂形成模块级循环依赖。
    """
    name = spec.get("name", "<unnamed>")
    if "model" not in spec:
        raise ValueError(f"SubAgent {name!r} must specify 'model'")
    if "tools" not in spec:
        raise ValueError(f"SubAgent {name!r} must specify 'tools'")

    from ..deep_agent import _build_agent, _make_registry

    prompt = spec.get("system_prompt", spec.get("prompt", ""))
    return _build_agent(
        spec["model"],
        _make_registry(spec["tools"]),
        prompt,
        spec.get("middleware", ()),
        None,                         # 子代理默认无 checkpointer、不可嵌套 task
        backend=backend,
        skills=spec.get("skills"),
        summarization=spec.get("summarization", True),
    )


def _texts(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        block.get("text", "") for block in content
        if isinstance(block, dict) and block.get("type") == "text")


def _final_assistant_text(result: dict) -> str:
    if "messages" not in result:
        raise ValueError(
            "CompiledSubAgent must return a state containing a 'messages' key")
    for message in reversed(result["messages"]):
        if isinstance(message, dict) and message.get("role") == "assistant":
            text = _texts(message.get("content")).rstrip()
            if text:
                return text
    return ""


def _build_task_tool(subagents: Sequence[_SubAgentSpec],
                     task_description: str | None = None, *,
                     backend=None) -> Tool:
    """编译子代理并创建分发它们的 ``task`` 工具。"""
    _validate_subagents(subagents)
    available_agents = "\n".join(
        f"- {spec['name']}: {spec['description']}" for spec in subagents)
    if task_description is None:
        description = TASK_TOOL_DESCRIPTION.format(
            available_agents=available_agents)
    elif "{available_agents}" in task_description:
        description = task_description.format(available_agents=available_agents)
    else:
        description = task_description

    graphs: dict[str, Any] = {}
    for spec in subagents:
        runnable = (spec["runnable"] if "runnable" in spec
                    else create_sub_agent(spec, backend=backend))
        if not callable(getattr(runnable, "invoke", None)):
            raise TypeError(
                f"SubAgent {spec['name']!r} runnable must provide invoke()")
        graphs[spec["name"]] = runnable

    @tool(name="task", description=description)
    def task(description: str, subagent_type: str = "general-purpose",
             prompt: str | None = None) -> str:
        """派发一个隔离子任务并返回子代理的最终报告。

        Args:
            description: 给子代理的完整任务说明和必要背景
            subagent_type: 可用子代理的名称，默认 general-purpose
            prompt: 兼容旧调用；提供时替代 description 作为实际任务说明
        """
        app = graphs.get(subagent_type)
        if app is None:
            allowed = ", ".join(f"`{name}`" for name in graphs)
            return (f"错误：未知子代理 {subagent_type!r}；"
                    f"可用类型：{allowed}")
        task_prompt = description if prompt is None else prompt
        result = app.invoke(
            {"messages": [{
                "id": f"sub-human-{uuid.uuid4().hex[:12]}",
                "role": "user",
                "content": task_prompt,
            }]},
            config={"thread_id": f"sub-{uuid.uuid4().hex[:12]}"},
        )
        return _final_assistant_text(result) or "（子代理没有产出文本）"

    return task


class SubAgentMiddleware(Middleware):
    """通过中间件工具表向 agent 添加 ``task``。

    Args:
        backend: 声明式子代理共享的文件 backend；预编译 runnable 原样使用。
        subagents: 至少一个声明式或预编译子代理规格。
        system_prompt: 追加到主代理 system 的派工说明；会自动附上可用类型。
        task_description: 自定义 task 描述，可含 ``{available_agents}``。
    """

    def __init__(self, *, backend=None,
                 subagents: Sequence[_SubAgentSpec],
                 system_prompt: str | None = None,
                 task_description: str | None = None) -> None:
        _validate_subagents(subagents)
        self._backend = backend
        self._subagents = tuple(subagents)
        self.subagent_names = frozenset(spec["name"] for spec in subagents)
        self.tools = (_build_task_tool(
            self._subagents, task_description, backend=backend),)
        if system_prompt:
            agents = "\n".join(
                f"- {spec['name']}: {spec['description']}"
                for spec in subagents)
            self.system_prompt = (
                f"{system_prompt}\n\nAvailable subagent types:\n\n{agents}")
        else:
            self.system_prompt = system_prompt

    def modify_request(self, request: ModelRequest) -> ModelRequest:
        if not self.system_prompt:
            return request
        system = (f"{request.system}\n\n{self.system_prompt}"
                  if request.system else self.system_prompt)
        return replace(request, system=system)

    def wrap_model_call(self, request: ModelRequest, handler):
        return handler(self.modify_request(request))


__all__ = [
    "CompiledSubAgent",
    "DEFAULT_GENERAL_PURPOSE_DESCRIPTION",
    "DEFAULT_SUBAGENT_PROMPT",
    "GENERAL_PURPOSE_SUBAGENT",
    "SubAgent",
    "SubAgentMiddleware",
    "TASK_TOOL_DESCRIPTION",
    "create_sub_agent",
]
