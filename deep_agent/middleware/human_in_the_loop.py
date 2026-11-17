"""
人在环路（对齐 LangChain v1 的 middleware/human_in_the_loop.py）。

被"关卡"拦住的工具调用要先等人批准才执行。等人这件事用
minigraph 的 interrupt() 实现：图挂起，调用方拿到一份"待审清单"，
答复后用 config={"resume": ...} 继续。

四种决定：
    approve  按原样执行
    edit     换成人工改过的调用（可以改名字和参数）
    reject   不执行，给模型一条"被拒了"的结果
    respond  不执行，人工直接代答（工具的"问用户"场景）

两种暂停粒度：
    "batched"   一轮模型输出只挂起一次，待审清单里包含本轮所有被拦的调用；
                 答复是一个 {"decisions": [...]}，按顺序和清单对应
    "per_call"  每个被拦的调用各挂起一次，答复是单个决定

batched 是默认；per_call 的语义更清楚（一次只做一个决定），也不需要
调用方手工配对 interrupt id：同一审批任务按 interrupt() 的调用顺序
记录并逐个重放答复。全部决定收齐后才并行执行工具，避免后续审批导致
已批准工具重复执行。

用法：

    hitl = HumanInTheLoopMiddleware(interrupt_on={"send_email": True})
    app = create_agent(model, tools=[send_email], middleware=[hitl],
                       checkpointer=InMemorySaver())

    cfg = {"thread_id": "t1"}
    app.invoke({"messages": [...]}, cfg)          # 挂起
    app.invoke(None, {"thread_id": "t1",         # 答复
                      "resume": {"decisions": [{"type": "approve"}]}})
"""

from __future__ import annotations

import json
from typing import Any, Literal, TypedDict

from minigraph.context import interrupt
from minigraph.state import State

from .core import Middleware, ToolRequest, hook_config

__all__ = [
    "HumanInTheLoopMiddleware",
    "HumanInTheLoopState",
    "InterruptOnConfig",
    "InterruptMode",
    "DecisionType",
    "Action",
    "ActionRequest",
    "ReviewConfig",
    "HITLRequest",
    "HITLResponse",
    "ToolApprovalRequest",
]


# ---------------------------------------------------------------------------
# 数据形状
# ---------------------------------------------------------------------------

class Action(TypedDict):
    """一次动作：名字 + 参数。"""

    name: str
    args: dict[str, Any]


class ActionRequest(TypedDict):
    """待审动作：名字 + 参数 + 说明。"""

    name: str
    args: dict[str, Any]
    description: str


class ReviewConfig(TypedDict):
    """某个动作的审阅策略。"""

    action_name: str
    allowed_decisions: list[str]


class HITLRequest(TypedDict):
    """挂起时交给调用方的待审清单。"""

    action_requests: list[ActionRequest]
    review_configs: list[ReviewConfig]


class HITLResponse(TypedDict):
    """对 HITLRequest 的答复。"""

    decisions: list[dict[str, Any]]


class ToolApprovalRequest(TypedDict):
    """per_call 模式下单个调用的审批请求。"""

    type: str              # 恒为 "tool_approval"
    tool_call_id: str
    name: str
    args: dict[str, Any]
    description: str


DecisionType = Literal["approve", "edit", "reject", "respond"]

#: 暂停粒度
InterruptMode = Literal["batched", "per_call"]

#: 编辑决定里的"改好的动作"
_EDIT_NOTICE = (
    "Note: a human reviewer replaced this tool call before it ran. The call recorded in "
    "your message is the one you produced, not the one that executed. This was "
    "intentional and authorized. Do not re-issue your original call."
)

#: after_model 记下人工决定的 state 字段，wrap_tool_call 在执行前消费
_DECISIONS_KEY = "hitl_decisions"


class InterruptOnConfig(TypedDict, total=False):
    """某个工具的审阅策略。"""

    allowed_decisions: list[str]
    description: Any                      # str 或 callable(tool_call) -> str
    when: Any                             # callable(tool_call) -> bool


class HumanInTheLoopState(State):
    """按 tool_call id 记录人工决定（after_model 写，wrap_tool_call 读）。"""

    hitl_decisions: dict[str, dict[str, Any]]


def _answer_message(tool_call: dict, decision: dict) -> dict:
    """reject / respond 时给模型的结果（不执行工具）。"""
    if decision["type"] == "respond":
        # 人工代答：工具根本不执行，但结果是"成功"的
        content = decision["message"]
    elif reason := decision.get("message"):
        content = (f"User rejected the tool call for `{tool_call.get('name')}` "
                   f"with reason: {reason}")
    else:
        content = (f"User rejected the tool call for `{tool_call.get('name')}` "
                   f"with id {tool_call.get('id')}. The tool was not executed. "
                   "Do not retry this tool call unless the user explicitly "
                   "requests it.")
    return {
        "type": "tool_result",
        "tool_use_id": tool_call.get("id"),
        "content": content,
        "is_error": decision["type"] != "respond",
    }


# ---------------------------------------------------------------------------
# 中间件
# ---------------------------------------------------------------------------

class HumanInTheLoopMiddleware(Middleware):
    """
    需要人批准才执行工具。

        hitl = HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": True,                             # 全部决定都允许
                "delete_file": {"allowed_decisions": ["approve", "reject"]},
            },
            interrupt_mode="batched",     # 或 "per_call"
        )

    没在 interrupt_on 里出现的工具**自动放行**。

    需要 checkpointer —— interrupt() 的状态要存下来。
    """

    state_schema = HumanInTheLoopState

    def __init__(self, interrupt_on: dict[str, Any], *,
                 description_prefix: str = "Tool execution requires approval",
                 edit_notice: str | None = _EDIT_NOTICE,
                 interrupt_mode: InterruptMode = "batched") -> None:
        if interrupt_mode not in ("batched", "per_call"):
            raise ValueError(
                f"interrupt_mode 只能是 'batched' 或 'per_call'，"
                f"收到 {interrupt_mode!r}")

        resolved: dict[str, InterruptOnConfig] = {}
        for tool_name, tool_config in (interrupt_on or {}).items():
            if isinstance(tool_config, bool):
                if tool_config:
                    resolved[tool_name] = InterruptOnConfig(
                        allowed_decisions=["approve", "edit", "reject", "respond"])
                # False = 自动放行，压根不进表
            elif tool_config.get("allowed_decisions"):
                resolved[tool_name] = tool_config
            else:
                raise ValueError(
                    f"interrupt_on 里 {tool_name!r} 的配置无效："
                    "allowed_decisions 必须是非空列表"
                    f"（比如 ['approve', 'reject']），收到 "
                    f"{tool_config.get('allowed_decisions')!r}")

        self.interrupt_on = resolved
        self.description_prefix = description_prefix
        self.edit_notice = edit_notice
        self.interrupt_mode = interrupt_mode

    # -- 构造待审清单 ------------------------------------------------------

    def _create_action_and_config(self, tool_call: dict,
                                  config: InterruptOnConfig) -> tuple[ActionRequest, ReviewConfig]:
        """一条待审请求 + 它的审阅策略。"""
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("input", {}) or {}

        description_value = config.get("description")
        if callable(description_value):
            description = description_value(tool_call)
        elif description_value is not None:
            description = description_value
        else:
            description = (f"{self.description_prefix}\n\n"
                           f"Tool: {tool_name}\nArgs: {tool_args}")

        action_request: ActionRequest = {
            "name": tool_name, "args": tool_args, "description": description,
        }
        review_config: ReviewConfig = {
            "action_name": tool_name,
            "allowed_decisions": list(config["allowed_decisions"]),
        }
        return action_request, review_config

    def _should_interrupt(self, tool_call: dict, config: InterruptOnConfig) -> bool:
        """`when` 谓词决定这次调用要不要拦。"""
        when = config.get("when")
        if when is None:
            return True
        return bool(when(tool_call))

    @staticmethod
    def _process_decision(decision: dict, tool_call: dict,
                          config: InterruptOnConfig) -> tuple[dict | None, dict | None]:
        """
        处理一个决定。

        返回 (改后的 tool_call 或 None, 给模型的结果或 None)。

        改后的调用为 None = **这次调用不执行**（被拒 / 人工代答）。
        原始 assistant 消息不改写：决定存在 state 里，由 wrap_tool_call
        在工具真正执行前拦截，并返回对应的 tool_result 文本。这样既保留
        tool_use / tool_result 的配对，也不依赖路由器过滤 pending calls。
        """
        allowed = config["allowed_decisions"]
        dtype = decision.get("type")

        if dtype == "approve" and "approve" in allowed:
            return tool_call, None
        if dtype == "edit" and "edit" in allowed:
            # 消息里保留模型原样发出的调用；真正执行时由 wrap_tool_call 换成人工改的
            return tool_call, None
        if dtype in allowed and dtype in ("reject", "respond"):
            # 被拒 / 代答：调用从消息里拿掉，另给一条结果
            return None, _answer_message(tool_call, decision)

        raise ValueError(
            f"意外的人工决定：{decision}。决定类型 {dtype!r} 不在工具 "
            f"{tool_call.get('name')!r} 的允许列表里（允许：{allowed}）")

    # -- 人工审批：after_model 挂起 ----------------------------------------

    def _clear_decisions(self, state) -> dict | None:
        """新一轮没有待审调用时，清掉上一轮的决定，避免 id 复用误伤。"""
        if state.get(_DECISIONS_KEY):
            return {_DECISIONS_KEY: {}}
        return None

    def after_model(self, state) -> dict | None:
        """
        模型返回后，检查它要调的工具里有没有被关卡拦住的。

        batched 模式在这一步一次性挂起；per_call 模式在这里按调用顺序
        逐个挂起。恢复后只把决定按 tool_call id 写进 state。真正的放行、
        替换或拒绝都在 wrap_tool_call 里完成，因此不需要改写 assistant
        消息，也不会绕过工具节点。

        per_call 不在并行工具线程里等待后续审批：ToolNode 没返回前，已经
        完成的子调用结果无法提交；若另一个调用挂起，整个工具节点重跑会把
        已批准的工具再执行一次。先收齐并提交决定，再并行执行，既保留
        一次只问一个的交互，也避免这种重复副作用。
        """
        messages = state.get("messages") or []
        if not messages:
            return self._clear_decisions(state)

        last = messages[-1]
        if last.get("role") != "assistant":
            return self._clear_decisions(state)
        calls = [b for b in (last.get("content") or [])
                 if isinstance(b, dict) and b.get("type") == "tool_use"]
        if not calls:
            return self._clear_decisions(state)

        action_requests: list[ActionRequest] = []
        review_configs: list[ReviewConfig] = []
        gated_calls: list[dict] = []

        for call in calls:
            config = self.interrupt_on.get(call.get("name", ""))
            if config is None or not self._should_interrupt(call, config):
                continue
            if not call.get("id"):
                raise ValueError(
                    f"工具调用 {call.get('name')!r} 没有 id，人工决定没法和它配对。"
                    "确认模型返回了 tool call id")
            action_request, review_config = self._create_action_and_config(call, config)
            action_requests.append(action_request)
            review_configs.append(review_config)
            gated_calls.append(call)

        if not action_requests:
            return self._clear_decisions(state)

        if self.interrupt_mode == "per_call":
            # 同一任务内按 interrupt() 调用序号匹配答复。节点重跑时，前面的
            # interrupt() 重放旧决定，只有第一个未答复的调用会再次挂起。
            recorded: dict[str, dict[str, Any]] = {}
            for call, action_request in zip(gated_calls, action_requests):
                value: ToolApprovalRequest = {
                    "type": "tool_approval",
                    "tool_call_id": call["id"],
                    "name": call.get("name", ""),
                    "args": call.get("input", {}) or {},
                    "description": action_request["description"],
                }
                decision = interrupt(value)
                self._process_decision(
                    decision, call, self.interrupt_on[call["name"]])
                recorded[call["id"]] = decision
            return {_DECISIONS_KEY: recorded}

        # batched：一次挂起，这一行的返回就是给调用方看的待审清单
        hitl_request: HITLRequest = {
            "action_requests": action_requests,
            "review_configs": review_configs,
        }
        response = interrupt(hitl_request)
        decisions = (response or {}).get("decisions") or []

        if len(decisions) != len(gated_calls):
            raise ValueError(
                f"人工决定的数量（{len(decisions)}）和待审调用数"
                f"（{len(gated_calls)}）对不上")

        recorded: dict[str, dict[str, Any]] = {}
        for call, decision in zip(gated_calls, decisions):
            # 在这里就校验决定类型；执行前还会再校验一次，防止 state 被篡改。
            self._process_decision(
                decision, call, self.interrupt_on[call["name"]])
            recorded[call["id"]] = decision
        return {_DECISIONS_KEY: recorded}

    # -- 人工决定的落地 ----------------------------------------------------

    def _reviewer_decision(self, request: ToolRequest) -> dict | None:
        """这次调用有没有人工决定。"""
        tool_call_id = request.tool_call.get("id")
        if not tool_call_id:
            return None
        state = request.state or {}
        decisions = state.get(_DECISIONS_KEY) or {}
        return decisions.get(tool_call_id)

    def _apply_edit(self, request: ToolRequest, executed: Action) -> ToolRequest:
        """把请求指向人工改过的调用。"""
        tool_call = {
            **request.tool_call,
            "name": executed["name"],
            "input": executed["args"],
        }
        return ToolRequest(tool_call, state=request.state)

    def _notice(self, executed: Action) -> str:
        """告诉模型"你原来的调用被人改了，实际跑的是这个"。"""
        # ensure_ascii=False：参数里有中文时不要转成 \uXXXX —— 模型读不懂
        return (f"{self.edit_notice} Executed instead: {executed['name']} with "
                f"arguments {json.dumps(executed['args'], default=str, ensure_ascii=False)}.")

    def _prepend_notice(self, result: Any, executed: Action | None) -> Any:
        """给工具结果前面加一段说明（没被编辑过就原样返回）。"""
        if not self.edit_notice or executed is None:
            return result
        notice = self._notice(executed)

        # 工具返回的是纯文本（minigraph 的工具就返回字符串）
        if isinstance(result, str):
            return f"{notice}\n\n{result}" if result else notice

        if isinstance(result, dict) and result.get("type") == "tool_result":
            content = result.get("content")
            if isinstance(content, str):
                if notice in content:
                    return result
                result = {**result, "content": f"{notice}\n\n{content}" if content else notice}
            else:
                result = {**result, "content": [notice, *(content or [])]}
        return result

    # -- per_call 兼容路径：未接 after_model 时在 wrap_tool_call 挂起 --------

    def _resolve_per_call(self, request: ToolRequest) -> tuple[ToolRequest, Action | None] | dict:
        """
        就一个调用问一次人。

        返回 (要执行的请求, 人工改成的动作) 或 直接给模型的结果。
        """
        tool_call = request.tool_call
        config = self.interrupt_on.get(tool_call.get("name", ""))
        if config is None:
            return request, None
        if not self._should_interrupt(tool_call, config):
            return request, None

        tool_call_id = tool_call.get("id")
        if not tool_call_id:
            raise ValueError(
                f"工具调用 {tool_call.get('name')!r} 没有 id，结果没法和它配对。"
                "确认模型返回了 tool call id")

        action_request, _ = self._create_action_and_config(tool_call, config)
        value: ToolApprovalRequest = {
            "type": "tool_approval",
            "tool_call_id": tool_call_id,
            "name": tool_call.get("name", ""),
            "args": tool_call.get("input", {}) or {},
            "description": action_request["description"],
        }

        decision = interrupt(value)
        _, result = self._process_decision(decision, tool_call, config)
        if result is not None:
            return result

        if decision["type"] == "edit":
            executed = decision["edited_action"]
            return self._apply_edit(request, executed), executed
        return request, None

    def wrap_tool_call(self, request: ToolRequest, handler):
        """执行前过一遍关卡；被拒绝的调用绝不会进入 handler。"""
        decision = self._reviewer_decision(request)
        if decision is None and self.interrupt_mode == "per_call":
            # 兼容没把 after_model 接进图的自定义组装：正常 create_agent /
            # create_deep_agent 已在 after_model 收齐决定，不会走这里。
            resolved = self._resolve_per_call(request)
            if isinstance(resolved, dict):
                # 工具 handler 的返回值最终成为 tool_result.content，故这里
                # 返回文本，而不是再套一层 tool_result dict。
                return resolved["content"]   # reject / respond：不执行
            to_run, executed = resolved
            return self._prepend_notice(handler(to_run), executed)

        if decision is None:
            return handler(request)

        config = self.interrupt_on.get(request.name)
        if config is None:
            return handler(request)
        _, result = self._process_decision(decision, request.tool_call, config)
        if result is not None:
            return result["content"]         # reject / respond：不执行

        executed = None
        if decision["type"] == "edit":
            executed = decision["edited_action"]
            request = self._apply_edit(request, executed)
        return self._prepend_notice(handler(request), executed)
