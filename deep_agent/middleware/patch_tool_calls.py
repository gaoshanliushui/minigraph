"""修复消息历史里没有对应结果的悬空工具调用。

对照 deepagents 的 ``PatchToolCallsMiddleware``：进程在工具执行期间中断、
历史被人工编辑或 checkpoint 恢复后，assistant 的 tool_use 可能永远没有
tool_result。下一次调用模型前必须为它们补上错误结果，否则 Messages API
会拒绝这段不满足配对协议的历史。

minigraph 使用 Anthropic 风格 dict 消息，所以参考实现里的 ToolMessage 在
这里表示为 ``role="user"``、content 含 ``tool_result`` block 的消息。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from minigraph.state import REMOVE_ALL_MESSAGES
from .core import Middleware

_TOOL_CALL_TYPES = {"tool_use", "invalid_tool_call"}


def _tool_calls(message: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if message.get("role") != "assistant":
        return []
    content = message.get("content")
    if not isinstance(content, (list, tuple)):
        return []
    return [block for block in content
            if isinstance(block, Mapping)
            and block.get("type") in _TOOL_CALL_TYPES]


def _answered_ids(messages: list[Mapping[str, Any]]) -> set[Any]:
    """已有结果（包括错误结果）的调用不再补，避免重复 tool_result。"""
    answered = set()
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, (list, tuple)):
            continue
        for block in content:
            if isinstance(block, Mapping) and block.get("type") == "tool_result" \
                    and block.get("tool_use_id") is not None:
                answered.add(block["tool_use_id"])
    return answered


def _missing_calls(message: Mapping[str, Any], answered: set[Any]) -> list[Mapping[str, Any]]:
    missing = []
    for call in _tool_calls(message):
        call_id = call.get("id")
        # 与参考实现一致：没有 id 的 malformed call 无法配对，只能跳过。
        if call_id is not None and call_id not in answered:
            missing.append(call)
    return missing


def _error_result(call: Mapping[str, Any]) -> dict[str, Any]:
    call_id = call["id"]
    name = call.get("name") or "unknown"
    if call.get("type") == "invalid_tool_call":
        content = (f"Tool call {name} with id {call_id} could not be executed - "
                   "arguments were malformed or truncated.")
    else:
        content = (f"Tool call {name} with id {call_id} did not complete - "
                   "no result was recorded. It may have been cancelled or interrupted.")
    return {
        "type": "tool_result",
        "tool_use_id": call_id,
        "content": content,
        "is_error": True,
    }


class PatchToolCallsMiddleware(Middleware):
    """在 agent 启动时补齐悬空 tool_use 的错误 tool_result。

    补丁紧跟在对应 assistant 消息之后；同一 assistant 的多个缺失结果合并到
    一条 user 消息，符合 minigraph 工具节点原本产生的消息形状。原始消息内容
    不被修改。若历史完整或为空，不返回任何状态更新。
    """

    def before_agent(self, state) -> dict[str, Any] | None:
        messages = list(state.get("messages") or [])
        if not messages:
            return None

        answered = _answered_ids(messages)
        if not any(_missing_calls(message, answered) for message in messages):
            return None

        patched: list[dict] = []
        for index, message in enumerate(messages):
            patched.append(message)
            missing = _missing_calls(message, answered)
            if not missing:
                continue
            patched.append({
                # 由位置生成的稳定 id：checkpoint 重放及节点重试保持幂等。
                "id": f"patch-tool-calls-{index}",
                "role": "user",
                "content": [_error_result(call) for call in missing],
            })

        # add_messages 普通写入只能追加或按 id 替换，无法把结果插回 assistant
        # 后面。remove-all + 完整重放才能保持原顺序，同时不原地修改调用方历史。
        remove_all = {"id": REMOVE_ALL_MESSAGES, "role": "remove", "content": ""}
        return {"messages": [remove_all, *patched]}


__all__ = ["PatchToolCallsMiddleware"]
