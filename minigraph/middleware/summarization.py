"""对话摘要与历史转存中间件（对齐 deepagents 的核心摘要语义）。

与 LangChain 直接删除旧消息不同，这里保留 ``state["messages"]`` 的完整原始
历史，只在 state 中记录 ``summarization_event``：

    {
        "cutoff_index": 4,                 # 原始消息列表的绝对下标
        "summary_message": {"role": "user", ...},
        "file_path": "/conversation_history/session_xxx.md",
    }

``before_model`` 判断是否达到阈值、转存旧历史并生成摘要事件；
``wrap_model_call`` 再把本次请求改为 ``summary + messages[cutoff:]``。
因此 checkpoint、回放和评估仍能看到完整消息，而模型只看到压缩后的视图。

支持 ``("messages", N)``、``("tokens", N)``、``("fraction", F)`` 阈值。
trigger 的 dict 内各条件是 AND，list 中各 clause 是 OR；keep 指定摘要后
保留的最近消息数/token 数/上下文比例。切割点不会拆开 tool_use/tool_result。

backend 使用 ``download_files/write/edit`` 协议，历史默认追加到
``/conversation_history/<session_id>.md``。转存失败不会阻止摘要，但会 warning，
且摘要消息不虚构可恢复的文件路径。
"""

from __future__ import annotations

import html
import json
import logging
import uuid
import warnings
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, TypedDict

from .core import Middleware, ModelRequest
from ..state import State

logger = logging.getLogger(__name__)

SUMMARIZATION_EVENT_KEY = "summarization_event"
SUMMARIZATION_SESSION_ID_KEY = "summarization_session_id"
DEFAULT_MESSAGES_TO_KEEP = 20
DEFAULT_TRIM_TOKEN_LIMIT = 4000

DEFAULT_SUMMARY_PROMPT = """<role>Context Extraction Assistant</role>

Extract the most important context needed to continue this conversation.
Do not repeat completed actions. Respond ONLY with the extracted context, using
these sections: SESSION INTENT, SUMMARY, ARTIFACTS, NEXT STEPS. Record important
decisions and their rationale, file paths, unresolved work, and rejected options.

<messages>
{messages}
</messages>"""


class SummarizationEvent(TypedDict):
    cutoff_index: int
    summary_message: dict
    file_path: str | None


class TriggerClause(TypedDict, total=False):
    tokens: int
    messages: int
    fraction: float


class SummarizationState(State):
    summarization_event: SummarizationEvent | None
    summarization_session_id: str | None


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, Mapping) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(json.dumps(block, ensure_ascii=False, default=str))
        return "\n".join(parts)
    return str(content)


def count_tokens_approximately(messages: Iterable[dict]) -> int:
    """零依赖的保守近似：每条消息 4 token 开销，正文约 4 字符/token。"""
    total = 0
    for message in messages:
        text = f"{message.get('role', '')}:{_content_text(message.get('content'))}"
        total += 4 + max(1, (len(text) + 3) // 4)
    return total


def _tool_use_ids(message: Mapping) -> set[str]:
    if message.get("role") != "assistant":
        return set()
    content = message.get("content")
    if not isinstance(content, list):
        return set()
    return {str(block["id"]) for block in content
            if isinstance(block, Mapping) and block.get("type") == "tool_use"
            and block.get("id")}


def _tool_result_ids(message: Mapping) -> set[str]:
    if message.get("role") != "user":
        return set()
    content = message.get("content")
    if not isinstance(content, list):
        return set()
    return {str(block["tool_use_id"]) for block in content
            if isinstance(block, Mapping) and block.get("type") == "tool_result"
            and block.get("tool_use_id")}


def _is_tool_result_message(message: Mapping) -> bool:
    content = message.get("content")
    return (message.get("role") == "user" and isinstance(content, list)
            and any(isinstance(block, Mapping)
                    and block.get("type") == "tool_result" for block in content))


def _format_messages_xml(messages: Iterable[dict]) -> str:
    """给摘要模型和历史文件使用的可读、可转义 XML 文本。"""
    rendered = []
    for index, message in enumerate(messages):
        if (message.get("additional_kwargs") or {}).get("lc_source") == "summarization":
            continue                         # 原始历史已经转存，不重复保存旧摘要
        role = html.escape(str(message.get("role", "unknown")), quote=True)
        msg_id = html.escape(str(message.get("id", f"index-{index}")), quote=True)
        body = html.escape(_content_text(message.get("content")))
        rendered.append(f'<message role="{role}" id="{msg_id}">\n{body}\n</message>')
    return "\n\n".join(rendered)


def _field(value: object, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _profile_limit(model: object) -> int | None:
    if isinstance(model, Mapping):
        profile = model.get("profile", model)
    else:
        profile = getattr(model, "profile", None)
    if not isinstance(profile, Mapping):
        return None
    value = profile.get("max_input_tokens")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _validate_context_size(context, parameter_name: str) -> tuple[str, int | float]:
    if not isinstance(context, tuple) or len(context) != 2:
        raise ValueError(f"{parameter_name} 必须是 (kind, value) 二元组")
    kind, value = context
    if isinstance(value, bool):
        raise ValueError(f"{parameter_name} 数值不能是 bool")
    if kind == "fraction":
        if not isinstance(value, (int, float)) or not 0 < value <= 1:
            raise ValueError(f"fractional {parameter_name} 必须在 (0, 1] 内")
    elif kind in {"tokens", "messages"}:
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{parameter_name} 的 {kind} 必须是正整数")
    else:
        raise ValueError(f"不支持的 context size 类型：{kind!r}")
    return kind, value


def _normalize_trigger(trigger) -> list[TriggerClause]:
    if trigger is None:
        return []
    items = trigger if isinstance(trigger, list) else [trigger]
    clauses = []
    for item in items:
        if isinstance(item, tuple):
            kind, value = _validate_context_size(item, "trigger")
            clauses.append(TriggerClause(**{kind: value}))
        elif isinstance(item, Mapping):
            if not item:
                raise ValueError("trigger clause 不能为空")
            clause = {}
            for kind, value in item.items():
                validated_kind, validated_value = _validate_context_size(
                    (kind, value), "trigger")
                clause[validated_kind] = validated_value
            clauses.append(TriggerClause(**clause))
        else:
            raise TypeError(f"不支持的 trigger 项：{type(item).__name__}")
    return clauses


def compute_summarization_defaults(model) -> dict[str, tuple[str, int | float]]:
    """有模型 profile 用比例阈值，否则采用保守的固定阈值。"""
    if _profile_limit(model) is not None:
        return {"trigger": ("fraction", 0.85), "keep": ("fraction", 0.10)}
    return {"trigger": ("tokens", 170000), "keep": ("messages", 6)}


class SummarizationMiddleware(Middleware):
    """达到上下文阈值时自动摘要，并把被压缩的原始历史转存到 backend。

    ``model`` 可以是与 create_deep_agent 相同的 callable(ModelRequest)，也可以
    只是带 ``profile={"max_input_tokens": ...}`` 的配置对象。若传了
    ``summarizer(messages) -> str``，摘要文本完全由它生成，``model`` 只用于
    fraction 阈值的 profile；否则 ``model`` 必须是 callable。
    """

    state_schema = SummarizationState

    def __init__(self, model=None, *, backend=None, trigger=None,
                 keep=("messages", DEFAULT_MESSAGES_TO_KEEP),
                 token_counter: Callable[[Iterable[dict]], int] = count_tokens_approximately,
                 summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
                 trim_tokens_to_summarize: int | None = DEFAULT_TRIM_TOKEN_LIMIT,
                 summarizer: Callable[[list[dict]], str] | None = None,
                 max_input_tokens: int | None = None,
                 history_path_prefix: str = "/conversation_history") -> None:
        self.model = model
        self._backend = backend
        self.trigger = trigger
        self._trigger_clauses = _normalize_trigger(trigger)
        self.keep = _validate_context_size(keep, "keep")
        self.token_counter = token_counter
        self.summary_prompt = summary_prompt
        if "{messages}" not in summary_prompt:
            raise ValueError("summary_prompt 必须包含 {messages} 占位符")
        if trim_tokens_to_summarize is not None and trim_tokens_to_summarize <= 0:
            raise ValueError("trim_tokens_to_summarize 必须是正整数或 None")
        self.trim_tokens_to_summarize = trim_tokens_to_summarize
        self.summarizer = summarizer
        self._max_input_tokens = max_input_tokens or _profile_limit(model)
        self._history_path_prefix = "/" + history_path_prefix.strip("/")
        if summarizer is None and not callable(model):
            raise ValueError("未提供 summarizer 时，model 必须是 callable(ModelRequest)")
        requires_profile = (self.keep[0] == "fraction" or any(
            "fraction" in clause for clause in self._trigger_clauses))
        if requires_profile and self._max_input_tokens is None:
            raise ValueError("使用 fraction 阈值需要 model profile 或 max_input_tokens")

    # -- 阈值与安全切割 -----------------------------------------------------

    def _should_summarize(self, messages: list[dict], total_tokens: int) -> bool:
        for clause in self._trigger_clauses:
            met = True
            for kind, value in clause.items():
                if kind == "messages":
                    met = len(messages) >= value
                elif kind == "tokens":
                    met = total_tokens >= value
                elif kind == "fraction":
                    threshold = max(1, int(self._max_input_tokens * value))
                    met = total_tokens >= threshold
                if not met:
                    break
            if met:
                return True
        return False

    @staticmethod
    def _find_safe_cutoff_point(messages: list[dict], cutoff: int) -> int:
        if cutoff >= len(messages) or not _is_tool_result_message(messages[cutoff]):
            return cutoff
        wanted: set[str] = set()
        index = cutoff
        while index < len(messages) and _is_tool_result_message(messages[index]):
            wanted.update(_tool_result_ids(messages[index]))
            index += 1
        for i in range(cutoff - 1, -1, -1):
            if wanted & _tool_use_ids(messages[i]):
                return i                       # 连 assistant 的 tool_use 一起摘要
        return index                           # 找不到配对时至少不留孤儿 tool_result

    def _find_safe_cutoff(self, messages: list[dict], keep: int) -> int:
        if len(messages) <= keep:
            return 0
        return self._find_safe_cutoff_point(messages, len(messages) - keep)

    def _find_token_cutoff(self, messages: list[dict]) -> int:
        kind, value = self.keep
        target = (int(self._max_input_tokens * value) if kind == "fraction"
                  else int(value))
        target = max(1, target)
        if self.token_counter(messages) <= target:
            return 0
        kept = 0
        cutoff = len(messages)
        for i in range(len(messages) - 1, -1, -1):
            count = self.token_counter([messages[i]])
            if kept + count > target:
                break
            kept += count
            cutoff = i
        if cutoff >= len(messages):
            cutoff = 0 if len(messages) == 1 else len(messages) - 1
        return self._find_safe_cutoff_point(messages, cutoff)

    def _determine_cutoff_index(self, messages: list[dict]) -> int:
        if self.keep[0] == "messages":
            return self._find_safe_cutoff(messages, self.keep[1])
        return self._find_token_cutoff(messages)

    @staticmethod
    def _partition_messages(messages: list[dict], cutoff: int,
                            ) -> tuple[list[dict], list[dict]]:
        return messages[:cutoff], messages[cutoff:]

    # -- 摘要事件与有效消息 -------------------------------------------------

    @staticmethod
    def _apply_event_to_messages(messages: list[dict],
                                 event: SummarizationEvent | None) -> list[dict]:
        if event is None:
            return list(messages)
        try:
            cutoff = event["cutoff_index"]
            summary = event["summary_message"]
        except (KeyError, TypeError) as exc:
            logger.warning("Malformed summarization_event: %s", exc)
            return list(messages)
        if not isinstance(cutoff, int) or cutoff < 0:
            logger.warning("Malformed summarization cutoff: %r", cutoff)
            return list(messages)
        if cutoff > len(messages):
            logger.warning("Summary cutoff %d exceeds %d messages", cutoff, len(messages))
            return [summary]
        return [summary, *messages[cutoff:]]

    @staticmethod
    def _compute_state_cutoff(previous: SummarizationEvent | None,
                              effective_cutoff: int) -> int:
        if previous is None:
            return effective_cutoff
        previous_cutoff = previous.get("cutoff_index")
        if not isinstance(previous_cutoff, int):
            return effective_cutoff
        # effective[0] 是旧摘要，不对应 raw state 中的一条消息。
        return previous_cutoff + effective_cutoff - 1

    def _get_session_id(self, state) -> str:
        existing = state.get(SUMMARIZATION_SESSION_ID_KEY)
        return existing or f"session_{uuid.uuid4().hex}"

    def _build_summary_message(self, summary: str, file_path: str | None,
                               session_id: str, cutoff: int) -> dict:
        if file_path:
            content = (
                "You are in the middle of a conversation that has been summarized.\n\n"
                f"The full conversation history has been saved to {file_path} "
                "should you need to refer back to it for details.\n\n"
                f"A condensed summary follows:\n\n<summary>\n{summary}\n</summary>")
        else:
            content = f"Here is a summary of the conversation to date:\n\n{summary}"
        return {
            "id": f"summary-{session_id}-{cutoff}",
            "role": "user",
            "content": content,
            "additional_kwargs": {"lc_source": "summarization"},
        }

    # -- 历史转存 -----------------------------------------------------------

    def _download_text(self, path: str) -> str:
        if self._backend is None:
            return ""
        try:
            responses = list(self._backend.download_files([path]))
            if not responses:
                return ""
            response = responses[0]
            if _field(response, "error") is not None:
                return ""
            content = _field(response, "content")
            return bytes(content).decode("utf-8") if content is not None else ""
        except Exception as exc:
            logger.debug("Cannot read existing history %s: %s", path, exc)
            return ""

    def _offload_to_backend(self, messages: list[dict], session_id: str) -> str | None:
        if self._backend is None:
            return None
        path = f"{self._history_path_prefix}/{session_id}.md"
        existing = self._download_text(path)
        timestamp = datetime.now(UTC).isoformat()
        section = f"## Summarized at {timestamp}\n\n{_format_messages_xml(messages)}\n\n"
        combined = existing + section
        try:
            if existing and hasattr(self._backend, "edit"):
                result = self._backend.edit(path, existing, combined)
            else:
                result = self._backend.write(path, combined)
            error = _field(result, "error") if result is not None else None
            if result is None or error:
                raise RuntimeError(error or "backend returned None")
        except Exception as exc:
            logger.warning("Failed to offload conversation history to %s: %s", path, exc)
            return None
        return path

    # -- 生成摘要 -----------------------------------------------------------

    def _trim_for_summary(self, messages: list[dict]) -> list[dict]:
        if self.trim_tokens_to_summarize is None:
            return messages
        kept = 0
        cutoff = len(messages)
        for i in range(len(messages) - 1, -1, -1):
            count = self.token_counter([messages[i]])
            if kept + count > self.trim_tokens_to_summarize:
                break
            kept += count
            cutoff = i
        if cutoff == len(messages):
            cutoff = len(messages) - 1         # 至少给摘要模型最近一条
        cutoff = self._find_safe_cutoff_point(messages, cutoff)
        trimmed = messages[cutoff:]
        return trimmed or messages[-1:]

    @staticmethod
    def _extract_model_text(result: object) -> str:
        if isinstance(result, str):
            return result.strip()
        if isinstance(result, list):
            return "\n".join(
                str(block.get("text", "")) for block in result
                if isinstance(block, Mapping) and block.get("type") == "text").strip()
        text = getattr(result, "text", None)
        if isinstance(text, str):
            return text.strip()
        raise TypeError("摘要模型必须返回 str 或 Anthropic text content blocks")

    def _create_summary(self, messages: list[dict]) -> str:
        if not messages:
            return "No previous conversation history."
        trimmed = self._trim_for_summary(messages)
        if self.summarizer is not None:
            summary = self.summarizer(trimmed)
            if not isinstance(summary, str):
                raise TypeError("summarizer 必须返回 str")
            return summary.strip()
        prompt = self.summary_prompt.format(messages=_format_messages_xml(trimmed)).rstrip()
        request = ModelRequest(
            model=str(getattr(self.model, "model", self.model or "")),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )
        return self._extract_model_text(self.model(request))

    # -- 钩子 ---------------------------------------------------------------

    def before_model(self, state) -> dict | None:
        raw_messages = list(state.get("messages") or [])
        previous = state.get(SUMMARIZATION_EVENT_KEY)
        effective = self._apply_event_to_messages(raw_messages, previous)
        total_tokens = self.token_counter(effective)
        if not self._should_summarize(effective, total_tokens):
            return None
        cutoff = self._determine_cutoff_index(effective)
        if cutoff <= 0:
            return None
        to_summarize, _preserved = self._partition_messages(effective, cutoff)
        session_id = self._get_session_id(state)
        file_path = self._offload_to_backend(to_summarize, session_id)
        if file_path is None:
            warnings.warn(
                "Offloading conversation history failed; older messages remain in "
                "raw state but are not recoverable through a history file.",
                stacklevel=2,
            )
        summary = self._create_summary(to_summarize)
        state_cutoff = self._compute_state_cutoff(previous, cutoff)
        summary_message = self._build_summary_message(
            summary, file_path, session_id, state_cutoff)
        event = SummarizationEvent(
            cutoff_index=state_cutoff,
            summary_message=summary_message,
            file_path=file_path,
        )
        return {SUMMARIZATION_EVENT_KEY: event,
                SUMMARIZATION_SESSION_ID_KEY: session_id}

    def modify_request(self, request: ModelRequest) -> ModelRequest:
        event = (request.state or {}).get(SUMMARIZATION_EVENT_KEY)
        messages = self._apply_event_to_messages(request.messages, event)
        # summary_message 的 id/additional_kwargs 只用于内部识别；它是在
        # call_model 清洗原始消息之后才插入的，必须在这里再走一次协议边界，
        # 否则真实 Anthropic Messages 请求会收到未知字段。
        messages = [
            {k: v for k, v in message.items()
             if k not in {"id", "additional_kwargs"}}
            if (message.get("additional_kwargs") or {}).get("lc_source") == "summarization"
            else message
            for message in messages
        ]
        return replace(request, messages=messages)

    def wrap_model_call(self, request: ModelRequest, handler):
        return handler(self.modify_request(request))


def create_summarization_middleware(model, backend=None, **kwargs,
                                    ) -> SummarizationMiddleware:
    """按模型 profile 选择 deepagents 风格的 trigger/keep 默认值。

    与 deepagents factory 一致，这里默认不裁剪送入摘要模型的旧历史；直接
    构造 ``SummarizationMiddleware`` 则默认裁剪到 4000 token。
    """
    defaults = compute_summarization_defaults(model)
    for key, value in defaults.items():
        kwargs.setdefault(key, value)
    kwargs.setdefault("trim_tokens_to_summarize", None)
    return SummarizationMiddleware(model, backend=backend, **kwargs)


__all__ = [
    "DEFAULT_SUMMARY_PROMPT", "SUMMARIZATION_EVENT_KEY",
    "SUMMARIZATION_SESSION_ID_KEY", "SummarizationEvent", "SummarizationMiddleware",
    "SummarizationState", "TriggerClause", "compute_summarization_defaults",
    "count_tokens_approximately", "create_summarization_middleware",
]
