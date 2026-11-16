"""把 :class:`BackendProtocol` 暴露为 agent 可直接调用的文件工具。

中间件负责工具注册和面向模型的文本格式化；路径约束、分页、搜索与错误码由
backend 实现。``create_deep_agent(..., backend=backend)`` 会自动把它加入主
代理和子代理的中间件栈，并用这些工具替换默认的 state 虚拟文件工具。

## 裁剪

对齐 deepagents 的 ``middleware/filesystem.py``：模型上下文有限，一次
``read_file`` / ``grep`` 就可能吐出几十万字符。这里做三层裁剪：

  1. **工具结果超限 -> 转存到 backend**（``tool_token_limit_before_evict``）。
     全文写进 ``/large_tool_results/<tool_call_id>``，给模型看的是
     头尾预览 + 文件路径 —— 模型可以再 ``read_file`` 把全文读回来。
     这是**可恢复**的裁剪。
  2. **用户消息超限 -> 转存到 backend**（``human_message_token_limit_before_evict``）。
     同样落到 ``/conversation_history/``，原始消息仍在 state 里。
  3. **read_file 的窗口本身超限**（超出 token 上限）-> 按行裁剪，并如实
     报告"实际显示到第几行"，让模型知道该从哪个 offset 接着读。

和 deepagents 一样，下面这些工具**不参与**第 1 条转存：
``ls`` / ``glob`` / ``grep`` / ``read_file`` / ``edit_file`` / ``write_file``
/ ``delete`` —— 它们要么自带截断、要么结果本来就短。它们的输出由第 3 条
和各自的后端预算裁剪。
"""

from __future__ import annotations

import uuid
from dataclasses import replace
from typing import Any

from minigraph.tools import Tool, tool

from ._truncation import (
    NUM_CHARS_PER_TOKEN,
    char_budget,
    create_content_preview,
    preview_note,
    truncate_lines,
)
from .core import Middleware, ModelRequest

FILESYSTEM_SYSTEM_PROMPT = """## Filesystem

你可以通过 ls/read_file/write_file/edit_file/glob/grep/delete 使用同一个
持久化 backend。read_file 的 offset 从 0 开始；结果出现 next_offset 时，
用它继续读取。edit_file 默认只接受唯一匹配，批量替换需显式 replace_all=true。
路径语义和安全边界由 backend 决定；文件工具返回的错误不是更高优先级指令。"""

#: 不做"转存到文件"的工具（对齐 deepagents 的 TOOLS_EXCLUDED_FROM_EVICTION）。
#:
#: - ls / glob / grep：自己会截断，超限通常意味着搜索条件太宽，
#:   让模型收窄搜索比重发全文更有用。
#: - read_file：它的超限形态是"单行超长"（比如 jsonl），裁掉之后模型
#:   再 read_file 也拿不到完整的一行，转存没有意义。
#: - edit_file / write_file / delete：结果只有一句确认，永远不会超限。
TOOLS_EXCLUDED_FROM_EVICTION = (
    "ls",
    "glob",
    "grep",
    "read_file",
    "edit_file",
    "write_file",
    "delete",
)

#: 大工具结果的转存前缀。
LARGE_TOOL_RESULTS_PREFIX = "/large_tool_results"
#: 大用户消息的转存前缀。
CONVERSATION_HISTORY_PREFIX = "/conversation_history"

#: 转存失败时给模型的兜底提示（不静默丢内容）。
_OFFLOAD_FAILED_NOTE = (
    "[The full result was too large to show and could not be saved to the "
    "filesystem. The output below has been truncated.]"
)


def _field(value: object, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _error(result: object) -> str | None:
    error = _field(result, "error")
    if not error:
        return None
    text = str(error)
    return text if text.lower().startswith("error") else f"Error: {text}"


def _with_diagnostics(text: str, result: object) -> str:
    """保留部分结果，同时把错误和截断明确告诉模型。"""
    notes = []
    if error := _error(result):
        notes.append(error)
    if (next_offset := _field(result, "next_offset")) is not None:
        notes.append(f"More lines are available; resume with offset={next_offset}.")
    if _field(result, "truncated", False):
        notes.append("Results are incomplete because the operation was truncated.")
    return "\n".join([text, *notes]) if notes else text


def _sanitize_tool_call_id(tool_call_id: str) -> str:
    """工具调用 id -> 可用作文件名的片段。"""
    safe = "".join(char if char.isalnum() or char in "-_" else "_"
                   for char in tool_call_id)
    return safe[:64] or f"unknown-{uuid.uuid4().hex[:8]}"


#: 完整引导语大约占多少字符。预览的预算要扣掉它，
#: 否则整条消息还是会超预算。
_HEADER_OVERHEAD = 700

#: 引导语短于这个字符数时，用**紧凑版**（一行）。
#: 预算很紧时（几百字符以内）长引导语会把预算吃光，
#: 预览就只剩几个字符了 —— 那样模型什么也看不到。
_COMPACT_HEADER_MAX = 400


def _preview_budget(budget: int | None) -> int | None:
    """整条消息的预算 -> 预览本身的预算。"""
    if budget is None:
        return None
    return max(0, budget - _HEADER_OVERHEAD)


def _render_too_large_tool_msg(*, tool_call_id: str, file_path: str,
                               content: str, budget: int | None = None) -> str:
    """大工具结果的替代文本：路径 + 头尾预览。

    整条消息（含引导语）都要塞进预算，所以引导语会在"长版"和
    "紧凑版"之间选，预览本身也要按剩余预算裁。
    """
    budget = budget if budget is not None else _HEADER_OVERHEAD
    if budget <= _COMPACT_HEADER_MAX:
        header = (f"[Tool result too large; saved to {file_path}. "
                  f"Read it back with read_file(offset=0, limit=100).]\n")
        preview = create_content_preview(content, max_chars=max(0, budget - len(header)))
        return header + preview.text

    preview = create_content_preview(content, max_chars=_preview_budget(budget))
    note = preview_note(lines_omitted=preview.lines_omitted,
                        lines_clipped=preview.lines_clipped, subject="result")
    header = (
        f"Tool result too large. The result of this tool call {tool_call_id} "
        f"was saved in the filesystem at this path: {file_path}\n\n"
        "You can read the result from the filesystem by using the read_file "
        "tool, but make sure to only read part of the result at a time.\n\n"
        "You can do this by specifying an offset and limit in the read_file "
        "tool call. For example, to read the first 100 lines, you can use the "
        "read_file tool with offset=0 and limit=100.\n\n"
        f"{note}\n\n"
    )
    return header + preview.text


def _render_too_large_human_msg(*, file_path: str, content: str,
                                budget: int | None = None) -> str:
    """大用户消息的替代文本：路径 + 头尾预览。"""
    budget = budget if budget is not None else _HEADER_OVERHEAD
    if budget <= _COMPACT_HEADER_MAX:
        header = (f"[Message content too large; saved to {file_path}. "
                  f"Read it back with read_file(offset=0, limit=100).]\n")
        preview = create_content_preview(content, max_chars=max(0, budget - len(header)))
        return header + preview.text

    preview = create_content_preview(content, max_chars=_preview_budget(budget))
    note = preview_note(lines_omitted=preview.lines_omitted,
                        lines_clipped=preview.lines_clipped, subject="content")
    header = (
        f"Message content too large and was saved to the filesystem at: "
        f"{file_path}\n\n"
        "You can read the full content using the read_file tool with "
        "pagination (offset and limit parameters).\n\n"
        f"{note}\n\n"
    )
    return header + preview.text


class FilesystemMiddleware(Middleware):
    """提供 backend 驱动的文件工具，并注入简短使用说明。

    Args:
        backend: 文件 backend。
        system_prompt: 追加到系统提示里的文件工具说明。
        tool_token_limit_before_evict: 工具结果的 token 上限。超过就把全文
            转存到 backend，给模型看头尾预览。``None`` 关闭该行为。
        human_message_token_limit_before_evict: 用户消息的 token 上限。
            超过就把全文转存到 backend。``None`` 关闭该行为。
        large_tool_results_prefix: 大工具结果的转存前缀。
        conversation_history_prefix: 大用户消息的转存前缀。
    """

    def __init__(self, *, backend, system_prompt: str | None = FILESYSTEM_SYSTEM_PROMPT,
                 tool_token_limit_before_evict: int | None = 20000,
                 human_message_token_limit_before_evict: int | None = 50000,
                 large_tool_results_prefix: str = LARGE_TOOL_RESULTS_PREFIX,
                 conversation_history_prefix: str = CONVERSATION_HISTORY_PREFIX,
                 ) -> None:
        if backend is None:
            raise ValueError("FilesystemMiddleware requires a backend")
        self.backend = backend
        self.system_prompt = system_prompt
        self.tool_token_limit_before_evict = tool_token_limit_before_evict
        self.human_message_token_limit_before_evict = human_message_token_limit_before_evict
        self.large_tool_results_prefix = "/" + large_tool_results_prefix.strip("/")
        self.conversation_history_prefix = "/" + conversation_history_prefix.strip("/")
        self.tools = self._build_tools()

    # -- 转存 ---------------------------------------------------------------

    def _offload(self, path: str, content: str) -> str | None:
        """把内容写进 backend；成功返回路径，失败返回 None。"""
        try:
            result = self.backend.write(path, content)
        except Exception:
            return None
        if result is None:
            return None
        if _error(result):
            return None
        return _field(result, "path") or path

    def _evict_tool_result(self, request, content: str) -> str:
        """工具结果超限 -> 转存 + 预览；转存失败 -> 硬裁 + 说明。"""
        tool_call_id = str(request.tool_call.get("id") or "unknown")
        path = (f"{self.large_tool_results_prefix}/"
                f"{_sanitize_tool_call_id(tool_call_id)}")
        saved = self._offload(path, content)
        budget = char_budget(self.tool_token_limit_before_evict)
        if saved is None:
            return _OFFLOAD_FAILED_NOTE + "\n" + truncate_lines(content, budget)
        return _render_too_large_tool_msg(
            tool_call_id=tool_call_id, file_path=saved, content=content,
            budget=budget)

    def _evict_human_message(self, message: dict) -> tuple[dict, str | None]:
        """用户消息超限 -> 转存 + 预览。

        返回 (给模型看的消息, 转存路径)。转存失败时原样返回消息
        （内容仍在 state 里，不会丢），只是不裁剪。

        ``content`` 既可能是字符串，也可能是 block 列表（Anthropic 格式）。
        列表形态只处理纯文本块：非文本块（图片等）原样保留，
        文本块被替换成"预览 + 路径"。
        """
        budget = char_budget(self.human_message_token_limit_before_evict)
        if budget is None:
            return message, None

        content = message.get("content")
        if isinstance(content, str):
            if len(content) <= budget:
                return message, None
            path = f"{self.conversation_history_prefix}/{uuid.uuid4().hex}.md"
            saved = self._offload(path, content)
            if saved is None:
                return message, None
            replacement = _render_too_large_human_msg(file_path=saved,
                                                     content=content,
                                                     budget=budget)
            return {**message, "content": replacement}, saved

        if isinstance(content, list):
            texts = [block for block in content
                     if isinstance(block, dict) and block.get("type") == "text"]
            joined = "\n".join(str(block.get("text", "")) for block in texts)
            if len(joined) <= budget or not texts:
                return message, None
            path = f"{self.conversation_history_prefix}/{uuid.uuid4().hex}.md"
            saved = self._offload(path, joined)
            if saved is None:
                return message, None
            replacement = _render_too_large_human_msg(file_path=saved,
                                                     content=joined,
                                                     budget=budget)
            # 第一个文本块换成预览，其余文本块**丢掉** ——
            # 预览已经包含全部文本，留着就是重复一遍。
            # 非文本块（图片等）保留：它们不是被裁剪的对象。
            rebuilt: list[Any] = []
            replaced = False
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "text":
                    rebuilt.append(block)
                    continue
                if replaced:
                    continue                      # 其余文本块丢掉
                rebuilt.append({**block, "text": replacement})
                replaced = True
            return {**message, "content": rebuilt}, saved

        return message, None

    # -- 工具 ---------------------------------------------------------------

    def _build_tools(self) -> tuple[Tool, ...]:
        backend = self.backend
        # read_file 的 token 上限：和工具结果上限同源，
        # 保证"一次 read_file 的输出"永远塞得进一次工具结果。
        read_token_limit = self.tool_token_limit_before_evict
        read_budget = char_budget(read_token_limit)

        @tool(name="ls", description="列出一个目录的直接子文件和子目录。")
        def ls(path: str = "/") -> str:
            """列出目录内容。

            Args:
                path: 要列出的目录路径，默认是 backend 根目录
            """
            result = backend.ls(path)
            entries = _field(result, "entries")
            if entries is None and (error := _error(result)):
                return error
            text = "\n".join(str(entry.get("path", "")) for entry in entries or [])
            return _with_diagnostics(
                truncate_lines(text or "Directory is empty.", read_budget), result)

        @tool(name="read_file", description="读取文件；大文本用 offset/limit 分页。")
        def read_file(path: str, offset: int = 0, limit: int = 2000) -> str:
            """读取文本或 base64 二进制内容。

            Args:
                path: 文件路径
                offset: 从 0 开始的行偏移，只作用于文本
                limit: 最多读取的行数，只作用于文本
            """
            result = backend.read(path, offset=offset, limit=limit)
            if error := _error(result):
                return error
            data = _field(result, "file_data")
            if not data:
                return f"Error: backend returned no file data for {path!r}"
            content = str(data.get("content", ""))
            if data.get("encoding") == "base64":
                content = f"Binary file {path} (base64):\n{content}"
            return _with_diagnostics(
                _clip_read_result(content, result, read_budget), result)

        @tool(name="write_file", description="新建或完整覆盖一个 UTF-8 文本文件。")
        def write_file(path: str, content: str) -> str:
            """写入完整文件内容。

            Args:
                path: 文件路径
                content: 要写入的完整文本
            """
            result = backend.write(path, content)
            return _error(result) or f"Successfully wrote {path}."

        @tool(name="edit_file", description="精确替换文件文本，默认要求唯一匹配。")
        def edit_file(path: str, old_string: str, new_string: str,
                      replace_all: bool = False) -> str:
            """替换文件中的精确文本。

            Args:
                path: 文件路径
                old_string: 要被替换的原文
                new_string: 替换后的文本
                replace_all: 是否替换全部匹配；false 时多个匹配会报错
            """
            result = backend.edit(path, old_string, new_string, replace_all=replace_all)
            if error := _error(result):
                return error
            occurrences = _field(result, "occurrences")
            suffix = f" ({occurrences} replacements)" if occurrences is not None else ""
            return f"Successfully edited {path}{suffix}."

        @tool(name="glob", description="按 glob 模式递归查找文件。")
        def glob_files(pattern: str, path: str = "/") -> str:
            """查找匹配文件。

            Args:
                pattern: glob 模式；不含 / 时匹配任意深度 basename，支持 **
                path: 搜索根目录
            """
            result = backend.glob(pattern, path)
            matches = _field(result, "matches")
            if matches is None and (error := _error(result)):
                return error
            text = "\n".join(str(item.get("path", "")) for item in matches or [])
            return _with_diagnostics(
                truncate_lines(text or "No files found.", read_budget), result)

        @tool(name="grep", description="在文件中做字面量搜索，不使用正则表达式。")
        def grep_files(pattern: str, path: str = "/", glob: str | None = None) -> str:
            """搜索字面量文本。

            Args:
                pattern: 要查找的字面量文本
                path: 搜索目录或单个文件
                glob: 可选的文件 include glob
            """
            result = backend.grep(pattern, path, glob)
            matches = _field(result, "matches")
            if matches is None and (error := _error(result)):
                return error
            text = "\n".join(
                f"{match.get('path', '')}:{match.get('line', 0)}:{match.get('text', '')}"
                for match in matches or [])
            return _with_diagnostics(
                truncate_lines(text or "No matches found.", read_budget), result)

        @tool(name="delete", description="删除文件或递归删除目录；此操作不可撤销。")
        def delete_file(path: str) -> str:
            """删除文件、符号链接或目录。

            Args:
                path: 要删除的路径
            """
            delete = getattr(backend, "delete", None)
            if delete is None:
                return "Error: this backend does not support delete"
            result = delete(path)
            return _error(result) or f"Successfully deleted {path}."

        return (ls, read_file, write_file, edit_file, glob_files, grep_files, delete_file)

    # -- 钩子 ---------------------------------------------------------------

    def wrap_tool_call(self, request, handler):
        """执行工具；结果超限就转存成"预览 + 文件路径"。"""
        result = handler(request)
        if (self.tool_token_limit_before_evict is None
                or request.name in TOOLS_EXCLUDED_FROM_EVICTION):
            return result
        if not isinstance(result, str):
            return result
        budget = char_budget(self.tool_token_limit_before_evict)
        if budget is None or len(result) <= budget:
            return result
        return self._evict_tool_result(request, result)

    def modify_request(self, request: ModelRequest) -> ModelRequest:
        if not self.system_prompt:
            return request
        system = (f"{request.system}\n\n{self.system_prompt}"
                  if request.system else self.system_prompt)
        return replace(request, system=system)

    def _evict_oversized_human_messages(self, request: ModelRequest) -> ModelRequest:
        """把超限的用户消息换成预览视图（state 里的原文不动）。

        只改**送进模型的那份**消息列表：state 里的原文完整保留，
        所以 checkpoint、回放和评估仍然看得到全文。
        """
        if self.human_message_token_limit_before_evict is None:
            return request
        messages = request.messages or []
        rewritten: list[dict] = []
        changed = False
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "user":
                rewritten.append(message)
                continue
            evicted, _path = self._evict_human_message(message)
            if evicted is not message:
                changed = True
            rewritten.append(evicted)
        return replace(request, messages=rewritten) if changed else request

    def wrap_model_call(self, request: ModelRequest, handler):
        request = self.modify_request(request)
        request = self._evict_oversized_human_messages(request)
        return handler(request)


def _clip_read_result(content: str, result: object,
                      budget: int | None) -> str:
    """把 read_file 的结果裁到字符预算内，并如实报告"看到到第几行"。

    关键点：裁剪**只能落在行边界**上，而且裁完之后的 ``next_offset``
    必须指向"最后一条完整显示的行"的下 一行 —— 直接沿用 backend 报的
    ``next_offset`` 会让模型跳过被丢掉的那几行，再也读不到。
    """
    if budget is None or len(content) <= budget:
        return content

    start_line = _field(result, "start_line")
    end_line = _field(result, "end_line")
    next_offset = _field(result, "next_offset")
    if start_line is None or end_line is None:
        # 后端没给窗口信息 —— 无法安全地重算续读位置，
        # 只能硬裁，并说明内容不完整。
        return truncate_lines(content, budget)

    lines = content.splitlines()
    kept: list[str] = []
    size = 0
    for line in lines:
        added = len(line) + 1
        if size + added > budget:
            break
        kept.append(line)
        size += added
    if not kept:
        return truncate_lines(content, budget)

    shown_end = start_line + len(kept) - 1
    omitted = end_line - shown_end
    clipped = "\n".join(kept)
    if omitted > 0:
        clipped += (f"\n... [{omitted} lines truncated] ...\n"
                    f"[Shown lines {start_line}-{shown_end} of {end_line}. "
                    f"Continue with offset={shown_end} to read the rest.]")
    elif next_offset is not None:
        clipped += f"\n[Continue with offset={next_offset} to read the rest.]"
    return clipped


__all__ = [
    "CONVERSATION_HISTORY_PREFIX", "FILESYSTEM_SYSTEM_PROMPT",
    "FilesystemMiddleware", "LARGE_TOOL_RESULTS_PREFIX",
    "NUM_CHARS_PER_TOKEN", "TOOLS_EXCLUDED_FROM_EVICTION",
]
