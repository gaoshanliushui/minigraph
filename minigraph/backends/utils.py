"""FilesystemBackend 使用的零依赖路径、分页与文本替换工具。"""

from __future__ import annotations

import fnmatch
from functools import lru_cache
from pathlib import PurePosixPath

from .protocol import FileData, ReadResult

EMPTY_CONTENT_WARNING = "System reminder: File exists but has empty contents"
EMPTY_OLD_STRING_ERROR = "Error: old_string cannot be empty. Provide the exact text to replace."
MAX_VIDEO_INPUT_BYTES = 1024 * 1024 * 1024
MAX_GLOB_EXPANSIONS = 64

_BINARY_EXTENSIONS = {
    ".png", ".jpeg", ".jpg", ".webp", ".gif", ".heic", ".heif",
    ".mp4", ".mpeg", ".mov", ".avi", ".flv", ".mpg", ".webm", ".wmv",
    ".3gpp", ".mkv", ".wav", ".mp3", ".aiff", ".aac", ".ogg", ".flac",
    ".pdf", ".ppt", ".pptx",
}
_VIDEO_EXTENSIONS = {
    ".mp4", ".mpeg", ".mov", ".avi", ".flv", ".mpg", ".webm", ".wmv",
    ".3gpp", ".mkv",
}


class InvalidGlobPatternError(ValueError):
    """glob 遍历或展开规模超过安全限制。"""


def is_binary_path(path: str) -> bool:
    return PurePosixPath(path).suffix.lower() in _BINARY_EXTENSIONS


def is_video_path(path: str) -> bool:
    return PurePosixPath(path).suffix.lower() in _VIDEO_EXTENSIONS


def _split_brace_options(text: str) -> list[str] | None:
    options, depth, start = [], 0, 0
    for i, char in enumerate(text):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        elif char == "," and depth == 0:
            options.append(text[start:i])
            start = i + 1
    if not options:
        return None
    options.append(text[start:])
    return options


def _expand_braces(pattern: str) -> list[str]:
    """递归展开 {a,b}；不把没有逗号的括号当特殊语法。"""
    search_from = 0
    while True:
        start = pattern.find("{", search_from)
        if start < 0:
            return [pattern]
        depth, end = 0, -1
        for i in range(start, len(pattern)):
            if pattern[i] == "{":
                depth += 1
            elif pattern[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end < 0:
            return [pattern]
        options = _split_brace_options(pattern[start + 1:end])
        if options is None:
            search_from = end + 1
            continue
        suffixes = _expand_braces(pattern[end + 1:])
        expanded = []
        for option in options:
            for prefix_part in _expand_braces(pattern[:start] + option):
                for suffix in suffixes:
                    expanded.append(prefix_part + suffix)
                    if len(expanded) > MAX_GLOB_EXPANSIONS:
                        raise InvalidGlobPatternError(
                            f"Glob brace expansion exceeds {MAX_GLOB_EXPANSIONS} patterns")
        return expanded


def _segment_matches(pattern: str, name: str) -> bool:
    if name.startswith(".") and not pattern.startswith("."):
        return False
    return fnmatch.fnmatchcase(name, pattern)


@lru_cache(maxsize=8192)
def _path_segments_match(pattern_parts: tuple[str, ...], path_parts: tuple[str, ...]) -> bool:
    if not pattern_parts:
        return not path_parts
    head, *tail = pattern_parts
    if head == "**":
        # ** 可以匹配零层；消费目录/文件段时不进入隐藏段。
        if _path_segments_match(tuple(tail), path_parts):
            return True
        return bool(path_parts and not path_parts[0].startswith(".")
                    and _path_segments_match(pattern_parts, path_parts[1:]))
    return bool(path_parts and _segment_matches(head, path_parts[0])
                and _path_segments_match(tuple(tail), path_parts[1:]))


@lru_cache(maxsize=256)
def compile_glob(pattern: str):
    """编译共享 glob 规则，返回 ``matcher(search_root_relative_posix_path)``。

    - 不含 ``/``：匹配任意深度的 basename；
    - 含 ``/``：匹配相对搜索根的完整路径，支持 ``**``；
    - 隐藏段要求对应 pattern 段以 ``.`` 开头，``**`` 不进入隐藏目录；
    - 任意展开结果含 ``..`` 路径段都会拒绝。
    """
    if not isinstance(pattern, str):
        raise InvalidGlobPatternError("Glob pattern must be a string")
    normalized = pattern.replace("\\", "/")
    anchored = "/" in normalized
    stripped = normalized.lstrip("/")
    expanded = _expand_braces(stripped)
    for item in expanded:
        if ".." in item.split("/"):
            raise InvalidGlobPatternError(
                f"Path traversal not allowed in glob pattern {pattern!r}")
    alternatives = [tuple(part for part in item.split("/") if part not in {"", "."})
                    for item in expanded]

    def matcher(rel_path: str) -> bool:
        rel_path = rel_path.replace("\\", "/").lstrip("/")
        if anchored:
            path_parts = tuple(part for part in rel_path.split("/") if part)
            return any(_path_segments_match(parts, path_parts) for parts in alternatives)
        basename = PurePosixPath(rel_path).name
        # bare pattern 的 alternatives 每项只有一个段（含 brace 时已展开）。
        return any(len(parts) == 1 and _segment_matches(parts[0], basename)
                   for parts in alternatives)

    return matcher


def slice_read_response(file_data: FileData, offset: int, limit: int) -> ReadResult:
    """按行切片并保留行终止符；offset 从 0 开始，返回行号从 1 开始。"""
    content = file_data["content"]
    offset, limit = max(int(offset), 0), max(int(limit), 0)
    if not content or not content.strip():
        return ReadResult(file_data=FileData(content=content, encoding="utf-8"))
    if limit == 0:
        return ReadResult(file_data=FileData(content="", encoding="utf-8"),
                          no_lines_requested=True)
    lines = content.splitlines(keepends=True)
    if offset >= len(lines):
        return ReadResult(
            error=f"Line offset {offset} exceeds file length ({len(lines)} lines)")
    end = min(offset + limit, len(lines))
    sliced = "".join(lines[offset:end]).replace("\r\n", "\n").replace("\r", "\n")
    return ReadResult(
        file_data=FileData(content=sliced, encoding="utf-8"),
        total_lines=len(lines), start_line=offset + 1, end_line=end,
        next_offset=end if end < len(lines) else None,
    )


def perform_string_replacement(content: str, old_string: str, new_string: str,
                               replace_all: bool = False) -> tuple[str, int] | str:
    if not old_string:
        return EMPTY_OLD_STRING_ERROR
    occurrences = content.count(old_string)
    if occurrences == 0:
        if (old_string.endswith("\n") and len(old_string) > 1
                and content.endswith(old_string.removesuffix("\n"))):
            stripped = old_string.removesuffix("\n")
            count = content.count(stripped)
            if count == 1:
                return ("Error: old_string ends with a newline, but the file does not "
                        "end with a newline. Retry with the trailing newline removed.")
            return (f"Error: removing the trailing newline would produce {count} matches. "
                    "Remove it and add surrounding context so the match is unique.")
        return f"Error: String not found in file: '{old_string}'"
    if occurrences > 1 and not replace_all:
        return (f"Error: String '{old_string}' appears {occurrences} times in file. "
                "Use replace_all=True or provide a more specific string.")
    return content.replace(old_string, new_string), occurrences


__all__ = [
    "EMPTY_CONTENT_WARNING", "EMPTY_OLD_STRING_ERROR", "InvalidGlobPatternError",
    "MAX_VIDEO_INPUT_BYTES", "compile_glob", "is_binary_path", "is_video_path",
    "perform_string_replacement", "slice_read_response",
]
