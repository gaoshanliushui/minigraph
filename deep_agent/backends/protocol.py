"""可插拔文件 backend 的公共协议与结果类型（deepagents 核心子集）。

所有同步方法都有对应的 ``a*`` 转发，默认通过 ``asyncio.to_thread`` 执行，
不要求 minigraph 引入第三方依赖。文件下载/上传允许批内部分失败，错误码
尽量归一为 ``file_not_found/permission_denied/is_directory/invalid_path``。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict

logger = logging.getLogger(__name__)

DEFAULT_GREP_TIMEOUT = 15
ASYNC_GREP_TIMEOUT = 35

FILE_NOT_FOUND = "file_not_found"
PERMISSION_DENIED = "permission_denied"
IS_DIRECTORY = "is_directory"
INVALID_PATH = "invalid_path"
FileOperationError = Literal[
    "file_not_found", "permission_denied", "is_directory", "invalid_path"]


class FileInfo(TypedDict):
    path: str
    is_dir: NotRequired[bool]
    size: NotRequired[int]
    modified_at: NotRequired[str]


class ContextLine(TypedDict):
    line: int
    text: str


class GrepMatch(TypedDict):
    path: str
    line: int
    text: str
    context_before: NotRequired[list[ContextLine]]
    context_after: NotRequired[list[ContextLine]]


class FileData(TypedDict):
    content: str
    encoding: str
    created_at: NotRequired[str]
    modified_at: NotRequired[str]


@dataclass
class FileDownloadResponse:
    path: str
    content: bytes | None = None
    error: FileOperationError | str | None = None


@dataclass
class FileUploadResponse:
    path: str
    error: FileOperationError | str | None = None


@dataclass
class ReadResult:
    error: str | None = None
    file_data: FileData | None = None
    total_lines: int | None = None
    start_line: int | None = None
    end_line: int | None = None
    next_offset: int | None = None
    no_lines_requested: bool = False

    def __post_init__(self) -> None:
        """防止分页字段组合让调用方跳过未展示的行。"""
        if (self.start_line is None) != (self.end_line is None):
            raise ValueError("start_line and end_line must be set together")
        if self.no_lines_requested and (
                self.error is not None or self.start_line is not None
                or self.next_offset is not None or self.total_lines is not None):
            raise ValueError("no_lines_requested cannot include error/pagination fields")
        if self.next_offset is not None and self.start_line is None:
            raise ValueError("next_offset requires a line window")
        if self.total_lines is not None and self.start_line is None:
            raise ValueError("total_lines requires a line window")
        if self.start_line is not None:
            if self.start_line < 1 or self.end_line < self.start_line:
                raise ValueError("read window must satisfy 1 <= start_line <= end_line")
            if self.total_lines is not None and self.total_lines < self.end_line:
                raise ValueError("total_lines cannot be less than end_line")
            if self.next_offset is not None and self.next_offset != self.end_line:
                raise ValueError("next_offset must equal end_line")


@dataclass
class WriteResult:
    error: str | None = None
    path: str | None = None


@dataclass
class EditResult:
    error: str | None = None
    path: str | None = None
    occurrences: int | None = None


@dataclass
class DeleteResult:
    error: str | None = None
    path: str | None = None


@dataclass
class LsResult:
    error: str | None = None
    entries: list[FileInfo] | None = None


@dataclass
class GrepResult:
    error: str | None = None
    matches: list[GrepMatch] | None = None
    truncated: bool = False


@dataclass
class GlobResult:
    error: str | None = None
    matches: list[FileInfo] | None = None
    truncated: bool = False
    truncation_reason: Literal["budget", "unreadable", "transport"] | None = None


class BackendProtocol:
    """统一的同步文件接口；异步方法默认只是线程转发，不是非阻塞 I/O。"""

    def ls(self, path: str) -> LsResult:
        raise NotImplementedError

    async def als(self, path: str) -> LsResult:
        return await asyncio.to_thread(self.ls, path)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        raise NotImplementedError

    async def aread(self, file_path: str, offset: int = 0,
                    limit: int = 2000) -> ReadResult:
        return await asyncio.to_thread(self.read, file_path, offset, limit)

    def grep(self, pattern: str, path: str | None = None, glob: str | None = None,
             *, max_count: int | None = None, context_lines: int = 0) -> GrepResult:
        raise NotImplementedError

    async def agrep(self, pattern: str, path: str | None = None,
                    glob: str | None = None, *, max_count: int | None = None,
                    context_lines: int = 0) -> GrepResult:
        if context_lines < 0:
            raise ValueError("context_lines must be non-negative")
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self.grep, pattern, path, glob,
                                  max_count=max_count, context_lines=context_lines),
                timeout=ASYNC_GREP_TIMEOUT,
            )
        except TimeoutError:
            return GrepResult(
                error=f"Error: grep timed out after {ASYNC_GREP_TIMEOUT}s. "
                      "Try a more specific pattern or a narrower path.")

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        raise NotImplementedError

    async def aglob(self, pattern: str, path: str | None = None) -> GlobResult:
        return await asyncio.to_thread(self.glob, pattern, path)

    def write(self, file_path: str, content: str) -> WriteResult:
        raise NotImplementedError

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        return await asyncio.to_thread(self.write, file_path, content)

    def edit(self, file_path: str, old_string: str, new_string: str,
             replace_all: bool = False) -> EditResult:
        raise NotImplementedError

    async def aedit(self, file_path: str, old_string: str, new_string: str,
                    replace_all: bool = False) -> EditResult:
        return await asyncio.to_thread(
            self.edit, file_path, old_string, new_string, replace_all)

    def delete(self, file_path: str) -> DeleteResult:
        raise NotImplementedError

    async def adelete(self, file_path: str) -> DeleteResult:
        return await asyncio.to_thread(self.delete, file_path)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        raise NotImplementedError

    async def aupload_files(self, files: list[tuple[str, bytes]],
                            ) -> list[FileUploadResponse]:
        return await asyncio.to_thread(self.upload_files, files)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        raise NotImplementedError

    async def adownload_files(self, paths: list[str],
                              ) -> list[FileDownloadResponse]:
        return await asyncio.to_thread(self.download_files, paths)


def supports_delete(backend: BackendProtocol) -> bool:
    return type(backend).delete is not BackendProtocol.delete


__all__ = [
    "ASYNC_GREP_TIMEOUT", "BackendProtocol", "ContextLine", "DEFAULT_GREP_TIMEOUT",
    "DeleteResult", "EditResult", "FILE_NOT_FOUND", "FileData", "FileDownloadResponse",
    "FileInfo", "FileOperationError", "FileUploadResponse", "GlobResult", "GrepMatch",
    "GrepResult", "INVALID_PATH", "IS_DIRECTORY", "LsResult", "PERMISSION_DENIED",
    "ReadResult", "WriteResult", "supports_delete",
]
