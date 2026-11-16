"""直接读写本地文件系统的 ``FilesystemBackend``。

默认 ``virtual_mode=True``：``/notes/a.txt`` 表示 ``root_dir/notes/a.txt``，
而不是宿主机的 ``/notes/a.txt``。路径遍历和解析后逃逸 root 的符号链接都会
被拒绝，返回给调用方的路径也保持虚拟形式，不泄露真实 root。

``virtual_mode=False`` 仅用于可信的本地开发：绝对路径按原样访问，相对路径
以 root_dir 为基准，不提供目录边界。这个类提供路径约束，但不是进程沙箱；
不可信工作负载仍应配合 HITL、操作系统权限或真正的沙箱使用。

搜索使用纯 Python 字面量匹配，带文件大小和时间预算。grep/glob 超时返回
已找到的部分结果并设置 ``truncated=True``，不会把部分结果伪装成完整结果。
"""

from __future__ import annotations

import base64
import errno
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path, PurePosixPath

from .protocol import (
    DEFAULT_GREP_TIMEOUT,
    FILE_NOT_FOUND,
    INVALID_PATH,
    IS_DIRECTORY,
    PERMISSION_DENIED,
    BackendProtocol,
    DeleteResult,
    EditResult,
    FileData,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)
from .utils import (
    EMPTY_CONTENT_WARNING,
    MAX_VIDEO_INPUT_BYTES,
    InvalidGlobPatternError,
    compile_glob,
    is_binary_path,
    is_video_path,
    perform_string_replacement,
    slice_read_response,
)

logger = logging.getLogger(__name__)
DEFAULT_GLOB_TIMEOUT = 5


class FilesystemBackend(BackendProtocol):
    """以 root_dir 为虚拟根目录的本地文件 backend。"""

    def __init__(self, root_dir: str | Path | None = None, virtual_mode: bool = True,
                 max_file_size_mb: int = 10, *, grep_timeout: float = DEFAULT_GREP_TIMEOUT,
                 glob_timeout: float = DEFAULT_GLOB_TIMEOUT) -> None:
        if max_file_size_mb <= 0:
            raise ValueError("max_file_size_mb must be positive")
        if grep_timeout <= 0 or glob_timeout <= 0:
            raise ValueError("search timeouts must be positive")
        self.cwd = Path(root_dir).resolve() if root_dir is not None else Path.cwd().resolve()
        self.root_dir = self.cwd
        self.virtual_mode = virtual_mode
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)
        self.grep_timeout = float(grep_timeout)
        self.glob_timeout = float(glob_timeout)

    # -- 路径与元数据 -------------------------------------------------------

    def _resolve_path(self, key: str, *, follow_symlinks: bool = True) -> Path:
        if not isinstance(key, str):
            raise ValueError("path must be a string")
        if self.virtual_mode:
            normalized = key.replace("\\", "/")
            if normalized.startswith("~"):
                raise ValueError("Path traversal not allowed")
            vpath = normalized if normalized.startswith("/") else "/" + normalized
            if ".." in PurePosixPath(vpath).parts:
                raise ValueError("Path traversal not allowed")
            candidate = self.cwd / vpath.lstrip("/")
            full = (candidate.resolve() if follow_symlinks else
                    candidate.parent.resolve() / candidate.name)
            try:
                full.relative_to(self.cwd)
            except ValueError:
                raise ValueError("Path outside root directory") from None
            if follow_symlinks:
                _raise_if_symlink_loop(full)
            return full

        path = Path(key)
        candidate = path if path.is_absolute() else self.cwd / path
        if not follow_symlinks:
            return candidate.parent.resolve() / candidate.name
        # 与参考实现一致：非虚拟模式的绝对路径保持调用方给出的形状。
        full = path if path.is_absolute() else candidate.resolve()
        _raise_if_symlink_loop(full)
        return full

    def _to_virtual_path(self, path: Path) -> str:
        relative = path.resolve().relative_to(self.cwd).as_posix()
        return "/" if relative == "." else "/" + relative

    def _public_path(self, path: Path) -> str:
        return self._to_virtual_path(path) if self.virtual_mode else str(path)

    def _display_path(self, path: Path) -> str:
        try:
            return self._public_path(path)
        except (OSError, RuntimeError, ValueError):
            return path.name or "/" if self.virtual_mode else str(path)

    def _file_info(self, path: Path, is_dir: bool) -> FileInfo:
        public = self._public_path(path)
        if is_dir:
            public = public.rstrip("/") + "/"
        info = FileInfo(path=public, is_dir=is_dir)
        try:
            stat = path.stat()
            info["size"] = 0 if is_dir else int(stat.st_size)
            info["modified_at"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
        except OSError:
            pass
        return info

    def _check_virtual_containment(self, path: Path) -> bool:
        if not self.virtual_mode:
            return True
        try:
            path.resolve().relative_to(self.cwd)
            return True
        except (OSError, RuntimeError, ValueError):
            return False

    # -- 基础读写 -----------------------------------------------------------

    def ls(self, path: str) -> LsResult:
        try:
            directory = self._resolve_path(path)
            if not directory.exists():
                return LsResult(error=f"Path '{path}': path_not_found", entries=None)
            if not directory.is_dir():
                return LsResult(error=f"Path '{path}': not_a_directory", entries=None)
            entries, errors = [], []
            for child in directory.iterdir():
                try:
                    is_file, is_dir = child.is_file(), child.is_dir()
                    if not is_file and not is_dir:
                        continue
                    if not self._check_virtual_containment(child):
                        continue
                    entries.append(self._file_info(child, is_dir))
                except (OSError, RuntimeError, ValueError) as exc:
                    errors.append(f"Cannot inspect '{child.name}': {_safe_error(exc)}")
            entries.sort(key=lambda item: item["path"])
            error = "\n".join(sorted(errors)) if errors else None
            return LsResult(error=error, entries=entries)
        except (OSError, RuntimeError, ValueError) as exc:
            return LsResult(error=f"Cannot list '{path}': {_safe_error(exc)}", entries=None)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        try:
            path = self._resolve_path(file_path)
            if not path.exists() or not path.is_file():
                return ReadResult(error=f"File '{file_path}' not found")
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                if is_binary_path(file_path):
                    if is_video_path(file_path) and os.fstat(fd).st_size > MAX_VIDEO_INPUT_BYTES:
                        return ReadResult(error="Video file exceeds maximum input size")
                    with os.fdopen(fd, "rb") as handle:
                        fd = -1
                        raw = handle.read()
                    return ReadResult(file_data=FileData(
                        content=base64.standard_b64encode(raw).decode("ascii"),
                        encoding="base64"))
                with os.fdopen(fd, "r", encoding="utf-8") as handle:
                    fd = -1
                    content = handle.read()
            finally:
                if fd >= 0:
                    os.close(fd)
            if not content or not content.strip():
                return ReadResult(file_data=FileData(
                    content=EMPTY_CONTENT_WARNING, encoding="utf-8"))
            return slice_read_response(FileData(content=content, encoding="utf-8"),
                                       offset, limit)
        except (OSError, RuntimeError, UnicodeError, ValueError) as exc:
            return ReadResult(error=f"Error reading file '{file_path}': {_safe_error(exc)}")

    def write(self, file_path: str, content: str) -> WriteResult:
        try:
            path = self._resolve_path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(path, flags, 0o644)
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                handle.write(content)
            return WriteResult(path=file_path)
        except (OSError, RuntimeError, UnicodeError, ValueError, TypeError) as exc:
            return WriteResult(error=f"Error writing file '{file_path}': {_safe_error(exc)}")

    def edit(self, file_path: str, old_string: str, new_string: str,
             replace_all: bool = False) -> EditResult:
        try:
            path = self._resolve_path(file_path)
            if not path.exists() or not path.is_file():
                return EditResult(error=f"Error: File '{file_path}' not found")
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(fd, "r", encoding="utf-8") as handle:
                content = handle.read()
            old_string = old_string.replace("\r\n", "\n").replace("\r", "\n")
            new_string = new_string.replace("\r\n", "\n").replace("\r", "\n")
            result = perform_string_replacement(
                content, old_string, new_string, replace_all)
            if isinstance(result, str):
                return EditResult(error=result)
            new_content, occurrences = result
            flags = os.O_WRONLY | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(path, flags)
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                handle.write(new_content)
            return EditResult(path=file_path, occurrences=occurrences)
        except (OSError, RuntimeError, UnicodeError, ValueError, TypeError) as exc:
            return EditResult(error=f"Error editing file '{file_path}': {_safe_error(exc)}")

    def delete(self, file_path: str) -> DeleteResult:
        try:
            # 不 resolve 最后一级：删除符号链接本身，而不是递归删除其目标。
            path = self._resolve_path(file_path, follow_symlinks=False)
            if ((self.virtual_mode and path == self.cwd)
                    or path == Path(path.anchor)):
                return DeleteResult(error="Error: refusing to delete a root directory")
            if not path.exists() and not path.is_symlink():
                return DeleteResult(error=f"Error: '{file_path}' not found")
            if path.is_symlink() or not path.is_dir():
                path.unlink()
            else:
                shutil.rmtree(path)
            return DeleteResult(path=file_path)
        except (OSError, RuntimeError, ValueError) as exc:
            return DeleteResult(error=f"Error deleting '{file_path}': {_safe_error(exc)}")

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        responses = []
        for file_path, content in files:
            try:
                path = self._resolve_path(file_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
                fd = os.open(path, flags, 0o644)
                with os.fdopen(fd, "wb") as handle:
                    handle.write(content)
                responses.append(FileUploadResponse(path=file_path))
            except Exception as exc:
                responses.append(FileUploadResponse(
                    path=file_path, error=_map_exception_to_standard_error(exc)))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        responses = []
        for file_path in paths:
            try:
                path = self._resolve_path(file_path)
                if path.is_dir():
                    responses.append(FileDownloadResponse(
                        path=file_path, error=IS_DIRECTORY))
                    continue
                fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(fd, "rb") as handle:
                    content = handle.read()
                responses.append(FileDownloadResponse(path=file_path, content=content))
            except Exception as exc:
                responses.append(FileDownloadResponse(
                    path=file_path, error=_map_exception_to_standard_error(exc)))
        return responses

    # -- 搜索 ---------------------------------------------------------------

    def _walk_files(self, base: Path, errors: list[str]):
        if base.is_file():
            yield base, base.name
            return
        def onerror(exc):
            errors.append(_safe_error(exc))
        for root, directories, files in os.walk(base, followlinks=False, onerror=onerror):
            directories.sort()
            files.sort()
            root_path = Path(root)
            for name in files:
                path = root_path / name
                yield path, path.relative_to(base).as_posix()

    def grep(self, pattern: str, path: str | None = None, glob: str | None = None,
             *, max_count: int | None = None, context_lines: int = 0) -> GrepResult:
        if context_lines < 0:
            raise ValueError("context_lines must be non-negative")
        if max_count is not None and max_count < 0:
            raise ValueError("max_count must be non-negative or None")
        try:
            include = compile_glob(glob) if glob is not None else None
        except InvalidGlobPatternError as exc:
            return GrepResult(error=str(exc), matches=[])
        search_path = path or "."
        try:
            base = self._resolve_path(search_path)
        except ValueError:
            return GrepResult(matches=[])
        except (OSError, RuntimeError) as exc:
            return GrepResult(error=f"Error searching '{search_path}': {_safe_error(exc)}",
                              matches=[])
        if not base.exists():
            return GrepResult(matches=[])

        deadline = time.monotonic() + self.grep_timeout
        matches: list[GrepMatch] = []
        errors: list[str] = []
        truncated = False
        for file_path, rel_path in self._walk_files(base, errors):
            if time.monotonic() > deadline:
                truncated = True
                break
            try:
                if include is not None and not include(rel_path):
                    continue
                if not file_path.is_file() or not self._check_virtual_containment(file_path):
                    continue
                if file_path.stat().st_size > self.max_file_size_bytes:
                    continue
                public_path = self._public_path(file_path)
            except (OSError, RuntimeError, ValueError) as exc:
                errors.append(f"- {rel_path}: {_safe_error(exc)}")
                continue
            scanned_lines = 0
            try:
                with file_path.open(encoding="utf-8", errors="strict") as handle:
                    for line_number, raw_line in enumerate(handle, 1):
                        scanned_lines = line_number
                        if line_number % 2048 == 0 and time.monotonic() > deadline:
                            truncated = True
                            break
                        if pattern not in raw_line:
                            continue
                        if max_count is not None and len(matches) >= max_count:
                            truncated = True
                            break
                        matches.append(GrepMatch(
                            path=public_path, line=line_number,
                            text=raw_line.rstrip("\r\n")))
                    if truncated:
                        break
            except UnicodeDecodeError as exc:
                # 开头就无法解码视为二进制；读到一半才失败则明确标记部分搜索。
                if scanned_lines:
                    errors.append(f"- {public_path}: {_safe_error(exc)}")
                continue
            except (OSError, RuntimeError) as exc:
                errors.append(f"- {public_path}: {_safe_error(exc)}")

        if context_lines and matches:
            errors.extend(self._attach_grep_context(matches, context_lines, pattern))
        return GrepResult(error="\n".join(errors) or None,
                          matches=matches, truncated=truncated)

    def _attach_grep_context(self, matches: list[GrepMatch], context_lines: int,
                             pattern: str) -> list[str]:
        by_path: dict[str, list[GrepMatch]] = {}
        for match in matches:
            by_path.setdefault(match["path"], []).append(match)
        errors = []
        for file_path, file_matches in by_path.items():
            try:
                resolved = self._resolve_path(file_path)
                lines = resolved.read_text(encoding="utf-8").splitlines()
            except (OSError, RuntimeError, UnicodeError, ValueError) as exc:
                errors.append(f"Could not read context for '{file_path}': {_safe_error(exc)}")
                continue
            for match in file_matches:
                line_number = match["line"]
                before, after = [], []
                for number in range(max(1, line_number - context_lines), line_number):
                    text = lines[number - 1]
                    if pattern not in text:
                        before.append({"line": number, "text": text})
                for number in range(line_number + 1,
                                    min(len(lines), line_number + context_lines) + 1):
                    text = lines[number - 1]
                    if pattern not in text:
                        after.append({"line": number, "text": text})
                match["context_before"] = before
                match["context_after"] = after
        return errors

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        try:
            matches_pattern = compile_glob(pattern)
        except InvalidGlobPatternError as exc:
            return GlobResult(error=str(exc), matches=None)
        display_path = path if path is not None else "<default>"
        try:
            base = self.cwd if path is None else self._resolve_path(path)
            if not base.exists() or not base.is_dir():
                return GlobResult(matches=[])
        except (OSError, RuntimeError, ValueError) as exc:
            return GlobResult(error=f"Error globbing '{display_path}': {_safe_error(exc)}",
                              matches=[])

        deadline = time.monotonic() + self.glob_timeout
        results: list[FileInfo] = []
        errors: list[str] = []
        truncated = False
        for file_path, rel_path in self._walk_files(base, errors):
            if time.monotonic() > deadline:
                truncated = True
                break
            try:
                if not matches_pattern(rel_path) or not file_path.is_file():
                    continue
                if not self._check_virtual_containment(file_path):
                    continue
                results.append(self._file_info(file_path, False))
            except (OSError, RuntimeError, ValueError) as exc:
                errors.append(f"- {rel_path}: {_safe_error(exc)}")
        results.sort(key=lambda item: item["path"])
        return GlobResult(matches=results, error="\n".join(errors) or None,
                          truncated=truncated,
                          truncation_reason="budget" if truncated else None)


def _safe_error(exc: BaseException) -> str:
    """错误诊断不嵌入真实文件名，避免 virtual_mode 泄露 root_dir。"""
    if isinstance(exc, OSError):
        return exc.strerror or type(exc).__name__
    if isinstance(exc, UnicodeError):
        return exc.reason
    return str(exc) or type(exc).__name__


def _map_exception_to_standard_error(exc: Exception) -> str:
    if isinstance(exc, FileNotFoundError):
        return FILE_NOT_FOUND
    if _is_symlink_loop_error(exc) or isinstance(
            exc, (NotADirectoryError, FileExistsError, ValueError)):
        return INVALID_PATH
    if isinstance(exc, PermissionError):
        return PERMISSION_DENIED
    if isinstance(exc, IsADirectoryError):
        return IS_DIRECTORY
    return _safe_error(exc)


def _is_eloop_oserror(exc: BaseException | None) -> bool:
    return isinstance(exc, OSError) and (
        exc.errno == errno.ELOOP or getattr(exc, "winerror", None) == 1921)


def _is_symlink_loop_error(exc: BaseException) -> bool:
    return _is_eloop_oserror(exc) or (
        isinstance(exc, RuntimeError)
        and any(_is_eloop_oserror(item) for item in (exc.__cause__, exc.__context__)))


def _raise_if_symlink_loop(path: Path) -> None:
    """Python 3.13 resolve() 可能保留 loop，再用 stat 恢复 ELOOP 语义。"""
    if not path.is_symlink():
        return
    try:
        path.stat()
    except OSError as exc:
        if _is_eloop_oserror(exc):
            raise


__all__ = ["DEFAULT_GLOB_TIMEOUT", "FilesystemBackend"]
