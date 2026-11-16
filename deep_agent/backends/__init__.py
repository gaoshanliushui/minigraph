"""可插拔文件存储 backend。"""

from .filesystem import FilesystemBackend
from .protocol import (
    ASYNC_GREP_TIMEOUT,
    DEFAULT_GREP_TIMEOUT,
    FILE_NOT_FOUND,
    INVALID_PATH,
    IS_DIRECTORY,
    PERMISSION_DENIED,
    BackendProtocol,
    ContextLine,
    DeleteResult,
    EditResult,
    FileData,
    FileDownloadResponse,
    FileInfo,
    FileOperationError,
    FileUploadResponse,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
    supports_delete,
)

__all__ = [
    "ASYNC_GREP_TIMEOUT", "BackendProtocol", "ContextLine", "DEFAULT_GREP_TIMEOUT",
    "DeleteResult", "EditResult", "FILE_NOT_FOUND", "FileData", "FileDownloadResponse",
    "FileInfo", "FileOperationError", "FileUploadResponse", "FilesystemBackend",
    "GlobResult", "GrepMatch", "GrepResult", "INVALID_PATH", "IS_DIRECTORY", "LsResult",
    "PERMISSION_DENIED", "ReadResult", "WriteResult", "supports_delete",
]
