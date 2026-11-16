"""FilesystemBackend：虚拟路径、安全边界、读写分页、搜索与中间件复用。"""

from __future__ import annotations

import asyncio
import base64

import pytest

from deep_agent.backends import (
    FILE_NOT_FOUND,
    INVALID_PATH,
    IS_DIRECTORY,
    FilesystemBackend,
    ReadResult,
)
from deep_agent.backends.utils import EMPTY_CONTENT_WARNING
from deep_agent.middleware import ModelRequest, SkillsMiddleware, SummarizationMiddleware


@pytest.fixture
def root(tmp_path):
    path = tmp_path / "root"
    path.mkdir()
    return path


@pytest.fixture
def backend(root):
    return FilesystemBackend(root)


def _write(root, relative, content="content"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


def test_virtual_paths_are_anchored_to_root_not_host(backend, root):
    result = backend.write("/etc/example.txt", "virtual")
    assert result.error is None
    assert (root / "etc/example.txt").read_text() == "virtual"
    assert backend.read("etc/example.txt").file_data["content"] == "virtual"
    assert backend.download_files(["/etc/example.txt"])[0].content == b"virtual"


def test_virtual_mode_rejects_traversal_without_touching_outside(backend, root, tmp_path):
    outside = _write(tmp_path, "outside.txt", "secret")
    for operation in (
        lambda: backend.read("../outside.txt"),
        lambda: backend.write("../outside.txt", "changed"),
        lambda: backend.edit("../outside.txt", "secret", "changed"),
        lambda: backend.delete("../outside.txt"),
    ):
        result = operation()
        assert result.error is not None
    assert outside.read_text() == "secret"
    assert backend.download_files(["../outside.txt"])[0].error == INVALID_PATH
    assert backend.upload_files([("../outside.txt", b"changed")])[0].error == INVALID_PATH
    # '..' 只有作为路径段才是遍历，普通文件名可以使用两个点。
    assert backend.write("/foo..bar.txt", "ok").error is None


def test_virtual_mode_blocks_symlink_escape(backend, root, tmp_path):
    outside = _write(tmp_path, "secret.txt", "secret")
    try:
        (root / "escape").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("platform does not support symlinks")
    assert backend.read("/escape").error is not None
    assert backend.write("/escape", "changed").error is not None
    assert backend.download_files(["/escape"])[0].error == INVALID_PATH
    assert outside.read_text() == "secret"
    assert all(entry["path"] != "/escape" for entry in backend.ls("/").entries)


def test_nonvirtual_mode_explicitly_allows_absolute_paths(root, tmp_path):
    backend = FilesystemBackend(root, virtual_mode=False)
    target = tmp_path / "absolute.txt"
    result = backend.write(str(target), "absolute")
    assert result.error is None
    assert target.read_text() == "absolute"
    assert backend.download_files([str(target)])[0].content == b"absolute"


def test_ls_returns_sorted_metadata_and_structured_errors(backend, root):
    _write(root, "z.txt", "12345")
    _write(root, "a/child.txt")
    (root / "empty").mkdir()
    result = backend.ls("/")
    assert result.error is None
    assert [entry["path"] for entry in result.entries] == ["/a/", "/empty/", "/z.txt"]
    assert result.entries[0]["is_dir"] is True
    assert result.entries[0]["size"] == 0
    assert result.entries[-1]["size"] == 5
    assert "modified_at" in result.entries[-1]
    assert backend.ls("/empty").entries == []
    assert backend.ls("/missing").entries is None
    assert "path_not_found" in backend.ls("/missing").error
    assert "not_a_directory" in backend.ls("/z.txt").error


def test_read_pagination_preserves_newlines_and_reports_resume_offset(backend, root):
    _write(root, "lines.txt", "a\nb\nc")
    page = backend.read("/lines.txt", offset=1, limit=1)
    assert page.file_data == {"content": "b\n", "encoding": "utf-8"}
    assert (page.total_lines, page.start_line, page.end_line, page.next_offset) == (3, 2, 2, 2)
    last = backend.read("/lines.txt", offset=2, limit=1)
    assert last.file_data["content"] == "c"
    assert last.next_offset is None
    clamped = backend.read("/lines.txt", offset=-10, limit=2)
    assert clamped.file_data["content"] == "a\nb\n"
    empty_window = backend.read("/lines.txt", limit=0)
    assert empty_window.no_lines_requested is True
    assert empty_window.file_data["content"] == ""
    assert "exceeds file length" in backend.read("/lines.txt", offset=3).error


def test_read_empty_binary_and_invalid_utf8(backend, root):
    _write(root, "empty.txt", " \n\t")
    empty = backend.read("/empty.txt")
    assert empty.file_data["content"] == EMPTY_CONTENT_WARNING
    assert empty.total_lines is None

    payload = b"\x89PNG\r\n\x00\xff"
    _write(root, "image.png", payload)
    binary = backend.read("/image.png", offset=100, limit=0)
    assert binary.file_data["encoding"] == "base64"
    assert base64.b64decode(binary.file_data["content"]) == payload

    _write(root, "unknown.bin", b"\xff\xfe")
    assert backend.read("/unknown.bin").error is not None


def test_write_edit_unique_replace_all_and_crlf(backend, root):
    assert backend.write("/nested/a.txt", "same\nsame\n").error is None
    duplicate = backend.edit("/nested/a.txt", "same", "new")
    assert "appears 2 times" in duplicate.error
    assert backend.download_files(["/nested/a.txt"])[0].content == b"same\nsame\n"
    changed = backend.edit("/nested/a.txt", "same", "new", replace_all=True)
    assert changed.occurrences == 2
    assert backend.download_files(["/nested/a.txt"])[0].content == b"new\nnew\n"
    assert "cannot be empty" in backend.edit("/nested/a.txt", "", "x").error
    assert "not found" in backend.edit("/missing", "a", "b").error

    backend.upload_files([("/crlf.txt", b"a\r\nb\r\n")])
    assert backend.edit("/crlf.txt", "a\r\n", "x\r\n").error is None
    assert backend.download_files(["/crlf.txt"])[0].content == b"x\nb\n"


def test_edit_reports_eof_newline_mismatch(backend, root):
    _write(root, "eof.txt", "last line")
    result = backend.edit("/eof.txt", "last line\n", "new\n")
    assert "does not end with a newline" in result.error


def test_delete_file_directory_and_symlink_but_never_virtual_root(backend, root):
    _write(root, "dir/a.txt", "a")
    _write(root, "dir/sub/b.txt", "b")
    assert backend.delete("/dir").error is None
    assert not (root / "dir").exists()

    target = _write(root, "target.txt", "keep")
    try:
        (root / "link").symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("platform does not support symlinks")
    assert backend.delete("/link").error is None
    assert not (root / "link").exists()
    assert target.read_text() == "keep"

    _write(root, "sentinel.txt", "safe")
    assert "refusing" in backend.delete("/").error
    assert (root / "sentinel.txt").read_text() == "safe"


def test_upload_download_batch_order_and_partial_errors(backend, root):
    responses = backend.upload_files([
        ("/a.bin", b"a"), ("/nested/b.bin", b"b"), ("/../bad", b"x")])
    assert [response.path for response in responses] == ["/a.bin", "/nested/b.bin", "/../bad"]
    assert responses[0].error is None and responses[1].error is None
    assert responses[2].error == INVALID_PATH

    downloads = backend.download_files(["/nested/b.bin", "/missing", "/nested"])
    assert downloads[0].content == b"b" and downloads[0].error is None
    assert downloads[1].error == FILE_NOT_FOUND
    assert downloads[2].error == IS_DIRECTORY


def _make_search_tree(root):
    _write(root, "top.py", "alpha top\nliteral a.b\n")
    _write(root, "src/main.py", "alpha main\n")
    _write(root, "src/nested/mod.py", "alpha nested\n")
    _write(root, "src/readme.md", "alpha markdown\n")
    _write(root, ".env", "alpha secret\n")
    _write(root, ".hidden.py", "alpha hidden\n")
    _write(root, ".github/workflows/ci.yml", "alpha ci\n")
    _write(root, ".github/workflows/hidden.py", "alpha hidden tree\n")
    _write(root, "plain.txt", "axb\nbefore\nneedle\nneedle\nlast\n")


def test_glob_basename_anchor_globstar_hidden_and_braces(backend, root):
    _make_search_tree(root)
    def paths(pattern, path=None):
        result = backend.glob(pattern, path)
        assert result.error is None
        return [item["path"] for item in result.matches]

    assert paths("*.py") == [
        "/.github/workflows/hidden.py", "/src/main.py", "/src/nested/mod.py", "/top.py"]
    assert paths("**/*.py") == ["/src/main.py", "/src/nested/mod.py", "/top.py"]
    assert paths("/*.py") == ["/top.py"]
    assert paths("src/**/*.py") == ["/src/main.py", "/src/nested/mod.py"]
    assert paths("*.yml") == ["/.github/workflows/ci.yml"]
    assert paths("**/*.yml") == []
    assert paths(".env") == ["/.env"]
    assert paths("src/*.{py,md}") == ["/src/main.py", "/src/readme.md"]
    assert paths("*.py", "/src") == ["/src/main.py", "/src/nested/mod.py"]
    assert backend.glob("../*.py").matches is None
    assert "traversal" in backend.glob("../*.py").error


def test_grep_literal_glob_cap_and_missing_paths(backend, root):
    _make_search_tree(root)
    literal = backend.grep("a.b", "/")
    assert [(m["path"], m["line"]) for m in literal.matches] == [("/top.py", 2)]
    assert backend.grep("alpha", "/", glob="*.yml").matches[0]["path"] == "/.github/workflows/ci.yml"
    assert backend.grep("alpha", "/", glob="**/*.yml").matches == []

    capped = backend.grep("alpha", "/", max_count=1)
    assert len(capped.matches) == 1 and capped.truncated is True
    zero = backend.grep("alpha", "/", max_count=0)
    assert zero.matches == [] and zero.truncated is True
    exact = backend.grep("a.b", "/", max_count=1)
    assert len(exact.matches) == 1 and exact.truncated is False
    assert backend.grep("anything", "/missing").matches == []
    assert "traversal" in backend.grep("x", "/", glob="../*.py").error


def test_grep_context_excludes_other_match_lines(backend, root):
    _make_search_tree(root)
    result = backend.grep("needle", "/plain.txt", context_lines=1)
    first, second = result.matches
    assert first["context_before"] == [{"line": 2, "text": "before"}]
    assert first["context_after"] == []         # line 4 也是 match，不重复当 context
    assert second["context_before"] == []
    assert second["context_after"] == [{"line": 5, "text": "last"}]
    with pytest.raises(ValueError, match="non-negative"):
        backend.grep("x", context_lines=-1)


def test_grep_skips_binary_and_oversized_files(backend, root):
    _write(root, "binary.py", b"valid\n\xff")
    _write(root, "large.py", "needle\n")
    assert backend.grep("valid", "/").matches == []  # 非 UTF-8 当作二进制跳过
    assert len(backend.grep("needle", "/").matches) == 1
    backend.max_file_size_bytes = 2
    assert backend.grep("needle", "/").matches == []


def test_search_time_budget_returns_partial_results(backend, root, monkeypatch):
    import deep_agent.backends.filesystem as filesystem_module

    _make_search_tree(root)
    monotonic = iter(range(0, 100, 2))
    monkeypatch.setattr(filesystem_module.time, "monotonic",
                        lambda: next(monotonic, 100))
    backend.grep_timeout = backend.glob_timeout = 1.0
    grep = backend.grep("alpha", "/")
    assert grep.truncated is True and grep.matches == []
    glob = backend.glob("*.py", "/")
    assert glob.truncated is True and glob.truncation_reason == "budget"


def test_async_wrappers_use_same_semantics(backend):
    async def scenario():
        assert (await backend.awrite("/async.txt", "one\ntwo")).error is None
        read = await backend.aread("/async.txt", 1, 1)
        assert read.file_data["content"] == "two"
        downloads = await backend.adownload_files(["/async.txt"])
        assert downloads[0].content == b"one\ntwo"
        found = await backend.agrep("two", "/")
        assert found.matches[0]["path"] == "/async.txt"
        globbed = await backend.aglob("*.txt", "/")
        assert globbed.matches[0]["path"] == "/async.txt"
    asyncio.run(scenario())


def test_read_result_rejects_inconsistent_pagination():
    with pytest.raises(ValueError, match="together"):
        ReadResult(start_line=1)
    with pytest.raises(ValueError, match="next_offset"):
        ReadResult(start_line=1, end_line=2, total_lines=2, next_offset=1)


def test_skills_and_summarization_consume_real_filesystem_backend(backend):
    backend.write("/skills/research/SKILL.md", """---
name: research
description: Research workflow
---
Full research instructions.
""")
    skills = SkillsMiddleware(backend=backend, sources=["/skills"])
    state = skills.before_agent({})
    assert state["skills_metadata"][0]["path"] == "/skills/research/SKILL.md"
    request = skills.modify_request(ModelRequest(model="m", messages=[], state=state))
    assert "Research workflow" in request.system

    summarization = SummarizationMiddleware(
        backend=backend, trigger=("messages", 2), keep=("messages", 1),
        summarizer=lambda messages: "compact")
    update = summarization.before_model({"messages": [
        {"id": "1", "role": "user", "content": "old"},
        {"id": "2", "role": "user", "content": "latest"},
    ]})
    event = update["summarization_event"]
    downloaded = backend.download_files([event["file_path"]])[0]
    assert downloaded.error is None
    assert b"old" in downloaded.content and b"latest" not in downloaded.content
