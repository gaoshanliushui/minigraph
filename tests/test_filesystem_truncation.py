"""FilesystemMiddleware 的大输入/工具内容裁剪。

对齐 deepagents 的 ``middleware/filesystem.py``：
  - 超限的工具结果转存到 backend，给模型看头尾预览 + 文件路径；
  - 超限的用户消息转存到 backend，state 里的原文不动；
  - read_file 的窗口超限时按行裁，并如实报告"看到到第几行"。
"""

from __future__ import annotations

import pytest

from deep_agent.backends import FilesystemBackend
from deep_agent.middleware import ModelRequest, ToolRequest
from deep_agent.middleware.filesystem import (
    FilesystemMiddleware,
    TOOLS_EXCLUDED_FROM_EVICTION,
)
from minigraph.tools import tool
from deep_agent.middleware._truncation import (
    NUM_CHARS_PER_TOKEN,
    create_content_preview,
    truncate_lines,
    truncate_text,
)


@pytest.fixture
def root(tmp_path):
    path = tmp_path / "root"
    path.mkdir()
    return path


@pytest.fixture
def backend(root):
    return FilesystemBackend(root)


def _text(value):
    return {"type": "text", "text": value}


def _call(call_id, name, args):
    return {"type": "tool_use", "id": call_id, "name": name, "input": args}


class FakeModel:
    """按脚本返回固定回复的假模型。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        return self.responses.pop(0)


def _tool(middleware: FilesystemMiddleware, name: str):
    return next(t for t in middleware.tools if t.name == name)


def _run_tool(middleware: FilesystemMiddleware, name: str, args: dict) -> str:
    return _tool(middleware, name).invoke(args)


# ---------------------------------------------------------------------------
# 预览本身
# ---------------------------------------------------------------------------

def test_preview_keeps_head_and_tail_with_a_marker():
    content = "\n".join(f"line{i}" for i in range(1, 21))
    preview = create_content_preview(content)
    assert preview.lines_omitted is True
    assert "line1" in preview.text and "line20" in preview.text
    assert "... [10 lines truncated] ..." in preview.text
    assert "line10" not in preview.text          # 中间确实被丢掉了


def test_preview_of_short_content_has_no_marker():
    preview = create_content_preview("a\nb\nc")
    assert preview.lines_omitted is False
    assert "truncated" not in preview.text


def test_preview_clips_very_long_lines_and_says_so():
    preview = create_content_preview("x" * 5000)
    assert preview.lines_clipped is True
    assert "x" * 1000 in preview.text
    assert "x" * 1001 not in preview.text


def test_truncate_lines_drops_whole_lines_and_marks_them():
    clipped = truncate_lines("a\nb\nc\nd", 5)
    assert clipped.startswith("a\nb")
    assert "... [2 lines truncated] ..." in clipped


def test_truncate_text_is_a_hard_cut_with_marker():
    clipped = truncate_text("abcdefghij", 5)
    assert clipped == "ab..."
    assert truncate_text("abc", 5) == "abc"


# ---------------------------------------------------------------------------
# 工具结果超限 -> 转存
# ---------------------------------------------------------------------------

def test_oversized_tool_result_is_offloaded_with_a_recoverable_preview(backend):
    """超限的工具结果：全文落盘，模型看到预览 + 路径（可以再读回来）。"""
    middleware = FilesystemMiddleware(backend=backend,
                                      tool_token_limit_before_evict=200)
    payload = "\n".join(f"row{i}" for i in range(2000))

    request = ToolRequest({"id": "call-1", "name": "dump",
                           "input": {}})
    result = middleware.wrap_tool_call(request, lambda _req: payload)

    # 结果被裁短了
    assert len(result) < len(payload)
    assert "saved in the filesystem" in result
    assert "/large_tool_results/call-1" in result
    # 头尾都在，中间标了省略
    assert "row0" in result and "row199" in result
    assert "lines truncated" in result

    # 全文确实落盘了 —— 模型可以 read_file 把它读回来
    saved = backend.read("/large_tool_results/call-1")
    assert saved.error is None
    assert saved.file_data["content"] == payload


def test_small_tool_result_is_untouched(backend):
    middleware = FilesystemMiddleware(backend=backend,
                                      tool_token_limit_before_evict=20000)
    request = ToolRequest({"id": "c", "name": "ping", "input": {}})
    assert middleware.wrap_tool_call(request, lambda _r: "pong") == "pong"


def test_excluded_tools_are_never_offloaded(backend):
    """deepagents 的排除名单：这些工具自己会截断，不做转存。"""
    middleware = FilesystemMiddleware(backend=backend,
                                      tool_token_limit_before_evict=1)
    payload = "x" * 5000
    for name in TOOLS_EXCLUDED_FROM_EVICTION:
        request = ToolRequest({"id": f"c-{name}", "name": name, "input": {}})
        result = middleware.wrap_tool_call(request, lambda _r: payload)
        assert result == payload, name


def test_offload_failure_falls_back_to_hard_truncation(backend):
    """写不进去时不能静默丢内容 —— 要裁，但要说明。"""
    middleware = FilesystemMiddleware(backend=backend,
                                      tool_token_limit_before_evict=200)

    def broken_write(path, content):
        raise OSError("disk full")

    backend.write = broken_write
    request = ToolRequest({"id": "c1", "name": "dump", "input": {}})
    result = middleware.wrap_tool_call(request, lambda _r: "y" * 5000)

    assert "could not be saved to the filesystem" in result
    assert "truncated" in result
    assert len(result) < 5000


# ---------------------------------------------------------------------------
# read_file 的窗口裁剪
# ---------------------------------------------------------------------------

def test_read_file_output_over_the_limit_is_clipped_by_line(backend, root):
    """read_file 超限：按行裁，并告诉模型该从哪个 offset 接着读。"""
    (root / "big.txt").write_text("\n".join(f"line{i:03d}" for i in range(100)))
    # 20 token -> 80 字符预算，只装得下几行
    middleware = FilesystemMiddleware(backend=backend,
                                      tool_token_limit_before_evict=20)
    result = _run_tool(middleware, "read_file",
                       {"path": "/big.txt", "offset": 0, "limit": 100})

    assert "lines truncated" in result
    # 续读位置指向"最后一条完整显示的行"的下一行，不是 backend 报的 next_offset
    assert "offset=" in result
    shown = int(result.split("Shown lines ")[1].split("-")[0])
    assert shown == 1
    assert f"Continue with offset={shown}" in result


def test_read_file_clipping_never_skips_unread_lines(backend, root):
    """裁剪后的续读 offset 必须覆盖所有没显示的行。"""
    (root / "big.txt").write_text("\n".join(f"line{i:03d}" for i in range(100)))
    middleware = FilesystemMiddleware(backend=backend,
                                      tool_token_limit_before_evict=20)
    result = _run_tool(middleware, "read_file",
                       {"path": "/big.txt", "offset": 0, "limit": 100})
    shown = int(result.split("Shown lines ")[1].split("-")[0])
    # 从 shown 继续读，读到的是"被裁掉那段的下一行"，没有跳过内容
    tail = _run_tool(middleware, "read_file",
                      {"path": "/big.txt", "offset": shown, "limit": 1})
    assert f"line{shown:03d}" in tail


def test_read_file_within_limit_is_untouched(backend, root):
    (root / "small.txt").write_text("hello\nworld")
    middleware = FilesystemMiddleware(backend=backend,
                                      tool_token_limit_before_evict=20000)
    result = _run_tool(middleware, "read_file", {"path": "/small.txt"})
    assert "hello" in result and "world" in result
    assert "truncated" not in result


def test_grep_and_ls_results_are_clipped_not_silently_cut(backend, root):
    (root / "many.txt").write_text("\n".join("hit" for _ in range(500)))
    middleware = FilesystemMiddleware(backend=backend,
                                      tool_token_limit_before_evict=200)
    grep_result = _run_tool(middleware, "grep", {"pattern": "hit", "path": "/"})
    assert "lines truncated" in grep_result
    assert len(grep_result) < 500 * 10


# ---------------------------------------------------------------------------
# 用户消息超限 -> 转存
# ---------------------------------------------------------------------------

def test_oversized_human_message_is_offloaded_and_state_keeps_the_original(
        backend):
    """送进模型的是预览；state 里的原文不受影响。"""
    middleware = FilesystemMiddleware(backend=backend,
                                      human_message_token_limit_before_evict=200)
    original = "\n".join(f"para{i}" for i in range(2000))
    message = {"id": "h1", "role": "user", "content": original}

    request = ModelRequest(model="m", messages=[message])
    rewritten = middleware._evict_oversized_human_messages(request)

    sent = rewritten.messages[0]["content"]
    assert len(sent) < len(original)
    assert "saved to the filesystem at" in sent
    assert "/conversation_history/" in sent
    assert "para0" in sent and "para1999" in sent

    # 全文落盘了
    saved = backend.read(sent.split("at: ")[1].split("\n")[0])
    assert saved.file_data["content"] == original


def test_small_human_message_is_untouched(backend):
    middleware = FilesystemMiddleware(backend=backend,
                                      human_message_token_limit_before_evict=20000)
    message = {"id": "h1", "role": "user", "content": "hi"}
    request = ModelRequest(model="m", messages=[message])
    rewritten = middleware._evict_oversized_human_messages(request)
    assert rewritten.messages[0]["content"] == "hi"


def test_human_message_in_block_form_keeps_non_text_blocks(backend):
    """图片等非文本块原样保留，只裁文本。"""
    middleware = FilesystemMiddleware(backend=backend,
                                      human_message_token_limit_before_evict=200)
    message = {
        "id": "h1", "role": "user",
        "content": [
            {"type": "text", "text": "x" * 5000},
            {"type": "image", "source": {"data": "..."}},
        ],
    }
    request = ModelRequest(model="m", messages=[message])
    rewritten = middleware._evict_oversized_human_messages(request)
    blocks = rewritten.messages[0]["content"]
    assert blocks[1]["type"] == "image"           # 非文本块保留
    assert len(blocks[0]["text"]) < 5000
    assert "saved to the filesystem at" in blocks[0]["text"]


def test_assistant_and_tool_messages_are_never_evicted(backend):
    middleware = FilesystemMiddleware(backend=backend,
                                      human_message_token_limit_before_evict=200)
    messages = [
        {"id": "a1", "role": "assistant", "content": "x" * 500},
        {"id": "u1", "role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "c1", "content": "y" * 500}]},
    ]
    request = ModelRequest(model="m", messages=messages)
    rewritten = middleware._evict_oversized_human_messages(request)
    assert rewritten.messages[0]["content"] == "x" * 500
    assert rewritten.messages[1]["content"][0]["content"] == "y" * 500


# ---------------------------------------------------------------------------
# 通过模型请求的完整链路
# ---------------------------------------------------------------------------

def test_model_call_uses_the_truncated_messages(backend):
    middleware = FilesystemMiddleware(backend=backend,
                                      human_message_token_limit_before_evict=200)
    seen: list = []

    def handler(request):
        seen.extend(request.messages)
        return []

    middleware.wrap_model_call(
        ModelRequest(model="m", messages=[{"id": "h", "role": "user",
                                            "content": "z" * 5000}]),
        handler)

    assert len(seen[0]["content"]) < 5000
    assert "saved to the filesystem at" in seen[0]["content"]


def test_char_budget_matches_the_token_limit():
    from deep_agent.middleware._truncation import char_budget
    assert char_budget(100) == 100 * NUM_CHARS_PER_TOKEN
    assert char_budget(None) is None
    assert char_budget(0) is None


# ---------------------------------------------------------------------------
# 端到端：模型看到预览 -> 用 read_file 把全文读回来
# ---------------------------------------------------------------------------

def test_agent_can_read_back_the_offloaded_result(backend):
    """超限结果被转存后，模型能用 read_file 把全文读回来。"""
    from deep_agent import create_deep_agent

    @tool
    def dump(n: int) -> str:
        """产出大文本。"""
        return "\n".join(f"line{i:04d}" for i in range(n))

    model = FakeModel([
        [_call("t1", "dump", {"n": 3000})],
        [_call("t2", "read_file",
               {"path": "/large_tool_results/t1", "offset": 0, "limit": 3})],
        [_text("done")],
    ])
    app = create_deep_agent(
        model, tools=[dump], backend=backend,
        middleware=[FilesystemMiddleware(backend=backend,
                                        tool_token_limit_before_evict=200)])
    out = app.invoke({"messages": [{"id": "h1", "role": "user",
                                    "content": "跑"}]},
                     config={"thread_id": "trunc"})

    results = [block
               for message in out["messages"]
               if isinstance(message.get("content"), list)
               for block in message["content"]
               if isinstance(block, dict) and block.get("type") == "tool_result"]

    # 第一次：预览 + 路径，不是全文
    assert "/large_tool_results/t1" in results[0]["content"]
    assert "line2999" in results[0]["content"]
    assert len(results[0]["content"]) < 3000 * 8

    # 全文真的落盘了（read 默认只给 2000 行，所以用 download 看全量）
    on_disk = backend.download_files(["/large_tool_results/t1"])[0]
    assert on_disk.error is None
    assert len(on_disk.content) == len("\n".join(f"line{i:04d}"
                                                 for i in range(3000)))

    # 第二次：模型把全文读回来了
    assert "line0000" in results[1]["content"]
    assert "resume with offset=3" in results[1]["content"]


def test_create_deep_agent_config_threads_the_token_limits(backend):
    """create_deep_agent 的两个限额参数要真的传到中间件上。"""
    from deep_agent import create_deep_agent

    @tool
    def dump(n: int) -> str:
        """产出大文本。"""
        return "\n".join(f"line{i:04d}" for i in range(n))

    model = FakeModel([
        [_call("t1", "dump", {"n": 3000})],
        [_text("done")],
    ])
    app = create_deep_agent(
        model, tools=[dump], backend=backend,
        tool_token_limit_before_evict=200,
        human_message_token_limit_before_evict=200,
    )
    out = app.invoke({"messages": [{"id": "h1", "role": "user",
                                    "content": "跑"}]},
                     config={"thread_id": "cfg"})

    results = [block
               for message in out["messages"]
               if isinstance(message.get("content"), list)
               for block in message["content"]
               if isinstance(block, dict) and block.get("type") == "tool_result"]
    # 限额生效：模型看到的是预览 + 路径，而不是全文
    assert "/large_tool_results/t1" in results[0]["content"]
    assert len(results[0]["content"]) < 3000 * 8


def test_human_message_eviction_runs_inside_a_real_agent_loop(backend):
    """超长的用户消息在真实回路里被换成预览，state 里保留原文。"""
    from deep_agent import create_deep_agent

    long_input = "\n".join(f"para{i}" for i in range(2000))
    model = FakeModel([[_text("ok")]])
    app = create_deep_agent(
        model, backend=backend,
        human_message_token_limit_before_evict=200,
    )
    out = app.invoke({"messages": [{"id": "h1", "role": "user",
                                    "content": long_input}]},
                     config={"thread_id": "human"})

    # 送进模型的是预览
    sent = model.requests[0].messages[0]["content"]
    assert len(sent) < len(long_input)
    assert "saved to the filesystem at" in sent
    assert "para0" in sent and "para1999" in sent

    # state 里原文没被动过
    assert out["messages"][0]["content"] == long_input

    # 全文落在 backend 上，模型能读回来
    path = sent.split("at: ")[1].split("\n")[0]
    assert path.startswith("/conversation_history/")
    saved = backend.read(path)
    assert saved.error is None
    assert saved.file_data["content"] == long_input
