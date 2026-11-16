"""deepagents 风格 skills / summarization 中间件的离线测试。

backend 和模型都是内存 fake：覆盖元数据解析、缓存与 pin、阈值/切割算法、
历史转存、连续摘要，以及 create_deep_agent 的完整图组装，不依赖网络/API key。
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from minigraph.checkpoint import InMemorySaver
from minigraph.deep_agent import create_deep_agent
from minigraph.middleware import (
    ModelRequest,
    SkillsMiddleware,
    SummarizationMiddleware,
    ToolRequest,
    compute_summarization_defaults,
    create_summarization_middleware,
)


def _skill(name, description="Use this skill", *, body=None, extra=""):
    return (f"---\nname: {name}\ndescription: {description}\n{extra}---\n"
            f"{body if body is not None else f'Full instructions for {name}.'}\n")


class MemoryBackend:
    """同时满足 skills 的 ls/download 与 summarization 的 download/write/edit。"""

    def __init__(self):
        self.files = {}
        self.ls_calls = []
        self.download_calls = []
        self.write_calls = []
        self.edit_calls = []

    def add(self, path, content):
        self.files[path] = content.encode() if isinstance(content, str) else content

    def ls(self, path):
        self.ls_calls.append(path)
        prefix = path.rstrip("/") + "/"
        directories = set()
        for file_path in self.files:
            if not file_path.startswith(prefix):
                continue
            relative = file_path[len(prefix):]
            if "/" in relative:
                directories.add(relative.split("/", 1)[0])
        return [{"path": f"{path.rstrip('/')}/{name}", "is_dir": True}
                for name in sorted(directories)]

    def download_files(self, paths):
        self.download_calls.append(list(paths))
        return [
            {"path": path, "content": self.files[path], "error": None}
            if path in self.files else
            {"path": path, "content": None, "error": "file_not_found"}
            for path in paths
        ]

    def write(self, path, content):
        self.write_calls.append((path, content))
        self.files[path] = content.encode()
        return {"path": path}

    def edit(self, path, old_string, new_string):
        self.edit_calls.append((path, old_string, new_string))
        old = self.files[path].decode()
        if old_string not in old:
            return {"error": "string not found"}
        self.files[path] = old.replace(old_string, new_string, 1).encode()
        return {"path": path}


def _msg(index, content=None):
    return {"id": f"m{index}", "role": "user",
            "content": str(index) if content is None else content}


def _tool_pair():
    return [
        _msg(0, "start"),
        {"id": "a1", "role": "assistant", "content": [
            {"type": "tool_use", "id": "call-1", "name": "search", "input": {}}
        ]},
        {"id": "t1", "role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "call-1", "content": "result"}
        ]},
        {"id": "a2", "role": "assistant", "content": [
            {"type": "text", "text": "intermediate"}
        ]},
        _msg(4, "latest"),
    ]


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------

def test_skills_parse_metadata_override_sources_and_cache():
    backend = MemoryBackend()
    backend.add("/base/skills/research/SKILL.md", _skill(
        "research", "Base description", body="base body"))
    backend.add("/project/skills/research/SKILL.md", _skill(
        "research", "Project description", body="project body", extra="""license: MIT
compatibility: Requires file tools
metadata:
  owner: research-team
  version: '2'
allowed-tools:
  - read_file
  - "write_file"
"""))
    backend.add("/project/skills/planning/SKILL.md", _skill(
        "planning", "Plan carefully"))
    mw = SkillsMiddleware(
        backend=backend,
        sources=["/base/skills", ("/project/skills", "Project override")],
    )

    update = mw.before_agent({"messages": []})
    skills = update["skills_metadata"]
    assert [s["name"] for s in skills] == ["research", "planning"]
    research = skills[0]
    assert research["description"] == "Project description"
    assert research["path"] == "/project/skills/research/SKILL.md"
    assert research["license"] == "MIT"
    assert research["metadata"] == {"owner": "research-team", "version": "2"}
    assert research["allowed_tools"] == ["read_file", "write_file"]
    assert update["skills_load_errors"] == []

    # 缓存以 state 为准；[] 也是“已经加载”，不是每次重新扫描。
    assert mw.before_agent({"skills_metadata": skills}) is None
    assert mw.before_agent({"skills_metadata": []}) is None
    backend.add("/project/skills/new-skill/SKILL.md", _skill("new-skill", "New"))
    reloaded = mw.before_agent({"skills_metadata": None})
    assert {s["name"] for s in reloaded["skills_metadata"]} == {
        "research", "planning", "new-skill"}


def test_skills_prompt_has_metadata_not_body_and_preserves_base_prompt():
    backend = MemoryBackend()
    backend.add("/skills/research/SKILL.md", _skill(
        "research", "Investigate a topic", body="SECRET FULL BODY"))
    mw = SkillsMiddleware(backend=backend, sources=["/skills"])
    state = mw.before_agent({"messages": []})
    req = ModelRequest(model="m", messages=[], system="Base system", state=state)
    captured = []
    result = mw.wrap_model_call(req, lambda request: captured.append(request) or ["ok"])

    assert result == ["ok"]
    system = captured[0].system
    assert system.startswith("Base system\n\n")
    assert "**research**: Investigate a topic" in system
    assert "/skills/research/SKILL.md" in system
    assert "SECRET FULL BODY" not in system
    assert req.system == "Base system"             # 不原地修改调用方 request


def test_skills_pinned_full_text_is_injected_once_without_frontmatter():
    backend = MemoryBackend()
    backend.add("/skills/research/SKILL.md", _skill(
        "research", "Investigate", body="Step 1\nStep 2"))
    mw = SkillsMiddleware(backend=backend, sources=["/skills"])
    state = mw.before_agent({"messages": []})
    state["pinned_skills"] = ["missing", "research", "research"]

    update = mw.before_model(state)
    assert update["pinned_skills"] == []
    assert len(update["messages"]) == 1
    pinned = update["messages"][0]
    assert pinned["role"] == "user"
    assert "Step 1\nStep 2" in pinned["content"]
    assert "name: research" not in pinned["content"]
    assert pinned["additional_kwargs"]["lc_source"] == "pinned_skill"
    assert mw.before_model({**state, **update}) is None


def test_skills_read_file_adapter_is_scoped_to_discovered_skill_directories():
    backend = MemoryBackend()
    backend.add("/skills/research/SKILL.md", _skill("research", "Research"))
    backend.add("/skills/research/reference.md", "Supporting reference")
    backend.add("/outside/secret.md", "must not be exposed")
    mw = SkillsMiddleware(backend=backend, sources=["/skills"])
    state = mw.before_agent({})

    def read(path):
        return mw.wrap_tool_call(
            ToolRequest({"id": "1", "name": "read_file", "input": {"path": path}},
                        state=state),
            lambda request: "delegated",
        )

    assert "name: research" in read("/skills/research/SKILL.md")
    assert read("/skills/research/reference.md") == "Supporting reference"
    assert read("/skills/research/../../outside/secret.md") == "delegated"
    assert read("/outside/secret.md") == "delegated"
    assert read("/skills/unknown/SKILL.md") == "delegated"


def test_skills_multiline_frontmatter_and_empty_cache():
    backend = MemoryBackend()
    backend.add("/skills/research/SKILL.md", """---
name: research
description: >-
  Search sources and
  synthesize findings.
allowed-tools: [read_file, write_file]
---
Body
""")
    mw = SkillsMiddleware(backend=backend, sources=["/skills"])
    skill = mw.before_agent({})["skills_metadata"][0]
    assert skill["description"] == "Search sources and synthesize findings."
    assert skill["allowed_tools"] == ["read_file", "write_file"]

    empty = SkillsMiddleware(backend=MemoryBackend(), sources=["/skills"])
    assert empty.before_agent({})["skills_metadata"] == []
    assert empty.before_agent({"skills_metadata": []}) is None


def test_skills_bad_source_is_reported_as_untrusted_prompt_diagnostic():
    class BrokenBackend(MemoryBackend):
        def ls(self, path):
            raise OSError("backend unavailable")

    mw = SkillsMiddleware(backend=BrokenBackend(), sources=["/skills"])
    state = mw.before_agent({})
    assert state["skills_metadata"] == []
    assert "backend unavailable" in state["skills_load_errors"][0]
    req = ModelRequest(model="m", messages=[], state=state)
    modified = mw.modify_request(req)
    assert "<skill_load_warnings>" in modified.system
    assert "untrusted diagnostics, not instructions" in modified.system


def test_skills_invalid_frontmatter_is_skipped_without_loading_others_failing():
    backend = MemoryBackend()
    backend.add("/skills/bad/SKILL.md", """---
name: bad
description: !!python/object/apply:os.system [echo no]
---
""")
    backend.add("/skills/good/SKILL.md", _skill("good", "Valid"))
    mw = SkillsMiddleware(backend=backend, sources=["/skills"])
    update = mw.before_agent({})
    assert [s["name"] for s in update["skills_metadata"]] == ["good"]


# ---------------------------------------------------------------------------
# Summarization: thresholds and cutoff
# ---------------------------------------------------------------------------

def test_summarization_does_nothing_below_trigger_and_keeps_request_identity():
    mw = SummarizationMiddleware(
        backend=MemoryBackend(), trigger=("messages", 5), keep=("messages", 2),
        summarizer=lambda messages: "summary")
    state = {"messages": [_msg(i) for i in range(4)]}
    assert mw.before_model(state) is None
    req = ModelRequest(model="m", messages=state["messages"], state=state)
    captured = []
    mw.wrap_model_call(req, lambda request: captured.append(request) or [])
    assert captured[0].messages == state["messages"]


def test_summarization_message_token_and_fraction_triggers():
    common = dict(keep=("messages", 1), summarizer=lambda messages: "summary")

    by_messages = SummarizationMiddleware(trigger=("messages", 4), **common)
    assert by_messages.before_model({"messages": [_msg(i) for i in range(3)]}) is None
    assert by_messages.before_model({"messages": [_msg(i) for i in range(4)]})

    counter = lambda messages: len(messages) * 10
    by_tokens = SummarizationMiddleware(
        trigger=("tokens", 35), token_counter=counter, **common)
    assert by_tokens.before_model({"messages": [_msg(i) for i in range(3)]}) is None
    assert by_tokens.before_model({"messages": [_msg(i) for i in range(4)]})

    by_fraction = SummarizationMiddleware(
        model={"profile": {"max_input_tokens": 100}}, trigger=("fraction", 0.5),
        token_counter=counter, **common)
    assert by_fraction.before_model({"messages": [_msg(i) for i in range(4)]}) is None
    assert by_fraction.before_model({"messages": [_msg(i) for i in range(5)]})


def test_summarization_trigger_dict_is_and_list_is_or():
    counter = lambda messages: len(messages) * 10
    both = SummarizationMiddleware(
        trigger={"messages": 4, "tokens": 35}, keep=("messages", 1),
        token_counter=counter, summarizer=lambda messages: "s")
    assert both.before_model({"messages": [_msg(i) for i in range(3)]}) is None
    assert both.before_model({"messages": [_msg(i) for i in range(4)]})

    either = SummarizationMiddleware(
        trigger=[("messages", 5), ("tokens", 25)], keep=("messages", 1),
        token_counter=counter, summarizer=lambda messages: "s")
    assert either.before_model({"messages": [_msg(i) for i in range(3)]})


def test_summarization_safe_cutoff_preserves_tool_use_result_pair():
    backend = MemoryBackend()
    mw = SummarizationMiddleware(
        backend=backend, trigger=("messages", 2), keep=("messages", 3),
        summarizer=lambda messages: "compact")
    raw = _tool_pair()
    original = deepcopy(raw)
    update = mw.before_model({"messages": raw})
    event = update["summarization_event"]

    # keep=3 原会切在 tool_result（index 2），安全切割回退到 assistant（index 1）。
    assert event["cutoff_index"] == 1
    req = ModelRequest(model="m", messages=raw,
                       state={"summarization_event": event})
    effective = mw.modify_request(req).messages
    assert effective[0]["role"] == "user"
    assert "compact" in effective[0]["content"]
    assert effective[1:] == raw[1:]
    assert effective[1]["content"][0]["type"] == "tool_use"
    assert effective[2]["content"][0]["type"] == "tool_result"
    assert raw == original


def test_summarization_token_keep_and_fraction_keep():
    counter = lambda messages: sum(int(m["content"]) for m in messages)
    mw = SummarizationMiddleware(
        trigger=("tokens", 50), keep=("tokens", 50), token_counter=counter,
        summarizer=lambda messages: "s")
    raw = [_msg(i, value) for i, value in enumerate([10, 20, 30, 40])]
    assert mw.before_model({"messages": raw})["summarization_event"]["cutoff_index"] == 3

    fractional = SummarizationMiddleware(
        model={"profile": {"max_input_tokens": 100}}, trigger=("messages", 2),
        keep=("fraction", 0.5), token_counter=counter, summarizer=lambda messages: "s")
    assert fractional.before_model({"messages": raw})["summarization_event"]["cutoff_index"] == 3


def test_summarization_event_applies_summary_plus_tail_without_mutating_raw():
    backend = MemoryBackend()
    mw = SummarizationMiddleware(
        backend=backend, trigger=("messages", 3), keep=("messages", 2),
        summarizer=lambda messages: "important decisions")
    raw = [_msg(i, text) for i, text in enumerate(["zero", "one", "two", "three"])]
    original = deepcopy(raw)
    state = {"messages": raw}
    update = mw.before_model(state)
    event = update["summarization_event"]

    assert event["cutoff_index"] == 2
    assert event["file_path"].startswith("/conversation_history/session_")
    req = ModelRequest(model="m", messages=raw, state={**state, **update})
    effective = mw.modify_request(req).messages
    assert len(effective) == 3
    assert "important decisions" in effective[0]["content"]
    assert event["file_path"] in effective[0]["content"]
    assert [m["content"] for m in effective[1:]] == ["two", "three"]
    assert raw == original
    assert state["messages"] is raw

    history = backend.files[event["file_path"]].decode()
    assert "zero" in history and "one" in history
    assert "two" not in history and "three" not in history


def test_summarization_consecutive_events_use_absolute_raw_cutoff_and_append_history():
    backend = MemoryBackend()
    summaries = []

    def summarize(messages):
        text = f"summary-{len(summaries)}"
        summaries.append(messages)
        return text

    mw = SummarizationMiddleware(
        backend=backend, trigger=("messages", 5), keep=("messages", 2),
        summarizer=summarize)
    raw = [_msg(i, f"raw-{i}") for i in range(8)]
    first = mw.before_model({"messages": raw})
    assert first["summarization_event"]["cutoff_index"] == 6

    # 旧 effective = [summary, raw6, raw7]；新增两条后再次触发。
    more_raw = raw + [_msg(8, "raw-8"), _msg(9, "raw-9")]
    second = mw.before_model({"messages": more_raw, **first})
    event = second["summarization_event"]
    assert event["cutoff_index"] == 8
    assert second["summarization_session_id"] == first["summarization_session_id"]
    assert len(summaries) == 2
    assert summaries[1][0]["additional_kwargs"]["lc_source"] == "summarization"

    # 第二次 offload 会跳过旧摘要，只追加 raw6/raw7，不重复 raw0..raw5。
    history = backend.files[event["file_path"]].decode()
    assert history.count("raw-0") == 1
    assert "raw-6" in history and "raw-7" in history
    assert "raw-8" not in history
    assert len(backend.edit_calls) == 1


def test_summarization_offload_failure_still_compacts_but_does_not_claim_a_file():
    class FailingBackend(MemoryBackend):
        def write(self, path, content):
            return {"error": "disk full"}

    mw = SummarizationMiddleware(
        backend=FailingBackend(), trigger=("messages", 3), keep=("messages", 1),
        summarizer=lambda messages: "summary without file")
    with pytest.warns(UserWarning, match="Offloading conversation history failed"):
        update = mw.before_model({"messages": [_msg(i) for i in range(3)]})
    event = update["summarization_event"]
    assert event["file_path"] is None
    assert "summary without file" in event["summary_message"]["content"]
    assert "conversation_history" not in event["summary_message"]["content"]


def test_summarization_trim_limits_only_summary_input_not_offloaded_history():
    backend = MemoryBackend()
    seen = []
    counter = lambda messages: sum(int(m["content"]) for m in messages)

    def summarize(messages):
        seen.extend(m["content"] for m in messages)
        return "trimmed"

    mw = SummarizationMiddleware(
        backend=backend, trigger=("tokens", 1), keep=("messages", 1),
        token_counter=counter, trim_tokens_to_summarize=90, summarizer=summarize)
    raw = [_msg(i, value) for i, value in enumerate([100, 80, 50, 40])]
    update = mw.before_model({"messages": raw})
    assert seen == ["50"]
    history = backend.files[update["summarization_event"]["file_path"]].decode()
    assert "100" in history and "80" in history and "50" in history


def test_summarization_can_use_model_callable_to_generate_summary():
    requests = []

    def summary_model(request):
        requests.append(request)
        return [{"type": "text", "text": "model generated summary"}]

    mw = SummarizationMiddleware(
        model=summary_model, backend=MemoryBackend(), trigger=("messages", 2),
        keep=("messages", 1))
    update = mw.before_model({"messages": [_msg(0, "hello"), _msg(1, "world")]})
    assert len(requests) == 1
    assert requests[0].messages[0]["role"] == "user"
    assert '<message role="user" id="m0">' in requests[0].messages[0]["content"]
    assert "model generated summary" in update["summarization_event"]["summary_message"]["content"]


def test_summarization_validation_and_profile_defaults():
    with pytest.raises(ValueError, match="fraction"):
        SummarizationMiddleware(trigger=("fraction", 0.8),
                                summarizer=lambda messages: "s")
    with pytest.raises(ValueError, match="正整数"):
        SummarizationMiddleware(trigger=("messages", 0),
                                summarizer=lambda messages: "s")
    with pytest.raises(ValueError, match="trigger clause"):
        SummarizationMiddleware(trigger={}, summarizer=lambda messages: "s")
    with pytest.raises(ValueError, match="messages"):
        SummarizationMiddleware(summary_prompt="missing placeholder",
                                summarizer=lambda messages: "s")
    with pytest.raises(ValueError, match="callable"):
        SummarizationMiddleware()

    assert compute_summarization_defaults(
        {"profile": {"max_input_tokens": 1000}})["trigger"] == ("fraction", 0.85)
    mw = create_summarization_middleware(
        {"profile": {"max_input_tokens": 1000}}, summarizer=lambda messages: "s")
    assert mw.trigger == ("fraction", 0.85)
    assert mw.keep == ("fraction", 0.10)
    assert mw.trim_tokens_to_summarize is None


def test_summarization_malformed_event_falls_back_to_raw_messages():
    raw = [_msg(0)]
    req = ModelRequest(model="m", messages=raw,
                       state={"summarization_event": {"cutoff_index": -1}})
    mw = SummarizationMiddleware(summarizer=lambda messages: "s")
    assert mw.modify_request(req).messages == raw


# ---------------------------------------------------------------------------
# Full graph integration: schema, state, wrap request, checkpoint
# ---------------------------------------------------------------------------

class FakeModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request):
        assert all("id" not in message and "additional_kwargs" not in message
                   for message in request.messages)
        self.requests.append(request)
        return self.responses.pop(0)


def test_deep_agent_read_file_discloses_skill_from_backend():
    backend = MemoryBackend()
    backend.add("/skills/research/SKILL.md", _skill(
        "research", "Research workflows", body="READABLE SKILL BODY"))
    model = FakeModel([
        [{"type": "tool_use", "id": "read-1", "name": "read_file",
          "input": {"path": "/skills/research/SKILL.md"}}],
        [{"type": "text", "text": "done"}],
    ])
    app = create_deep_agent(
        model, middleware=[SkillsMiddleware(backend=backend, sources=["/skills"])])
    out = app.invoke({"messages": [_msg(0, "use the skill")]},
                     config={"thread_id": "skill-read"})
    tool_message = out["messages"][2]
    assert "READABLE SKILL BODY" in tool_message["content"][0]["content"]
    assert out["files"] == {}


def test_deep_agent_integrates_skills_and_summarization_with_checkpoint():
    backend = MemoryBackend()
    backend.add("/skills/research/SKILL.md", _skill(
        "research", "Research workflows", body="FULL SKILL BODY"))
    model = FakeModel([[{"type": "text", "text": "done"}]])
    checkpointer = InMemorySaver()
    app = create_deep_agent(
        model,
        middleware=[
            SkillsMiddleware(backend=backend, sources=["/skills"]),
            SummarizationMiddleware(
                backend=backend, trigger=("messages", 3), keep=("messages", 1),
                summarizer=lambda messages: "compressed context"),
        ],
        checkpointer=checkpointer,
    )
    cfg = {"thread_id": "deepagents-middleware"}
    initial = {"messages": [_msg(0, "first"), _msg(1, "second"), _msg(2, "third")]}
    out = app.invoke(initial, config=cfg)

    assert out["messages"][-1]["content"][0]["text"] == "done"
    # 原始三条 + assistant，一条也没有从 checkpoint 状态里删除。
    assert [m["content"] for m in out["messages"][:3]] == ["first", "second", "third"]
    assert len(out["messages"]) == 4
    assert out["skills_metadata"][0]["name"] == "research"
    assert out["summarization_event"]["cutoff_index"] == 2

    # 真实模型只收到 summary + 最后一条；系统提示只含技能元数据。
    request = model.requests[0]
    assert len(request.messages) == 2
    assert "compressed context" in request.messages[0]["content"]
    assert request.messages[1]["content"] == "third"
    assert "Research workflows" in request.system
    assert "FULL SKILL BODY" not in request.system

    saved = app.get_state(cfg).values
    assert len(saved["messages"]) == 4
    assert saved["summarization_event"] == out["summarization_event"]
    history = backend.files[out["summarization_event"]["file_path"]].decode()
    assert "first" in history and "second" in history and "third" not in history
