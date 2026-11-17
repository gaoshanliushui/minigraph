"""
examples/03_real_llm_agent.py 的离线测试。

用 fake 的 anthropic client 替代真 API —— 不需要 key、不需要网络，
但走的还是完整的图回环（agent → tools → agent）和 checkpoint 序列化。
provider 配置解析（.env / 环境变量）也在这里做纯逻辑测试。
"""

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# 只对"真的没装"skip；包内部的 ImportError 必须原样暴露
# （importorskip 会把依赖断裂也吞成 skip，排障时看不到真实 traceback）。
try:
    import anthropic  # noqa: F401
    import dotenv     # noqa: F401
except ModuleNotFoundError as e:
    pytest.skip(f"示例 3 依赖未安装（uv sync --extra llm）：{e}",
                allow_module_level=True)

from minigraph.checkpoint import InMemorySaver
from minigraph.checkpoint_sqlite import SqliteSaver

_SPEC = importlib.util.spec_from_file_location(
    "ex03_real_llm_agent",
    Path(__file__).resolve().parent.parent / "examples" / "03_real_llm_agent.py",
)
ex03 = importlib.util.module_from_spec(_SPEC)
# 必须先注册进 sys.modules 再 exec：State.__init_subclass__ 解析字符串注解时
# 按 cls.__module__ 查 sys.modules 拿全局命名空间；不注册的话
# Annotated[list, DeltaChannel(...)] 求值失败，messages 静默退化成
# last-write-wins（历史不累积）。
sys.modules[_SPEC.name] = ex03
_SPEC.loader.exec_module(ex03)


# ---------------------------------------------------------------------------
# provider 配置解析
# ---------------------------------------------------------------------------

def test_resolve_minimax_defaults():
    env = {"LLM_PROVIDER": "minimax", "MINIMAX_API_KEY": "mm-key"}
    with patch.dict(os.environ, env, clear=True):
        cfg = ex03._resolve(os.environ)
    assert cfg["provider"] == "minimax"
    assert cfg["api_key"] == "mm-key"
    assert cfg["base_url"] == "https://api.minimaxi.com/anthropic"
    # 默认模型跟 PROVIDERS 表走（厂商下线旧型号时只需改表，不动测试）
    assert cfg["model"] == ex03.PROVIDERS["minimax"]["model"]
    assert cfg["auth"] == "bearer"


def test_resolve_volcengine_with_overrides():
    env = {
        "LLM_PROVIDER": "volcengine",
        "ARK_API_KEY": "ark-key",
        "LLM_MODEL": "ep-123456",                       # 推理接入点覆盖默认模型
        "LLM_BASE_URL": "https://proxy.example.com/anthropic",
    }
    with patch.dict(os.environ, env, clear=True):
        cfg = ex03._resolve(os.environ)
    assert cfg["api_key"] == "ark-key"
    assert cfg["model"] == "ep-123456"
    assert cfg["base_url"] == "https://proxy.example.com/anthropic"
    assert cfg["auth"] == "bearer"


def test_resolve_generic_key_and_anthropic_default():
    # 不给 LLM_PROVIDER 时默认 anthropic；LLM_API_KEY 可替代厂商专属变量
    with patch.dict(os.environ, {"LLM_API_KEY": "generic"}, clear=True):
        cfg = ex03._resolve(os.environ)
    assert cfg["provider"] == "anthropic"
    assert cfg["api_key"] == "generic"
    assert cfg["auth"] == "x-api-key"
    assert cfg["base_url"] is None                      # 走 SDK 默认端点


def test_resolve_missing_key_and_unknown_provider():
    with patch.dict(os.environ, {"LLM_PROVIDER": "minimax"}, clear=True):
        with pytest.raises(RuntimeError, match="MINIMAX_API_KEY"):
            ex03._resolve(os.environ)
    with patch.dict(os.environ, {"LLM_PROVIDER": "gpt"}, clear=True):
        with pytest.raises(RuntimeError, match="未知 LLM_PROVIDER"):
            ex03._resolve(os.environ)


# ---------------------------------------------------------------------------
# 图回环（fake client）
# ---------------------------------------------------------------------------

class _Block:
    """模拟 SDK 返回的 pydantic block：示例只用到 .model_dump()。"""

    def __init__(self, **fields):
        self._fields = fields

    def model_dump(self):
        return dict(self._fields)


class _FakeStream:
    """模拟 SDK 的 MessageStream：text_stream 逐段吐文本，get_final_message 汇总。

    形状对齐真 SDK：text_stream 只吐 text block 的文字（tool_use 没有文字）；
    get_final_message 返回带 .content 的对象，content 是带 model_dump 的 block。
    """

    def __init__(self, blocks):
        self._blocks = blocks
        self.text_stream = iter(
            b["text"] for b in blocks if b.get("type") == "text"
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return SimpleNamespace(
            content=[_Block(**b) for b in self._blocks])


class _FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []                     # 每次调用的 kwargs，供断言请求体

    def create(self, **kwargs):
        self.calls.append(kwargs)
        assert self._responses, "fake client 的响应用完了（回环次数超预期）"
        return SimpleNamespace(content=[_Block(**b) for b in self._responses.pop(0)])

    def stream(self, **kwargs):
        # 示例的 _chat_once 现在走流式接口；复用同一份响应脚本。
        self.calls.append(kwargs)
        assert self._responses, "fake client 的响应用完了（回环次数超预期）"
        return _FakeStream(self._responses.pop(0))


class _FakeClient:
    def __init__(self, responses):
        self.messages = _FakeMessages(responses)


def _fake_llm(responses) -> SimpleNamespace:
    """patch ex03._get_llm 的替身：fake client + 固定 model/provider。"""
    return SimpleNamespace(client=_FakeClient(responses),
                           model="test-model", provider="fake")


_TOOL_CALL = {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
              "input": {"city": "北京"}}

_TWO_ROUNDS = [
    [{"type": "text", "text": "我查一下"}, _TOOL_CALL],
    [{"type": "text", "text": "北京今天晴，18°C"}],
]

_INPUT = {"messages": [{"id": "h1", "role": "user", "content": "北京天气怎么样"}]}


def test_react_loop_with_fake_client():
    llm = _fake_llm(_TWO_ROUNDS)
    with patch.object(ex03, "_get_llm", return_value=llm):
        app = ex03.build(InMemorySaver())
        out = app.invoke(dict(_INPUT), config={"thread_id": "t"})

    msgs = out["messages"]
    # user → assistant(tool_use) → user(tool_result) → assistant(text)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[1]["content"][1]["type"] == "tool_use"
    assert msgs[2]["content"][0]["type"] == "tool_result"
    assert msgs[2]["content"][0]["tool_use_id"] == "toolu_1"
    assert "晴" in msgs[2]["content"][0]["content"]
    assert msgs[3]["content"][0]["text"] == "北京今天晴，18°C"

    # 第一次调用：只有 user 消息，消息级 id 被剥掉，模型/工具 schema 都带上了
    first = llm.client.messages.calls[0]
    assert first["model"] == "test-model"
    assert first["tools"][0]["name"] == "get_weather"
    assert first["messages"] == [{"role": "user", "content": "北京天气怎么样"}]

    # 第二次调用：assistant 的 tool_use 与 user 的 tool_result 成对回传；
    # block 内的 id（tool_use_id 配对依据）必须保留，消息级 id 仍被剥掉
    second = llm.client.messages.calls[1]["messages"]
    assert len(second) == 3
    assert all("id" not in m for m in second)
    assert second[-2]["content"][1]["id"] == "toolu_1"
    assert second[-1]["content"][0]["tool_use_id"] == "toolu_1"


def test_hook_nodes_run_at_graph_positions(capsys):
    """状态钩子是独立图节点：agent 级各跑一次，model 级每轮各跑一次。"""
    llm = _fake_llm(_TWO_ROUNDS)
    with patch.object(ex03, "_get_llm", return_value=llm):
        app = ex03.build(InMemorySaver())
        app.invoke(dict(_INPUT), config={"thread_id": "t"})

    out = capsys.readouterr().out
    assert out.count("===== agent 开始") == 1        # before_agent 节点
    assert out.count("===== agent 结束") == 1        # after_agent 节点（END 分支）
    assert out.count("before_model：") == 2          # 两轮模型调用，各过一次
    assert out.count("after_model：") == 2


def test_sdk_blocks_survive_sqlite_checkpoint(tmp_path):
    """SDK 的 pydantic block 转 dict 后才能落 JSON checkpoint —— 验证这条路。"""
    llm = _fake_llm(_TWO_ROUNDS)
    saver = SqliteSaver(str(tmp_path / "ck.db"))
    with patch.object(ex03, "_get_llm", return_value=llm):
        app = ex03.build(saver)
        cfg = {"thread_id": "t"}
        app.invoke(dict(_INPUT), config=cfg)
        restored = app.get_state(cfg).values["messages"]

    assert len(restored) == 4
    assert restored[1]["content"][1] == _TOOL_CALL
    assert restored[-1]["content"][0]["text"] == "北京今天晴，18°C"


def test_tool_error_is_reported_back_to_model():
    """工具抛异常不该炸掉图：错误文本进 tool_result，模型下一轮能看到。"""
    llm = _fake_llm([
        [{"type": "tool_use", "id": "toolu_x", "name": "get_weather",
          "input": {"city": "北京", "unexpected": 1}}],   # 触发 TypeError
        [{"type": "text", "text": "工具挂了，换个说法"}],
    ])
    with patch.object(ex03, "_get_llm", return_value=llm):
        app = ex03.build(InMemorySaver())
        out = app.invoke(dict(_INPUT), config={"thread_id": "t"})

    result = out["messages"][2]["content"][0]
    assert result["type"] == "tool_result"
    assert result["content"].startswith("工具执行失败")
    assert out["messages"][-1]["content"][0]["text"] == "工具挂了，换个说法"
