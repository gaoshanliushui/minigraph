"""create_deep_agent：规划 / 虚拟文件系统 / 子代理 / 中间件节点。

model 是 callable(ModelRequest) 协议，测试用脚本化 fake —— 不依赖任何
SDK、不需要 API key，但走的是完整的图回环与 state 更新。
"""

from minigraph.checkpoint_sqlite import SqliteSaver
from minigraph.deep_agent import create_deep_agent
from minigraph.middleware import middleware


def _text(s):
    return {"type": "text", "text": s}


def _call(cid, name, args):
    return {"type": "tool_use", "id": cid, "name": name, "input": args}


class FakeModel:
    """按脚本逐次返回 content blocks；记录每次 ModelRequest 供断言。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        assert self.responses, "fake model 脚本耗尽（回环次数超预期）"
        r = self.responses.pop(0)
        return r(request) if callable(r) else r


def _results(out):
    """{tool_use_id: tool_result 文本}，从最终消息历史里收集。"""
    res = {}
    for m in out["messages"]:
        content = m["content"]
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    res[b["tool_use_id"]] = b["content"]
    return res


_INPUT = {"messages": [{"id": "h1", "role": "user", "content": "开始"}]}
_CFG = {"thread_id": "t"}


def test_todos_and_files_full_flow():
    model = FakeModel([
        [_call("1", "write_todos",
               {"todos": [{"content": "调研", "status": "in_progress"}]}),
         _call("2", "write_file", {"path": "/notes.md", "content": "北京晴"})],
        [_call("3", "read_file", {"path": "/notes.md"})],
        [_text("完成")],
    ])
    app = create_deep_agent(model, system_prompt="你是测试代理")
    out = app.invoke(dict(_INPUT), config=_CFG)

    # 规划与虚拟文件系统都落在 state 里
    assert out["todos"] == [{"content": "调研", "status": "in_progress"}]
    assert out["files"] == {"/notes.md": "北京晴"}
    # read_file 把文件内容回给了模型
    assert _results(out)["3"] == "北京晴"
    # system prompt = USER -> BASE 拼装；工具清单含全部内置工具
    req = model.requests[0]
    assert req.system.startswith("你是测试代理")
    assert "write_todos" in req.system
    names = {t["name"] for t in req.tools}
    assert {"write_todos", "ls", "read_file", "write_file", "edit_file",
            "task"} <= names


def test_edit_ls_and_error_paths():
    model = FakeModel([
        [_call("1", "write_file", {"path": "/a.txt", "content": "hello"})],
        [_call("2", "edit_file",
               {"path": "/a.txt", "old_string": "he", "new_string": "HE"}),
         _call("3", "read_file", {"path": "/missing"})],
        [_call("4", "ls", {})],
        [_text("done")],
    ])
    app = create_deep_agent(model)
    out = app.invoke(dict(_INPUT), config=_CFG)

    assert out["files"]["/a.txt"] == "HEllo"
    res = _results(out)
    assert "不存在" in res["3"]                  # 读不存在的文件 -> 错误文本
    assert "/a.txt" in res["4"]                  # ls 列出文件
    assert out["messages"][-1]["content"][0]["text"] == "done"


def test_task_subagent_returns_conclusion():
    def model(request):
        last = request.messages[-1]
        # 子代理的第一次调用：user 消息就是 task 的 prompt
        if last["role"] == "user" and isinstance(last["content"], str) \
                and "子任务" in last["content"]:
            return [_text("子代理结论：42")]
        # 主代理：还没有 tool_result 时派 task，否则收尾
        has_result = any(
            isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_result" for b in m["content"])
            for m in request.messages)
        if not has_result:
            return [_call("t1", "task",
                          {"description": "算数", "prompt": "请完成子任务：6*7",
                           "subagent_type": "general-purpose"})]
        return [_text("主代理完成")]

    app = create_deep_agent(model)
    out = app.invoke(dict(_INPUT), config=_CFG)

    assert "子代理结论：42" in _results(out)["t1"]
    assert out["messages"][-1]["content"][0]["text"] == "主代理完成"


def test_middleware_hook_nodes_and_custom_tool():
    events = []

    @middleware
    class Watch:
        def before_agent(self, state):
            events.append("before_agent")

        def after_agent(self, state):
            events.append("after_agent")

        def before_model(self, state):
            events.append("before_model")

        def after_model(self, state):
            events.append("after_model")

    def double(x: int) -> int:                     # 裸函数，自动 @tool 包装
        return x * 2

    model = FakeModel([
        [_call("1", "double", {"x": 21})],
        [_text("42")],
    ])
    app = create_deep_agent(model, tools=[double], middleware=[Watch])
    out = app.invoke(dict(_INPUT), config=_CFG)

    assert _results(out)["1"] == "42"
    assert events.count("before_agent") == 1       # agent 级：整个调用一次
    assert events.count("after_agent") == 1
    assert events.count("before_model") == 2       # model 级：每轮一次
    assert events.count("after_model") == 2


def test_state_survives_sqlite_checkpoint(tmp_path):
    model = FakeModel([
        [_call("1", "write_file", {"path": "/a.txt", "content": "hello"}),
         _call("2", "write_todos", {"todos": [{"content": "x", "status": "pending"}]})],
        [_text("done")],
    ])
    app = create_deep_agent(model, checkpointer=SqliteSaver(str(tmp_path / "d.db")))
    cfg = {"thread_id": "t"}
    app.invoke(dict(_INPUT), config=cfg)
    values = app.get_state(cfg).values

    assert values["files"] == {"/a.txt": "hello"}
    assert values["todos"] == [{"content": "x", "status": "pending"}]
    assert len(values["messages"]) == 4            # user/ai/tool_result/ai
