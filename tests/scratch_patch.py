"""把 deep_agent.py 里的 tool_node 换成 派活 + 单调用任务 + 汇总 三段。"""
import pathlib

NEW = '''    def tool_node(state: DeepAgentState) -> dict:
        """派活节点：把一条 assistant 消息里的多个 tool_use 扇出成并行任务。

        仿照 LangGraph 的 ``Send("tools", [tool_call])``：每个工具调用
        变成**一个独立任务**，任务之间并发跑、互不干扰。

        为什么不是"一个节点里用线程池并发"：
        节点是一个整体 —— 它挂起时整步回滚，节点里已经跑完的兄弟调用
        也会被**重跑一遍**，有副作用的工具就执行了两次。
        拆成独立任务后，每个任务有自己的槽位：别人挂起时，跑成功的
        那些任务的写入被存进 checkpoint，恢复时直接取用、不重跑。
        人工审批（per_call）也因此能一次只问一个。
        """
        last = state["messages"][-1]
        calls = [b for b in last["content"] if b.get("type") == "tool_use"]
        if not calls:
            # 没有调用就不派活。返回空 dict（而不是空 Send 列表）——
            # 空列表会被当成"我在派活"，把静态边也一起抑制掉，
            # 汇总节点就永远等不到触发。
            return {}
        return [Send("tool_call", {"tool_call": c}) for c in calls]

    def tool_call(state: DeepAgentState) -> dict:
        """跑**一个**工具调用，写出它的 tool_result。

        每个调用一个独立任务（见 tool_node 的说明），所以：
          - 多个调用天然并发（同一个超步里的多个任务）；
          - 某个调用挂起（人工审批）不影响已经跑完的兄弟任务；
          - 结果的**顺序**由任务的写入顺序决定 —— 运行时按任务顺序
            收写入，不是按完成顺序，所以顺序与调用顺序一致。
        """
        c = state["tool_call"]
        # 工作线程不会继承 thread-local：不装回去的话，工具里的 interrupt()
        # 找不到自己的槽位（"interrupt() 只能在图节点内调用"）。
        apply_context(current_context())

        t = registry.get(c.get("name"))
        upd: dict = {}

        def _exec(request: ToolRequest) -> str:
            if t is None:
                return f"错误：未知工具 {request.name!r}"
            if isinstance(t, _StateTool):
                text, state_upd = t.state_fn(state, request.args)
                upd.update(state_upd or {})
                return text
            return str(t.invoke(request.args))

        out = stack.tool_call(ToolRequest(c, state=state), _exec)
        # 每个任务写**自己**那条消息（id 里带上 call id）：
        # 汇总节点按调用顺序把它们拼成一条消息。
        return {**upd, "messages": [{
            "id": _result_id(c),
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": c["id"],
                         "content": out}],
        }]}

    def tool_results_join(state: DeepAgentState) -> dict:
        """把各工具任务的结果拼成**一条** user 消息。

        add_messages 只能追加或按 id 替换，没法把多条消息合成一条 ——
        所以这里走"remove-all + 完整重放"（和 PatchToolCallsMiddleware
        一个套路）：先删掉各任务写下的零散消息，再按**调用顺序**重放
        一条合并后的消息。
        """
        messages = state["messages"]
        last = messages[-1]
        calls = [b for b in last["content"] if b.get("type") == "tool_use"]
        order = {c["id"]: i for i, c in enumerate(calls)}

        # 挑出各工具任务写下的结果消息（id 形如 tool-result:<call id>）
        picked = [m for m in messages
                  if str(m.get("id", "")).startswith(_RESULT_PREFIX)]
        picked.sort(key=lambda m: order.get(
            _call_id_of(str(m.get("id", ""))), len(order)))

        blocks: list[dict] = []
        for m in picked:
            content = m.get("content")
            if isinstance(content, list):
                blocks.extend(b for b in content
                              if isinstance(b, dict)
                              and b.get("type") == "tool_result")
        if not blocks:
            return {}

        merged = {"id": _mid("tool"), "role": "user", "content": blocks}
        # 保留工具结果**之前**的所有消息，替换掉零散的那些。
        head = [m for m in messages
                if not str(m.get("id", "")).startswith(_RESULT_PREFIX)]
        remove_all = {"id": REMOVE_ALL_MESSAGES, "role": "remove", "content": ""}
        return {"messages": [remove_all, *head, merged]}

'''

p = pathlib.Path("deep_agent/deep_agent.py")
lines = p.read_text().splitlines(keepends=True)
start = next(i for i, l in enumerate(lines)
             if l.startswith("    def tool_node(state: DeepAgentState) -> dict:"))
end = next(i for i in range(start, len(lines))
           if lines[i].startswith("    def should_continue(state: DeepAgentState) -> str:"))
lines[start:end] = [NEW]
p.write_text("".join(lines))
print("patched", start + 1, "->", end)
