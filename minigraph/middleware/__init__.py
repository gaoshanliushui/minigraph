"""
内置中间件（对齐 LangChain v1 的 langchain/agents/middleware）。

六个：

    human_in_the_loop   工具调用要人批准才执行
    model_call_limit    限制模型调用次数
    model_fallback      主模型失败时降级到备用模型
    model_retry         模型调用失败自动重试
    tool_call_limit     限制工具调用次数
    tool_retry          工具调用失败自动重试

用：

    from minigraph.middleware import ModelRetryMiddleware, ToolCallLimitMiddleware

    app = create_agent(model, tools=[...], middleware=[
        ModelRetryMiddleware(max_retries=3),
        ToolCallLimitMiddleware(run_limit=10),
    ])
"""

from .core import (
    JUMP_TO_ANNOTATION,
    JUMP_TO_FIELD,
    JumpTo,
    Middleware,
    MiddlewareStack,
    ModelRequest,
    ModelResponse,
    ToolRequest,
    add_hook_nodes,
    agent_state_schema,
    before_agent,
    before_model,
    after_agent,
    after_model,
    hook_config,
    merge_state_schemas,
    middleware,
    wire_middleware_graph,
    wrap_model_call,
    wrap_tool_call,
)
from .human_in_the_loop import (
    Action,
    ActionRequest,
    DecisionType,
    HITLRequest,
    HITLResponse,
    HumanInTheLoopMiddleware,
    HumanInTheLoopState,
    InterruptMode,
    InterruptOnConfig,
    ReviewConfig,
    ToolApprovalRequest,
)
from .model_call_limit import (
    ModelCallLimitExceededError,
    ModelCallLimitMiddleware,
    ModelCallLimitState,
)
from .model_fallback import ModelFallbackMiddleware
from .model_retry import ModelRetryMiddleware
from .tool_call_limit import (
    ToolCallLimitExceededError,
    ToolCallLimitMiddleware,
    ToolCallLimitState,
)
from .tool_retry import ToolRetryMiddleware

__all__ = [
    # 核心
    "Middleware",
    "MiddlewareStack",
    "ModelRequest",
    "ModelResponse",
    "ToolRequest",
    "JumpTo",
    "JUMP_TO_FIELD",
    "JUMP_TO_ANNOTATION",
    "middleware",
    "hook_config",
    "wrap_model_call",
    "wrap_tool_call",
    "before_agent",
    "before_model",
    "after_agent",
    "after_model",
    "add_hook_nodes",
    "agent_state_schema",
    "wire_middleware_graph",
    "merge_state_schemas",
    # 内置
    "HumanInTheLoopMiddleware",
    "HumanInTheLoopState",
    "InterruptOnConfig",
    "InterruptMode",
    "DecisionType",
    "Action",
    "ActionRequest",
    "ReviewConfig",
    "HITLRequest",
    "HITLResponse",
    "ToolApprovalRequest",
    "ModelCallLimitMiddleware",
    "ModelCallLimitState",
    "ModelCallLimitExceededError",
    "ModelFallbackMiddleware",
    "ModelRetryMiddleware",
    "ToolCallLimitMiddleware",
    "ToolCallLimitState",
    "ToolCallLimitExceededError",
    "ToolRetryMiddleware",
]
