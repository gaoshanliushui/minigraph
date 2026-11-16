"""
内置中间件（对齐 LangChain v1 的 langchain/agents/middleware）。

通用中间件：

    human_in_the_loop   工具调用要人批准才执行
    model_call_limit    限制模型调用次数
    model_fallback      主模型失败时降级到备用模型
    model_retry         模型调用失败自动重试
    tool_call_limit     限制工具调用次数
    tool_retry          工具调用失败自动重试

Deep agents 中间件：

    filesystem          把可插拔 backend 暴露为文件工具
    patch_tool_calls    为历史中的悬空 tool_use 补错误 tool_result
    skills              技能元数据发现、提示注入和按需全文披露
    summarization       上下文摘要，原始历史保留并转存到 backend

用：

    from deep_agent.middleware import ModelRetryMiddleware, ToolCallLimitMiddleware

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
from .patch_tool_calls import PatchToolCallsMiddleware
from .filesystem import FILESYSTEM_SYSTEM_PROMPT, FilesystemMiddleware
from .skills import (
    SKILLS_SYSTEM_PROMPT,
    SkillMetadata,
    SkillSource,
    SkillsMiddleware,
    SkillsState,
)
from .summarization import (
    DEFAULT_SUMMARY_PROMPT,
    SUMMARIZATION_EVENT_KEY,
    SUMMARIZATION_SESSION_ID_KEY,
    SummarizationEvent,
    SummarizationMiddleware,
    SummarizationState,
    TriggerClause,
    compute_summarization_defaults,
    count_tokens_approximately,
    create_summarization_middleware,
)
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
    # Deep agents
    "PatchToolCallsMiddleware",
    "FilesystemMiddleware",
    "FILESYSTEM_SYSTEM_PROMPT",
    "SkillsMiddleware",
    "SkillsState",
    "SkillMetadata",
    "SkillSource",
    "SKILLS_SYSTEM_PROMPT",
    "SummarizationMiddleware",
    "SummarizationState",
    "SummarizationEvent",
    "TriggerClause",
    "SUMMARIZATION_EVENT_KEY",
    "SUMMARIZATION_SESSION_ID_KEY",
    "DEFAULT_SUMMARY_PROMPT",
    "count_tokens_approximately",
    "compute_summarization_defaults",
    "create_summarization_middleware",
]
