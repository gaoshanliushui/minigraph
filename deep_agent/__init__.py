"""基于 minigraph 运行时的 deep agent 产品层。

核心图、状态和 checkpoint 仍由 :mod:`minigraph` 提供；本包包含深度代理工厂、
中间件和可插拔文件 backend。
"""

from .deep_agent import BASE_PROMPT, DeepAgentState, create_deep_agent

__all__ = ["BASE_PROMPT", "DeepAgentState", "create_deep_agent"]
