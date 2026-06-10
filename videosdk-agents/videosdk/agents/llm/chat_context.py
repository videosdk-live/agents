"""Backward-compatibility shim. The implementation now lives in the
``videosdk.agents.llm.context`` package. Import from there for new code.
"""

from .context import (
    ChatRole,
    ImageContent,
    ChatContent,
    ChatMessage,
    FunctionCall,
    FunctionCallOutput,
    AgentHandoff,
    AgentConfigUpdate,
    ChatItem,
    ChatContext,
    ReadOnlyChatContext,
)

__all__ = [
    "ChatRole",
    "ImageContent",
    "ChatContent",
    "ChatMessage",
    "FunctionCall",
    "FunctionCallOutput",
    "AgentHandoff",
    "AgentConfigUpdate",
    "ChatItem",
    "ChatContext",
    "ReadOnlyChatContext",
]
