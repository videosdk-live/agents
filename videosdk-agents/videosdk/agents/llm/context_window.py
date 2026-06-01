"""Backward-compatibility shim. ContextWindow now lives in the
``videosdk.agents.llm.context`` package.
"""

from .context import ContextWindow

__all__ = ["ContextWindow"]
