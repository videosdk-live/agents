from __future__ import annotations

from abc import abstractmethod
from typing import Any, AsyncIterator, Literal, Optional

from pydantic import BaseModel
from ..event_emitter import EventEmitter
from .chat_context import ChatContext, ChatRole
from ..utils import FunctionTool


class LLMResponse(BaseModel):
    """Dataclass to hold LLM response data"""
    content: str
    role: ChatRole
    metadata: Optional[dict[str, Any]] = None
    

class LLM(EventEmitter[Literal["error"]]):
    """Base class for LLM implementations"""
    
    def __init__(self) -> None:
        super().__init__()
        self._label = f"{type(self).__module__}.{type(self).__name__}"
        
    @property
    def label(self) -> str:
        """Get the LLM provider label"""
        return self._label
    
    @abstractmethod
    async def chat(
        self,
        messages: ChatContext,
        tools: list[FunctionTool] | None = None,
        **kwargs: Any
    ) -> AsyncIterator[LLMResponse]:
        """
        Main method to interact with the LLM
        
        Args:
            messages: List of message dictionaries with role and content
            **kwargs: Additional arguments specific to LLM provider
            
        Returns:
            AsyncIterator yielding LLMResponse objects
        """
        raise NotImplementedError
    
    async def aclose(self) -> None:
        """Cleanup resources"""
        pass
    
    async def __aenter__(self) -> LLM:
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.aclose()
