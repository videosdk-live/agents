from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar, Union

from .event_emitter import EventEmitter

BaseEventTypes = Literal[
    "error"
]

TEvent = TypeVar("TEvent")

@dataclass
class InputTranscriptionCompleted:
    """Event data for transcription completion"""
    item_id: str
    transcript: str

@dataclass
class ErrorEvent:
    """Event data for errors"""
    message: str
    code: str | None = None

class RealtimeBaseModel(EventEmitter[Union[BaseEventTypes, TEvent]], Generic[TEvent], ABC):
    """
    Base class for realtime models with event emission capabilities.
    Allows for extension with additional event types through TEvent.
    """
    
    def __init__(self) -> None:
        """Initialize the realtime model"""
        super().__init__()

    @abstractmethod
    async def aclose(self) -> None:
        """Cleanup resources - must be implemented by subclasses"""
        pass
    
    async def cleanup(self) -> None:
        """Cleanup resources - calls aclose for compatibility"""
        await self.aclose()
    
    async def __aenter__(self) -> RealtimeBaseModel:
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit"""
        await self.aclose()