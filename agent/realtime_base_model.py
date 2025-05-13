from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, Literal, TypeVar, Union

from .event_emitter import EventEmitter

# Base event types that all realtime models should support
BaseEventTypes = Literal[
    "error"
]

# Generic type var for additional events
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
        self.config: Dict[str, Any] | None = None

    @abstractmethod
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Set configuration for the realtime model.
        Must be called before processing starts.
        
        Args:
            config: Configuration dictionary for the model
        """
        self.config = config

    @abstractmethod
    async def aclose(self) -> None:
        """Cleanup resources"""
        pass