from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from .event_emitter import EventEmitter

class Pipeline(EventEmitter[Literal["start"]], ABC):
    """
    Base Pipeline class that other pipeline types (RealTime, Cascading) will inherit from.
    Inherits from EventEmitter to provide event handling capabilities.
    """
    
    def __init__(self) -> None:
        """Initialize the pipeline with event emitter capabilities"""
        super().__init__()

    @abstractmethod
    async def start(self, **kwargs: Any) -> None:
        """
        Start the pipeline processing.
        This is an abstract method that must be implemented by child classes.
        
        Args:
            **kwargs: Additional arguments that may be needed by specific pipeline implementations
        """
        pass
    
    @abstractmethod
    async def send_message(self, message: str) -> None:
        """
        Send a message to the pipeline.
        """
        pass