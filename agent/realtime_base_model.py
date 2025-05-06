from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, Literal, TypeVar, Union

from .event_emitter import EventEmitter

BaseEventTypes = Literal[
    "input_speech_started",
    "input_speech_stopped",
    "input_audio_transcription_completed",
    "generation_created",
    "error"
]

TEvent = TypeVar("TEvent")

class RealtimeBaseModel(EventEmitter[Union[BaseEventTypes, TEvent]], Generic[TEvent], ABC):
    """
    Base class for realtime models that can be used in the realtime pipeline.
    """
    
    def __init__(self) -> None:
        """Initialize the realtime model"""
        pass