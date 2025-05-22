from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, List, Literal, Optional
from pydantic import BaseModel
from ..event_emitter import EventEmitter


class SpeechEventType(str, Enum):
    START = "start_of_speech"
    INTERIM = "interim_transcript"
    FINAL = "final_transcript"
    END = "end_of_speech"


@dataclass
class SpeechData:
    """Data structure for speech recognition results"""
    text: str
    confidence: float = 0.0
    language: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0


class STTResponse(BaseModel):
    """Response from STT processing"""
    event_type: SpeechEventType
    data: SpeechData
    metadata: Optional[dict[str, Any]] = None

class STT(EventEmitter[Literal["error"]]):
    """Base class for Speech-to-Text implementations"""
    
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self._label = f"{type(self).__module__}.{type(self).__name__}"
        
    @property
    def label(self) -> str:
        """Get the STT provider label"""
        return self._label

    @abstractmethod
    async def process_audio(
        self,
        audio_frames: AsyncIterator[bytes],
        language: Optional[str] = None,
        **kwargs: Any
    ) -> AsyncIterator[STTResponse]:
        """
        Process audio frames and convert to text
        
        Args:
            audio_frames: Iterator of bytes to process
            language: Optional language code for recognition
            **kwargs: Additional provider-specific arguments
            
        Returns:
            AsyncIterator yielding STTResponse objects
        """
        raise NotImplementedError

    async def aclose(self) -> None:
        """Cleanup resources"""
        pass
    
    async def __aenter__(self) -> STT:
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.aclose()