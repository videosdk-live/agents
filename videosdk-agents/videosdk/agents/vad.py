from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Literal, Optional
from pydantic import BaseModel

from .event_emitter import EventEmitter

class VADEventType(str, Enum):
    START = "start_of_speech"
    SPEECH = "speech_detected"
    SILENCE = "silence_detected"
    END = "end_of_speech"


@dataclass
class VADData:
    """Data structure for voice activity detection results"""
    is_speech: bool
    confidence: float = 0.0
    timestamp: float = 0.0
    speech_duration: float = 0.0
    silence_duration: float = 0.0


class VADResponse(BaseModel):
    """Response from VAD processing"""
    event_type: VADEventType
    data: VADData
    metadata: Optional[dict[str, Any]] = None


class VAD(EventEmitter[Literal["error"]]):
    """Base class for Voice Activity Detection implementations"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_duration: float = 0.5,
        min_silence_duration: float = 0.5
    ) -> None:
        super().__init__()
        self._label = f"{type(self).__module__}.{type(self).__name__}"
        self._sample_rate = sample_rate
        self._threshold = threshold
        self._min_speech_duration = min_speech_duration
        self._min_silence_duration = min_silence_duration

    @property
    def label(self) -> str:
        """Get the VAD provider label"""
        return self._label

    @property
    def sample_rate(self) -> int:
        """Get audio sample rate"""
        return self._sample_rate

    @abstractmethod
    async def process_audio(
        self,
        audio_frames: bytes,
        **kwargs: Any
    ) -> AsyncIterator[VADResponse]:
        """
        Process audio frames and detect voice activity
        
        Args:
            audio_frames: Iterator of audio frames to process
            **kwargs: Additional provider-specific arguments
            
        Returns:
            AsyncIterator yielding VADResponse objects
        """
        raise NotImplementedError

    async def aclose(self) -> None:
        """Cleanup resources"""
        pass
    
    async def __aenter__(self) -> VAD:
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.aclose()
