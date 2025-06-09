from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Literal, Optional
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
        self._transcript_callback: Optional[Callable[[STTResponse], Awaitable[None]]] = None
        
    @property
    def label(self) -> str:
        """Get the STT provider label"""
        return self._label

    def on_stt_transcript(self, callback: Callable[[STTResponse], Awaitable[None]]) -> None:
        """Set callback for receiving STT transcripts"""
        self._transcript_callback = callback

    @abstractmethod
    async def process_audio(
        self,
        audio_frames: bytes,
        language: Optional[str] = None,
        **kwargs: Any
    ) -> None:
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
