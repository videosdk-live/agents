from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Literal, Optional, AsyncIterator
from pydantic import BaseModel  
from ..event_emitter import EventEmitter
import logging
import asyncio
logger = logging.getLogger(__name__)

class SpeechEventType(str, Enum):
    """Type of speech event"""
    START = "start_of_speech"
    INTERIM = "interim_transcript"
    PREFLIGHT = "preflight_transcript"
    FINAL = "final_transcript"
    END = "end_of_speech"


@dataclass
class SpeechData:
    """Data structure for speech recognition results
    
    Attributes:
        text: The recognized text.
        confidence: The confidence level of the recognition.
        language: The language of the recognized text.
        start_time: The start time of the speech.
        end_time: The end time of the speech.
    """
    text: str
    confidence: float = 0.0
    language: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0


class STTResponse(BaseModel):
    """Response from STT processing
    
    Attributes:
        event_type: The type of speech event.
        data: The data from the speech event.
        metadata: Additional metadata from the speech event.
    """
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

    async def stream_transcribe(
        self,
        audio_stream: AsyncIterator[bytes],
        **kwargs: Any
    ) -> AsyncIterator[STTResponse]:
        """
        Process audio stream and yield STT events (simulated via process_audio).
        
        This default implementation allows using the streaming hook pattern even if
        the STT subclass doesn't implement a native streaming method, by redirecting
        callbacks to a queue.
        
        Args:
            audio_stream: Async iterator of audio bytes
            **kwargs: Additional provider-specific arguments
            
        Yields:
            STTResponse objects
        """
        event_queue = asyncio.Queue()
        
        async def capture_callback(response):
            await event_queue.put(response)
            
        original_callback = self._transcript_callback
        self.on_stt_transcript(capture_callback)
        
        async def feed_audio():
            try:
                async for chunk in audio_stream:
                    await self.process_audio(chunk, **kwargs)
            finally:
                pass

        feed_task = asyncio.create_task(feed_audio())
        
        try:
            while True:
                done, pending = await asyncio.wait(
                    [asyncio.create_task(event_queue.get()), feed_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                for task in done:
                    if task == feed_task:
                        pass
                    else:
                        event = task.result()
                        yield event
                        
                if feed_task.done() and event_queue.empty():
                    break
                    
        finally:
            self._transcript_callback = original_callback
            if not feed_task.done():
                feed_task.cancel()
    
    async def aclose(self) -> None:
        """Cleanup resources"""
        logger.info(f"[aclose] Cleaning up STT: {self.label}")
        self._transcript_callback = None
        try:
            import gc
            gc.collect()
            logger.info(f"[aclose] STT garbage collection completed: {self.label}")
        except Exception as e:
            logger.error(f"[aclose] Error during STT garbage collection: {e}")
        
        logger.info(f"[aclose] STT cleanup completed: {self.label}")
