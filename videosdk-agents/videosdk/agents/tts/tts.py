from __future__ import annotations

from abc import abstractmethod
from typing import Any, AsyncIterator, Literal, Optional, Callable, Awaitable
from ..event_emitter import EventEmitter
import logging
import asyncio
logger = logging.getLogger(__name__)

class TTS(EventEmitter[Literal["error"]]):
    """Base class for Text-to-Speech implementations"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1
    ) -> None:
        super().__init__()
        self._label = f"{type(self).__module__}.{type(self).__name__}"
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._first_audio_callback: Optional[Callable[[], Awaitable[None]]] = None
        self.audio_track = None 

    @property
    def label(self) -> str:
        """Get the TTS provider label"""
        return self._label
    
    @property
    def sample_rate(self) -> int:
        """Get audio sample rate"""
        return self._sample_rate
    
    @property
    def num_channels(self) -> int:
        """Get number of audio channels"""
        return self._num_channels

    def on_first_audio_byte(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Set callback for when first audio byte is produced"""
        self._first_audio_callback = callback

    def reset_first_audio_tracking(self) -> None:
        """Reset the first audio tracking state for next TTS task"""
        # To be overridden by implementations for TTFB metrics
        pass 

    async def pause(self) -> None:
        if self.audio_track and hasattr(self.audio_track, 'pause'):
            await self.audio_track.pause()
        else:
            await self.interrupt()

    async def resume(self) -> None:
        if self.audio_track and hasattr(self.audio_track, 'resume'):
            await self.audio_track.resume()

    @property
    def can_pause(self) -> bool:
        return self.audio_track and hasattr(self.audio_track, 'can_pause') and self.audio_track.can_pause

    @abstractmethod
    async def synthesize(
        self,
        text: AsyncIterator[str] | str,
        voice_id: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Convert text to speech
        
        Args:
            text: Text to convert to speech (either string or async iterator of strings)
            voice_id: Optional voice identifier
            **kwargs: Additional provider-specific arguments
            
        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    async def interrupt(self) -> None:
        """Interrupt the TTS process"""
        raise NotImplementedError

    async def stream_synthesize(
        self,
        text_stream: AsyncIterator[str],
        **kwargs: Any
    ) -> AsyncIterator[bytes]:
        """
        Synthesize text stream to audio stream.
        
        This default implementation mocks the audio track to capture frames.
        
        Args:
            text_stream: Async iterator of text
            **kwargs: Additional arguments
            
        Yields:
            Audio bytes
        """
        original_track = self.audio_track
        frame_queue = asyncio.Queue()
        
        class QueueTrack:
            def __init__(self):
                self.hooks = None
            async def add_new_bytes(self, audio_data: bytes):
                await frame_queue.put(audio_data)
            def on_last_audio_byte(self, cb):
                pass
            def set_pipeline_hooks(self, hooks):
                self.hooks = hooks
            def enable_audio_input(self, manual_control=False):
                pass
                
        mock_track = QueueTrack()
        self.audio_track = mock_track
        
        async def synthesize_task():
            try:
                await self.synthesize(text_stream, **kwargs)
            finally:
                await frame_queue.put(None)
                
        task = asyncio.create_task(synthesize_task())
        
        try:
            while True:
                get_task = asyncio.create_task(frame_queue.get())
                done, pending = await asyncio.wait(
                    [get_task, task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                if get_task in done:
                    data = get_task.result()
                    if data is None:
                        break
                    yield data
                    
                if task in done:
                    if task.exception():
                        raise task.exception()
                        
        finally:
            self.audio_track = original_track
            if not task.done():
                task.cancel()

    async def aclose(self) -> None:
        """Cleanup resources"""
        logger.info(f"Cleaning up TTS: {self.label}")
        self._first_audio_callback = None
        try:
            import gc
            gc.collect()
        except Exception as e:
            logger.error(f"Error during TTS garbage collection: {e}")
        
        logger.info(f"TTS cleanup completed: {self.label}")
    
    async def __aenter__(self) -> TTS:
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.aclose()
