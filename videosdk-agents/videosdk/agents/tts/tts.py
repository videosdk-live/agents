from __future__ import annotations

from abc import abstractmethod
from typing import Any, AsyncIterator, Literal, Optional, Callable, Awaitable

from ..event_emitter import EventEmitter

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

    async def aclose(self) -> None:
        """Cleanup resources"""
        pass
    
    async def __aenter__(self) -> TTS:
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.aclose()
