from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Literal, Optional

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

    async def aclose(self) -> None:
        """Cleanup resources"""
        pass
    
    async def __aenter__(self) -> TTS:
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.aclose()
