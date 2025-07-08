from __future__ import annotations

from abc import abstractmethod
from typing import Any, Literal

from .event_emitter import EventEmitter


class Denoise(EventEmitter[Literal["error"]]):
    """Base class for Denoise implementations"""

    def __init__(
        self,
    ) -> None:
        super().__init__()
        self._label = f"{type(self).__module__}.{type(self).__name__}"

    @property
    def label(self) -> str:
        """Get the Denoise provider label"""
        return self._label

    @abstractmethod
    async def denoise(
        self,
        audio_frames: bytes,
        **kwargs: Any,
    ) -> bytes:
        """
        Process audio frames to denoise them.
        Denoised audio frames should be sent via the on_denoised_audio callback.

        Args:
            audio_frames: bytes of audio to process
            **kwargs: Additional provider-specific arguments
        """
        raise NotImplementedError

    async def aclose(self) -> None:
        """Cleanup resources"""
        pass

    async def __aenter__(self) -> Denoise:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.aclose()
