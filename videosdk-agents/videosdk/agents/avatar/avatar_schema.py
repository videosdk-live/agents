from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Coroutine
from typing import Literal

from av import AudioFrame, VideoFrame
from videosdk.agents.event_emitter import EventEmitter

class AudioSegmentEnd:
    pass


class AvatarInput(ABC, EventEmitter[Literal["reset_stream"]]):
    def __init__(self):
        super().__init__()

    async def start_stream(self) -> None:
        pass

    @abstractmethod
    def notify_stream_ended(
        self, playback_position: float, interrupted: bool
    ) -> None | Coroutine[None, None, None]:
        """Notify the sender that playback has finished"""

    @abstractmethod
    def __aiter__(self) -> AsyncIterator[AudioFrame | AudioSegmentEnd]:
        """Continuously stream out audio frames or AudioSegmentEnd when the stream ends"""

    async def aclose(self) -> None:
        pass


class AvatarRenderer(ABC):
    @abstractmethod
    async def push_stream_chunk(self, frame: AudioFrame | AudioSegmentEnd) -> None:
        """Push an audio frame to the video generator"""

    @abstractmethod
    def reset_stream(self) -> None | Coroutine[None, None, None]:
        """Clear the audio buffer, stopping audio playback immediately"""

    @abstractmethod
    def __aiter__(
        self,
    ) -> AsyncIterator[VideoFrame | AudioFrame | AudioSegmentEnd]:
        """Continuously stream out video and audio frames, or AudioSegmentEnd when the audio segment ends"""  # noqa: E501
