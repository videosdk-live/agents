from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Coroutine
from typing import Literal

from av import AudioFrame, VideoFrame
from ..event_emitter import EventEmitter


class AudioSegmentEnd:
    """Sentinel yielded by an AvatarRenderer to signal that a TTS segment has fully played."""
    pass


class AvatarInput(ABC, EventEmitter[Literal["reset_stream"]]):
    """
    Abstract base for the avatar-worker-side audio receiver.

    The Avatar Server iterates over an AvatarInput to receive AudioFrame objects
    from the agent. When the agent finishes a TTS segment it sends a
    ``segment_end`` control message which causes the receiver to enqueue an
    AudioSegmentEnd sentinel. The receiver also emits a ``reset_stream`` event
    when an interrupt arrives from the agent.
    """

    def __init__(self):
        super().__init__()

    async def start_stream(self) -> None:
        """Optional hook called before the first frame is consumed."""
        pass

    @abstractmethod
    def notify_stream_ended(
        self, playback_position: float, interrupted: bool
    ) -> None | Coroutine[None, None, None]:
        """Send a stream_ended ack back to the agent."""

    @abstractmethod
    def __aiter__(self) -> AsyncIterator[AudioFrame | AudioSegmentEnd]:
        """Yield AudioFrame items, then AudioSegmentEnd when the segment is done."""

    async def aclose(self) -> None:
        pass


class AvatarRenderer(ABC):
    """
    Abstract base for the avatar-worker-side video/audio renderer.

    An AvatarRenderer receives audio frames via push_stream_chunk and yields
    interleaved VideoFrame / AudioFrame items plus an AudioSegmentEnd sentinel
    when the current segment has been fully rendered.
    """

    @abstractmethod
    async def push_stream_chunk(self, frame: AudioFrame | AudioSegmentEnd) -> None:
        """Receive an audio frame (or segment-end sentinel) from the controller."""

    @abstractmethod
    def reset_stream(self) -> None | Coroutine[None, None, None]:
        """Immediately discard buffered audio (called on interrupt)."""

    @abstractmethod
    def __aiter__(
        self,
    ) -> AsyncIterator[VideoFrame | AudioFrame | AudioSegmentEnd]:
        """Yield interleaved video+audio frames, then AudioSegmentEnd."""
