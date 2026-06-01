from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Literal, TypeVar, Union

from .event_emitter import EventEmitter

if TYPE_CHECKING:
    from .utterance_handle import UtteranceHandle

logger = logging.getLogger(__name__)

BaseEventTypes = Literal[
    "error"
]

TEvent = TypeVar("TEvent")

@dataclass
class InputTranscriptionCompleted:
    """Event data for transcription completion"""
    item_id: str
    transcript: str

@dataclass
class ErrorEvent:
    """Event data for errors"""
    message: str
    code: str | None = None

class RealtimeBaseModel(EventEmitter[Union[BaseEventTypes, TEvent]], Generic[TEvent], ABC):
    """
    Base class for realtime models with event emission capabilities.
    Allows for extension with additional event types through TEvent.
    """
    
    def __init__(self) -> None:
        """Initialize the realtime model"""
        super().__init__()
        self.current_utterance: UtteranceHandle | None = None
        self.audio_track = None  
        self.loop = None         

    @abstractmethod
    async def aclose(self) -> None:
        """Cleanup resources - must be implemented by subclasses."""
        self.audio_track = None
        self.loop = None
    
    async def cleanup(self) -> None:
        """Cleanup resources - calls aclose for compatibility"""
        await self.aclose()
    
    async def __aenter__(self) -> RealtimeBaseModel:
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit"""
        await self.aclose()

    def instructions_with_context(self, base: str | None) -> str:
        """Fold the agent's ``chat_context`` into the base instructions.

        Realtime providers do not reliably accept streamed
        ``conversation.item.create`` seeding — it tends to wedge or close the
        session. Instead, prior conversation rides along inside the system
        instructions on connect. This default produces a framing that
        explicitly tells the model it is mid-call, so the transcript is
        treated as live memory rather than backstory the model can improvise
        from (which leaks specifics like order numbers and prior tool
        results).

        Reads ``self._agent`` if set; best-effort — any failure (no agent,
        empty chat_context, render error) returns ``base`` unchanged. Override
        in a plugin only if a provider needs a different framing.
        """
        base = base or "You are a helpful assistant that can answer questions and help with tasks."
        agent = getattr(self, "_agent", None)
        if not agent or not getattr(agent, "chat_context", None):
            return base
        if not agent.chat_context.items:
            return base
        try:
            from .llm.format_converters import render_context_as_text
            prior = render_context_as_text(agent.chat_context)
            if not prior.strip():
                return base
            logger.info(
                "%s: seeded prior conversation into instructions",
                type(self).__name__,
            )
            return (
                f"{base}\n\n"
                "## Ongoing conversation — you are mid-call with this caller\n"
                "The transcript below is everything that has already been said "
                "and done on this call. You remember every detail. Continue "
                "from this point — do NOT re-introduce yourself, do NOT re-ask "
                "for information the caller already provided (names, IDs, "
                "order numbers, reasons), and refer back to specifics naturally "
                "when relevant. Treat any `[tool result]` lines as facts you "
                "already know; do NOT re-call those tools for the same "
                "inputs.\n\n"
                f"{prior}\n\n"
                "## End of prior transcript — continue the call now."
            )
        except Exception as e:
            logger.warning(
                "%s: chat context seeding failed: %s",
                type(self).__name__, e,
            )
            return base

    def reframe_audio_track(self, rate_hz: int, frame_ms: float = 20.0) -> None:
        """Reframe the shared audio track for this realtime model's PCM rate.

        After a cascade→realtime pipeline switch the shared ``audio_track``
        still carries the previous TTS framing (sample rate, samples per
        frame, chunk size), which misframes realtime audio and produces
        wrong-speed playback. Plugins call this once on connect with the
        provider's output PCM rate.

        No-op if no ``audio_track`` is attached (text-only mode or
        pre-attach). Best-effort — logs and continues on failure rather than
        crashing connect.
        """
        if not self.audio_track:
            return
        try:
            from fractions import Fraction
            self.audio_track.sample_rate = rate_hz
            self.audio_track.time_base_fraction = Fraction(1, rate_hz)
            self.audio_track.samples = int(frame_ms / 1000.0 * rate_hz)
            self.audio_track.chunk_size = int(
                self.audio_track.samples
                * getattr(self.audio_track, "channels", 1)
                * getattr(self.audio_track, "sample_width", 2)
            )
        except Exception as e:
            logger.warning(
                "%s: could not reconfigure audio track: %s",
                type(self).__name__, e,
            )