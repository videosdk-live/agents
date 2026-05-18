from typing import Any, Callable, TypeVar, Literal
from .event_emitter import EventEmitter

EventTypes = Literal[
    "AUDIO_STREAM_ENABLED",
    "PARTICIPANT_LEFT",
    "AGENT_STARTED",
    "AGENT_STATE_CHANGED",
    "USER_TRANSCRIPT_ADDED",
    "AGENT_TRANSCRIPT_ADDED",
    "TURN_METRICS_ADDED",
    "COMPONENT_METRIC",
    "PIPELINE_ERROR",
    "RECORDING_STATUS"
]

T = TypeVar('T')

class EventBus(EventEmitter[EventTypes]):
    """Per-session event emitter for agent lifecycle and transcript events."""
_fallback_event_bus: EventBus = EventBus()


def _resolve_event_bus() -> EventBus:
    """Resolve the EventBus for the currently-running JobContext, if any."""
    try:
        from .utils import resolve_from_current_job_context
    except Exception:
        return _fallback_event_bus
    return resolve_from_current_job_context("event_bus", _fallback_event_bus)


class GlobalEventEmitter:
    """Thin proxy that routes calls to the EventBus of the active JobContext."""

    __slots__ = ()

    def on(self, event: Any, callback: Callable[..., Any] | None = None) -> Callable[..., Any]:
        return _resolve_event_bus().on(event, callback)

    def off(self, event: Any, callback: Callable[..., Any]) -> None:
        return _resolve_event_bus().off(event, callback)

    def emit(self, event: Any, *args: Any) -> None:
        return _resolve_event_bus().emit(event, *args)

    def __getattr__(self, name: str) -> Any:
        return getattr(_resolve_event_bus(), name)


global_event_emitter: Any = GlobalEventEmitter()
