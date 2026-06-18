from typing import TypeVar, Literal, Optional
from contextvars import ContextVar
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
    """Process-wide default event bus. Constructing ``EventBus()`` always returns
    this one instance; per-session buses are plain ``EventEmitter`` instances bound
    via the ContextVar below. Consumers should use ``global_event_emitter`` (the
    proxy), which resolves to the current session's bus, not ``EventBus()`` directly."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EventBus, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            super().__init__()
            self._initialized = True

    @classmethod
    def get_instance(cls) -> 'EventBus':
        """Return the singleton EventBus instance, creating it if necessary."""
        return cls()
    
_default_event_bus = EventBus()

_current_event_bus: ContextVar[Optional[EventEmitter]] = ContextVar(
    "current_event_bus", default=None
)


def get_current_event_bus() -> EventEmitter:
    """Return the current session's event bus, or the process default."""
    return _current_event_bus.get() or _default_event_bus


def set_current_event_bus(bus: EventEmitter):
    """Bind a per-session event bus to the current context. Returns a reset token."""
    return _current_event_bus.set(bus)


def reset_current_event_bus(token) -> None:
    _current_event_bus.reset(token)


def new_session_event_bus() -> EventEmitter:
    """Create a fresh per-session bus (same on/off/emit interface as EventBus)."""
    return EventEmitter()


class _EventBusProxy:
    """Forwards on/off/emit to the current session's event bus, preserving the
    public ``global_event_emitter`` import while scoping the instance per-session.
    """

    __slots__ = ()

    def __getattr__(self, name):
        return getattr(get_current_event_bus(), name)


global_event_emitter = _EventBusProxy()


__all__ = [
    "EventBus",
    "EventTypes",
    "global_event_emitter",
    "get_current_event_bus",
    "set_current_event_bus",
    "reset_current_event_bus",
    "new_session_event_bus",
]
