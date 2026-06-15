from .integration import (
    auto_initialize_telemetry_and_logs,
    create_span,
    complete_span,
)
from .metrics_collector import MetricsCollector
from .metrics_schema import TurnMetrics, SessionMetrics, TimelineEvent
from .traces_flow import TracesFlowManager
from .logger_handler import LogManager, JobLogger

from contextvars import ContextVar
from typing import Optional


_default_metrics_collector = MetricsCollector()

_current_metrics_collector: ContextVar[Optional[MetricsCollector]] = ContextVar(
    "current_metrics_collector", default=None
)


def get_current_metrics_collector() -> MetricsCollector:
    """Return the current session's collector, or the process default."""
    return _current_metrics_collector.get() or _default_metrics_collector


def set_current_metrics_collector(collector: MetricsCollector):
    """Bind a per-session collector to the current context. Returns a reset token."""
    return _current_metrics_collector.set(collector)


def reset_current_metrics_collector(token) -> None:
    _current_metrics_collector.reset(token)


def new_session_metrics_collector() -> MetricsCollector:
    """Create a fresh collector for a new session."""
    return MetricsCollector()


class _MetricsCollectorProxy:
    """Forwards all attribute access to the current session's MetricsCollector.

    Preserves the public ``from videosdk.agents.metrics import metrics_collector``
    import while making the underlying instance session-scoped via a ContextVar.
    """

    __slots__ = ()

    def __getattr__(self, name):
        return getattr(get_current_metrics_collector(), name)

    def __setattr__(self, name, value):
        setattr(get_current_metrics_collector(), name, value)


# Public name preserved; now resolves per-session.
metrics_collector = _MetricsCollectorProxy()



__all__ = [
    'metrics_collector',
    'get_current_metrics_collector',
    'set_current_metrics_collector',
    'reset_current_metrics_collector',
    'new_session_metrics_collector',
    'MetricsCollector',
    'TurnMetrics',
    'SessionMetrics',
    'TimelineEvent',
    'auto_initialize_telemetry_and_logs',
    'create_span',
    'complete_span',
    'TracesFlowManager',
    'LogManager',
    'JobLogger',
]
