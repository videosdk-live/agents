from typing import Any
from .integration import (
    auto_initialize_telemetry_and_logs,
    create_span,
    complete_span,
)
from .metrics_collector import MetricsCollector
from .metrics_schema import TurnMetrics, SessionMetrics
from .traces_flow import TracesFlowManager
from .logger_handler import LogManager, JobLogger


_fallback_metrics_collector: MetricsCollector = MetricsCollector()


def _resolve_metrics_collector() -> MetricsCollector:
    """Return the MetricsCollector of the currently-active JobContext."""
    try:
        from ..utils import resolve_from_current_job_context
    except Exception:
        return _fallback_metrics_collector
    inst = resolve_from_current_job_context("metrics_collector", _fallback_metrics_collector)
    if isinstance(inst, MetricsCollector):
        return inst
    return _fallback_metrics_collector


class _MetricsCollectorProxy:
    """Proxy that routes attribute access to the active JobContext's MetricsCollector."""

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        return getattr(_resolve_metrics_collector(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(_resolve_metrics_collector(), name, value)


metrics_collector: Any = _MetricsCollectorProxy()


__all__ = [
    'metrics_collector',
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
