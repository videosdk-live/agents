from .models import TimelineEvent, CascadingTurnData, CascadingMetricsData
from .integration import (
    auto_initialize_telemetry_and_logs,
    create_span,
    complete_span,
    create_log,
)
from .metrics_collector import MetricsCollector
from .metrics_schema import TurnMetrics, SessionMetrics
from .traces_flow import TracesFlowManager

# Single unified metrics collector instance
metrics_collector = MetricsCollector()

# Backward compatibility aliases (same instance)
cascading_metrics_collector = metrics_collector
realtime_metrics_collector = metrics_collector

__all__ = [
    'metrics_collector',
    'MetricsCollector',
    'cascading_metrics_collector',
    'realtime_metrics_collector',
    'TurnMetrics',
    'SessionMetrics',
    'TimelineEvent',
    'CascadingTurnData',
    'CascadingMetricsData',
    'auto_initialize_telemetry_and_logs',
    'create_span',
    'complete_span',
    'create_log',
    'TracesFlowManager',
]
