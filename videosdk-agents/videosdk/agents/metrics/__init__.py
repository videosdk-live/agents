from .models import TimelineEvent, InteractionMetrics, MetricsData
from .collector import MetricsCollector
from .integration import (
    auto_initialize_telemetry_and_logs,
    create_span,
    complete_span,
    create_log,
)
from .realtime_collector import RealtimeMetricsCollector

metrics_collector = MetricsCollector()
realtime_metrics_collector = RealtimeMetricsCollector()

__all__ = [
    'metrics_collector',
    'MetricsCollector',
    'realtime_metrics_collector',
    'RealtimeMetricsCollector',
    'TimelineEvent',
    'InteractionMetrics', 
    'MetricsData',
    'auto_initialize_telemetry_and_logs',
    'create_span',
    'complete_span',
    'create_log',
]
