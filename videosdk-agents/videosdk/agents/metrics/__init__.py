from .models import TimelineEvent, InteractionMetrics, MetricsData
from .collector import MetricsCollector
from .integration import (
    auto_initialize_telemetry_and_logs,
    create_span,
    complete_span,
    create_log,
)

metrics_collector = MetricsCollector()

__all__ = [
    'metrics_collector',
    'MetricsCollector',
    'TimelineEvent',
    'InteractionMetrics', 
    'MetricsData',
    'auto_initialize_telemetry_and_logs',
    'create_span',
    'complete_span',
    'create_log',
]