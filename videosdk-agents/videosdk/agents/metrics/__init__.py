# New unified metrics system
from .unified_metrics_collector import UnifiedMetricsCollector
from .turn_lifecycle_tracker import TurnLifecycleTracker
from .component_manager import ComponentMetricsManager
from .metrics_schema import (
    SessionMetrics,
    TurnMetrics,
    VadMetrics,
    SttMetrics,
    EouMetrics,
    LlmMetrics,
    TtsMetrics,
    RealtimeMetrics,
    InterruptionMetrics,
    TimelineEvent,
    FallbackEvent,
    FunctionToolMetrics,
    McpToolMetrics,
    ParticipantMetrics,
    BaseComponentMetrics,
)

# Integration utilities (telemetry, logs, spans)
from .integration import (
    auto_initialize_telemetry_and_logs,
    create_span,
    complete_span,
    create_log,
)

component_metrics_manager = ComponentMetricsManager()
turn_lifecycle_tracker = TurnLifecycleTracker(component_metrics_manager)


__all__ = [
    # Unified metrics collector
    'UnifiedMetricsCollector',
    'turn_lifecycle_tracker',
    'component_metrics_manager',

    # Metrics schema
    'SessionMetrics',
    'TurnMetrics',
    'VadMetrics',
    'SttMetrics',
    'EouMetrics',
    'LlmMetrics',
    'TtsMetrics',
    'RealtimeMetrics',
    'InterruptionMetrics',
    'TimelineEvent',
    'FallbackEvent',
    'FunctionToolMetrics',
    'McpToolMetrics',
    'ParticipantMetrics',
    'BaseComponentMetrics',

    # Integration utilities
    'auto_initialize_telemetry_and_logs',
    'create_span',
    'complete_span',
    'create_log',
]
