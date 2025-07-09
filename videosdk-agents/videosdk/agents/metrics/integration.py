from typing import Dict, Any, Optional
from opentelemetry.trace import Span
from .telemetry import initialize_telemetry, get_telemetry
from .logs import initialize_logs, get_logs, shutdown_logs


def auto_initialize_telemetry_and_logs(room_id: str, peer_id: str, 
                                      room_attributes: Dict[str, Any] = None, session_id: str = None):
    """
    Auto-initialize telemetry and logs from room attributes
    """
    if not room_attributes:
        return
    
    observability_jwt = room_attributes.get('observability', '')
    
    traces_config = room_attributes.get('traces', {})
    if traces_config.get('enabled'):
        metadata = {
        }
        
        initialize_telemetry(
            room_id=room_id,
            peer_id=peer_id,
            sdk_name="agents",
            observability_jwt=observability_jwt,
            traces_config=traces_config,
            metadata=metadata
        )
        
    
    logs_config = room_attributes.get('logs', {})
    if logs_config.get('enabled'):
        initialize_logs(
            meeting_id=room_id,
            peer_id=peer_id,
            jwt_key=observability_jwt, 
            log_config=logs_config,
            session_id=session_id
        )
        


def create_span(span_name: str, attributes: Dict[str, Any] = None, parent_span: Optional[Span] = None):
    """
    Create a trace span (convenience method)
    
    Args:
        span_name: Name of the span
        attributes: Span attributes
        parent_span: Parent span (optional)
        
    Returns:
        Span object or None
    """
    telemetry = get_telemetry()
    if telemetry:
        return telemetry.trace(span_name, attributes, parent_span)
    return None


def complete_span(span: Optional[Span], status_code, message: str = ""):
    """
    Complete a trace span (convenience method)
    
    Args:
        span: Span to complete
        status_code: Status code
        message: Status message
    """
    telemetry = get_telemetry()
    if telemetry and span:
        telemetry.complete_span(span, status_code, message)


def create_log(message: str, log_level: str = "INFO", attributes: Dict[str, Any] = None):
    """
    Create a log entry (convenience method)
    
    Args:
        message: Log message
        log_level: Log level
        attributes: Additional attributes
    """
    logs = get_logs()
    if logs:
        logs.create_log(message, log_level, attributes)


