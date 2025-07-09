import os
import json
from typing import Dict, Any, Optional
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode, Span


class VideoSDKTelemetry:
    """OpenTelemetry traces for VideoSDK agents"""
    
    def __init__(self, room_id: str, peer_id: str, sdk_name: str, observability_jwt: str, 
                 traces_config: Dict[str, Any], metadata: Dict[str, Any]):
        """
        Initialize telemetry with OpenTelemetry configuration
        
        Args:
            room_id: Room/meeting ID
            peer_id: Peer/participant ID  
            sdk_name: SDK name (e.g., "agents")
            observability_jwt: JWT token for authentication
            traces_config: Trace configuration with 'enabled' and 'endPoint'
            metadata: Additional metadata like userId, email
        """
        self.room_id = room_id
        self.peer_id = peer_id
        self.sdk_name = sdk_name
        self.observability_jwt = observability_jwt
        self.traces_enabled = traces_config.get('enabled', False)
        self.pb_endpoint = traces_config.get('pbEndPoint')
        self.metadata = metadata
        
        self.tracer = None
        self.root_span = None
        self.tracer_provider = None
        
        self._initialize_tracer()
    
    def _initialize_tracer(self):
        """Initialize OpenTelemetry tracer and create root span"""
        try:
            if not self.traces_enabled:
                return
            
            resource = Resource(attributes={
                "service.name": "videosdk-otel-telemetry-agents",
                "sdk.version": "0.0.19"
            })
            
            headers = {}
            if self.observability_jwt:
                headers["Authorization"] = self.observability_jwt
            
            otlp_exporter = OTLPSpanExporter(
                endpoint=self.pb_endpoint,
                headers=headers
            )
            
            batch_processor = BatchSpanProcessor(otlp_exporter)
            self.tracer_provider = TracerProvider(resource=resource)
            self.tracer_provider.add_span_processor(batch_processor)
            
            trace.set_tracer_provider(self.tracer_provider)
            
            self.tracer = trace.get_tracer(self.peer_id)
            
            self._create_root_span()
            
        except Exception as e:
            print(f"[TELEMETRY ERROR] Failed to initialize telemetry: {e}")
    
    def _create_root_span(self):
        """Create root span for the session"""
        if not self.tracer:
            return
            
        try:
            span_name = f"room_{self.room_id}_peer_{self.peer_id}_sdk_{self.sdk_name}"
            self.root_span = self.tracer.start_span(span_name)
            
            self.root_span.set_attribute("roomId", self.room_id)
            self.root_span.set_attribute("peerId", self.peer_id)
            self.root_span.set_attribute("sdkName", self.sdk_name)
            
        except Exception as e:
            print(f"[TELEMETRY ERROR] Failed to create root span: {e}")
    
    def trace(self, span_name: str, attributes: Dict[str, Any] = None, parent_span: Optional[Span] = None) -> Optional[Span]:
        """
        Create a new trace span
        
        Args:
            span_name: Name of the span
            attributes: Key-value attributes to add to span
            parent_span: Parent span (optional, uses root span if None)
            
        Returns:
            Span object or None if tracing disabled
        """
        if not self.traces_enabled or not self.tracer:
            return None
            
        try:
            parent = parent_span or self.root_span

            with trace.use_span(parent):
                span = self.tracer.start_span(span_name)
                
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, str(value))
                
                return span
                
        except Exception as e:
            print(f"[TELEMETRY ERROR] Failed to create span '{span_name}': {e}")
            return None
    
    def complete_span(self, span: Optional[Span], status: StatusCode, message: str = ""):
        """
        Complete a span with status and message
        
        Args:
            span: Span to complete
            status: Status code
            message: Status message
        """
        if not self.traces_enabled or not span:
            return
            
        try:
            if message:
                span.set_attribute("message", message)
            span.set_status(Status(status, message))
            span.end()
                
        except Exception as e:
            print(f"[TELEMETRY ERROR] Failed to complete span: {e}")
    
    def flush(self):
        """Flush and shutdown the tracer provider"""
        if self.traces_enabled and self.tracer_provider:
            try:
                if self.root_span:
                    self.root_span.end()
                
                self.tracer_provider.shutdown()
                
            except Exception as e:
                print(f"[TELEMETRY ERROR] Failed to flush telemetry: {e}")


_telemetry_instance: Optional[VideoSDKTelemetry] = None


def get_telemetry() -> Optional[VideoSDKTelemetry]:
    """Get the global telemetry instance"""
    return _telemetry_instance


def initialize_telemetry(room_id: str, peer_id: str, sdk_name: str = "agents", 
                        observability_jwt: str = None, traces_config: Dict[str, Any] = None, 
                        metadata: Dict[str, Any] = None):
    """
    Initialize global telemetry instance
    
    Args:
        room_id: Room/meeting ID
        peer_id: Peer/participant ID
        sdk_name: SDK name
        observability_jwt: JWT token for authentication
        traces_config: Trace configuration with 'pbendPoint' and 'enabled' 
        metadata: Additional metadata
    """
    global _telemetry_instance
    
    if not observability_jwt:
        observability_jwt = ""
    
    if not traces_config:
        traces_config = {"enabled": False, "endPoint": ""}
    
    if not metadata:
        metadata = {}
    
    _telemetry_instance = VideoSDKTelemetry(
        room_id=room_id,
        peer_id=peer_id, 
        sdk_name=sdk_name,
        observability_jwt=observability_jwt,
        traces_config=traces_config,
        metadata=metadata
    )