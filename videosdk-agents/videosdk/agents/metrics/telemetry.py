import traceback
from typing import Dict, Any, Optional
import uuid
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode, Span
import time

def generate_id():
    return str(uuid.uuid4())

class VideoSDKTelemetry:
    """OpenTelemetry traces for VideoSDK agents"""
    
    def __init__(self, room_id: str, peer_id: str, sdk_name: str, observability_jwt: str, 
                 traces_config: Dict[str, Any], metadata: Dict[str, Any], sdk_metadata: Dict[str, Any] = None):
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
        self.sdk_metadata = sdk_metadata
        
        self.tracer = None
        self.root_span = None
        self.tracer_provider = None
        
        self._initialize_tracer()
        self.span_details = {}
    
    def _initialize_tracer(self):
        """Initialize OpenTelemetry tracer and create root span"""
        try:
            if not self.traces_enabled:
                return
            
            service_name = "videosdk-otel-telemetry-agents"

            if self.sdk_metadata and 'sdk' in self.sdk_metadata and 'sdk_version' in self.sdk_metadata:
                sdk_name = self.sdk_metadata['sdk'].upper()
                sdk_version = self.sdk_metadata['sdk_version']

            resource = Resource(attributes={
                "sdk.name": sdk_name,
                "sdk.version": sdk_version,
                "service.name": service_name,
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
            
        except Exception as e:
            print(f"[TELEMETRY ERROR] Failed to initialize telemetry: {e}")
    
    def trace(self, span_name: str, attributes: Dict[str, Any] = None, parent_span: Optional[Span] = None, start_time: Optional[float] = None) -> Optional[Span]:
        """
        Create a new trace span. If a parent is provided, the new span will be a
        child of it. Otherwise, it will be a child of the currently active span
        in the context.
        """
        if not self.traces_enabled or not self.tracer:
            return None
            
        try:
 
            ctx = trace.set_span_in_context(parent_span) if parent_span else None
            
            attiribute_id = generate_id()
            
            attributes = {} if attributes is None else attributes
            attributes["attiribute_id"] = attiribute_id
            span_kwargs = {"context": ctx}
            start_time = time.perf_counter() if start_time is None else start_time
            start_absolute_time = time.time_ns()
            self.span_details[attiribute_id] = {
                "start_time": start_time, # perf_counter
                "start_absolute_time": start_absolute_time # time.time()
            }
            span_kwargs["start_time"] = int(start_absolute_time)
            span = self.tracer.start_span(span_name, **span_kwargs)
                
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))
            
            return span
                
        except Exception as e:
            print(f"[TELEMETRY ERROR] Failed to create span '{span_name}': {e}")
            return None
    
    def complete_span(self, span: Optional[Span], status: StatusCode, message: str = "", end_time: Optional[float] = None):
        """
        Complete a span with status and message
        
        Args:
            span: Span to complete
            status: Status code
            message: Status message
            end_time: End time in seconds since epoch (optional)
        """
        if not self.traces_enabled or not span:
            return
            
        try:
            if message:
                span.set_attribute("message", message)
            span.set_status(Status(status, message))
            
            end_time = time.perf_counter() if end_time is None else end_time
            
            attribute_id = span._attributes["attiribute_id"]
            data = self.span_details.get(attribute_id)
            duration_ns = int((end_time - data["start_time"]) * 1_000_000_000)
            end_absolute_time = duration_ns + data["start_absolute_time"] # time.time()
            span.end(int(end_absolute_time))
            
            del self.span_details[attribute_id]
                
        except Exception as e:
            print(f"[TELEMETRY ERROR] Failed to complete span: {e}")
            traceback.print_exc()
    
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
                        metadata: Dict[str, Any] = None, sdk_metadata: Dict[str, Any] = None):
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
        metadata=metadata,
        sdk_metadata=sdk_metadata
    )