from __future__ import annotations

import os
import time
import requests
from typing import Any, Dict, Optional, Union
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode, Span


class VideoSDKTelemetry:
    """
    Central telemetry manager for VideoSDK Agents
    
    Provides a clean API for distributed tracing with automatic span hierarchy management.
    """
    
    def __init__(
        self, 
        meeting_id: str, 
        peer_id: str, 
        sdk_name: str = "videosdk-agents", 
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize VideoSDK telemetry
        
        Args:
            meeting_id: The meeting/room identifier
            peer_id: The participant/agent identifier  
            sdk_name: Name of the SDK component
            metadata: Additional metadata (userId, email, etc.)
        """
        self.meeting_id = meeting_id
        self.peer_id = peer_id
        self.sdk_name = sdk_name
        self.metadata = metadata or {}
        self.traces_enabled = True
        self.tracer: Optional[trace.Tracer] = None
        self.root_span: Optional[Span] = None
        self.tracer_provider: Optional[TracerProvider] = None
        
        try:
            self._initialize_telemetry()
        except Exception as e:
            print(f"Error initializing VideoSDK telemetry: {e}")
            self.traces_enabled = False
    
    def _initialize_telemetry(self) -> None:
        """Initialize OpenTelemetry configuration - matching traces.py format"""
        # Get configuration from environment (same logic as traces.py)
        traces_endpoint = os.getenv("OTLP_BASE_URL_TRACES") or os.getenv("OTLP_BASE_URL")
        api_key = os.getenv("OTLP_API_KEY")
        
        if not traces_endpoint:
            print("⚠️ No OTLP endpoint configured, disabling telemetry")
            self.traces_enabled = False
            return
            
        # Ensure endpoint has proper protocol (same as traces.py)
        if traces_endpoint and not traces_endpoint.startswith("http"):
            traces_endpoint = f"https://{traces_endpoint}"
        
        # Configure resource with service information (matching traces.py format)
        resource_attributes = {
            "service.name": os.getenv("OTEL_SERVICE_NAME", "videosdk-agents"),
            "service.version": os.getenv("OTEL_SERVICE_VERSION", "1.0.0"),
            "deployment.environment": os.getenv("OTEL_ENVIRONMENT", "development"),
            # Add our custom attributes
            "sdk.name": self.sdk_name,
            "meeting.id": self.meeting_id,
            "peer.id": self.peer_id,
        }
        
        # Add metadata attributes
        for key, value in self.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                resource_attributes[f"user.{key}"] = str(value)
        
        resource = Resource.create(resource_attributes)
        
        # Configure OTLP exporter (matching traces.py format)
        headers = {}
        if api_key:
            headers["api-key"] = api_key  # Use same header format as traces.py
        
        # Check if we need to disable SSL for testing (same as traces.py)
        disable_ssl = os.getenv("OTLP_DISABLE_SSL", "false").lower() in ("true", "1", "yes")
        
        exporter_options = {
            "endpoint": traces_endpoint,
            "headers": headers,
            "timeout": 30
        }
        
        # Add SSL workaround for testing if needed (same as traces.py)
        if disable_ssl:
            print("⚠️  WARNING: SSL verification disabled for testing - DO NOT use in production!")
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            session = requests.Session()
            session.verify = False
            exporter_options["session"] = session
        
        try:
            exporter = OTLPSpanExporter(**exporter_options)
            
            # Create tracer provider with resource
            self.tracer_provider = TracerProvider(resource=resource)
            
            processor = BatchSpanProcessor(exporter)
            self.tracer_provider.add_span_processor(processor)
            
            # Set global tracer provider
            trace.set_tracer_provider(self.tracer_provider)
            
            # Get tracer instance
            self.tracer = trace.get_tracer(self.peer_id)
            
            # Create root span for the session
            self._create_root_span()
            
            ssl_status = "SSL disabled" if disable_ssl else "SSL enabled"
            print(f" VideoSDK Telemetry initialized for meeting: {self.meeting_id} ({ssl_status})")
            
        except Exception as e:
            print(f" Failed to configure VideoSDK telemetry: {e}")
            print("   Telemetry will be disabled for this session.")
            self.traces_enabled = False
    
    def _create_root_span(self) -> None:
        """Create the root span for this agent session"""
        if not self.tracer:
            return
            
        span_name = f"meeting_{self.meeting_id}_agent_{self.peer_id}"
        self.root_span = self.tracer.start_span(span_name)
        
        # Set root span attributes
        if self.root_span:
            self.root_span.set_attributes({
                "meeting.id": self.meeting_id,
                "peer.id": self.peer_id,
                "sdk.name": self.sdk_name,
                "session.start_time": time.time(),
                **{f"user.{k}": str(v) for k, v in self.metadata.items() 
                   if isinstance(v, (str, int, float, bool))}
            })
    
    def trace(
        self, 
        span_name: str, 
        attributes: Optional[Dict[str, Any]] = None, 
        parent_span: Optional[Span] = None
    ) -> Optional[Span]:
        """
        Create a new span with optional parent
        
        Args:
            span_name: Name of the span
            attributes: Key-value attributes to add
            parent_span: Parent span (uses current or root if None)
            
        Returns:
            The created span or None if tracing disabled
        """
        if not self.traces_enabled or not self.tracer:
            return None
            
        try:
            # Determine parent span
            if parent_span is None:
                parent_span = self.get_current_span() or self.root_span
            
            # Create span
            if parent_span:
                ctx = trace.set_span_in_context(parent_span)
                span = self.tracer.start_span(span_name, context=ctx)
            else:
                span = self.tracer.start_span(span_name)
            
            # Add attributes
            if attributes and span:
                # Convert all values to strings for safety
                safe_attributes = {}
                for key, value in attributes.items():
                    if isinstance(value, (str, int, float, bool)):
                        safe_attributes[key] = value
                    elif value is not None:
                        safe_attributes[key] = str(value)
                
                span.set_attributes(safe_attributes)
            
            return span
            
        except Exception as e:
            print(f"Error creating span '{span_name}': {e}")
            return None
    
    def trace_auto_complete(
        self,
        span_name: str,
        attributes: Optional[Dict[str, Any]] = None,
        parent_span: Optional[Span] = None,
        status: StatusCode = StatusCode.OK,
        message: Optional[str] = None
    ) -> None:
        """
        Create and immediately complete a span (for simple operations)
        
        Args:
            span_name: Name of the span
            attributes: Key-value attributes
            parent_span: Parent span
            status: Span status (OK or ERROR)
            message: Optional status message
        """
        span = self.trace(span_name, attributes, parent_span)
        if span:
            self.complete_span(span, status, message)
    
    def complete_span(
        self, 
        span: Optional[Span], 
        status: StatusCode = StatusCode.OK, 
        message: Optional[str] = None
    ) -> None:
        """
        Complete a span with status and optional message
        
        Args:
            span: The span to complete
            status: Status code (OK or ERROR)
            message: Optional message
        """
        if not self.traces_enabled or not span:
            return
            
        try:
            # Set status
            if status == StatusCode.ERROR:
                span.set_status(Status(StatusCode.ERROR, message or "Operation failed"))
            else:
                # For OK status, don't set description unless there's an error
                span.set_status(Status(StatusCode.OK))
            
            # Add message as attribute if provided
            if message:
                span.set_attribute("message", message)
                
            # End the span
            span.end()
            
        except Exception as e:
            print(f"Error completing span: {e}")
    
    def get_current_span(self) -> Optional[Span]:
        """Get the currently active span"""
        if not self.traces_enabled:
            return None
            
        current = trace.get_current_span()
        if current and current.is_recording():
            return current
        return self.root_span
    
    def get_current_span_name(self) -> Optional[str]:
        """Get the name of the current span"""
        current = self.get_current_span()
        return getattr(current, 'name', None) if current else None
    
    @contextmanager
    def span_context(
        self, 
        span_name: str, 
        attributes: Optional[Dict[str, Any]] = None,
        parent_span: Optional[Span] = None
    ):
        """
        Context manager for automatic span lifecycle management
        
        Args:
            span_name: Name of the span
            attributes: Span attributes
            parent_span: Parent span
            
        Usage:
            with telemetry.span_context("operation_name", {"key": "value"}):
                # Your code here
                pass
        """
        span = self.trace(span_name, attributes, parent_span)
        try:
            if span:
                with trace.use_span(span):
                    yield span
            else:
                yield None
        except Exception as e:
            if span:
                self.complete_span(span, StatusCode.ERROR, str(e))
            raise
        else:
            if span:
                self.complete_span(span, StatusCode.OK)
    
    def add_span_attribute(self, span: Optional[Span], key: str, value: Any) -> None:
        """Add an attribute to a span safely"""
        if span and self.traces_enabled:
            try:
                if isinstance(value, (str, int, float, bool)):
                    span.set_attribute(key, value)
                elif value is not None:
                    span.set_attribute(key, str(value))
            except Exception as e:
                print(f"Error setting span attribute {key}: {e}")
    
    def record_exception(self, span: Optional[Span], exception: Exception) -> None:
        """Record an exception in a span"""
        if span and self.traces_enabled:
            try:
                span.record_exception(exception)
                span.set_status(Status(StatusCode.ERROR, str(exception)))
            except Exception as e:
                print(f"Error recording exception: {e}")
    
    def flush(self) -> None:
        """Flush and shutdown telemetry"""
        if self.traces_enabled and self.tracer_provider:
            try:
                # Complete root span
                if self.root_span:
                    self.complete_span(self.root_span, StatusCode.OK, "Session completed")
                    
                # Shutdown tracer provider
                self.tracer_provider.shutdown()
                print("📊 VideoSDK Telemetry flushed")
            except Exception as e:
                print(f"Error flushing telemetry: {e}")


# Global telemetry instance
_global_telemetry: Optional[VideoSDKTelemetry] = None


def initialize_telemetry(
    meeting_id: str, 
    peer_id: str, 
    sdk_name: str = "videosdk-agents",
    metadata: Optional[Dict[str, Any]] = None
) -> VideoSDKTelemetry:
    """Initialize global telemetry instance"""
    global _global_telemetry
    _global_telemetry = VideoSDKTelemetry(meeting_id, peer_id, sdk_name, metadata)
    return _global_telemetry


def get_telemetry() -> Optional[VideoSDKTelemetry]:
    """Get the global telemetry instance"""
    return _global_telemetry


def cleanup_telemetry() -> None:
    """Cleanup global telemetry"""
    global _global_telemetry
    if _global_telemetry:
        _global_telemetry.flush()
        _global_telemetry = None 