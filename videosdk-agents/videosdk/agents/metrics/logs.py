from datetime import datetime
from typing import Dict, Any, Optional
from opentelemetry.trace import get_current_span
import asyncio
import aiohttp


class VideoSDKLogs:
    """VideoSDK logs for agents using direct API calls"""

    def __init__(self, meeting_id: str, peer_id: str, jwt_key: str, log_config: Dict[str, Any], session_id: str = None, sdk_metadata: Dict[str, Any] = None):
        """
        Initialize logs with direct API configuration

        Args:
            meeting_id: Meeting/room ID
            peer_id: Peer/participant ID
            jwt_key: JWT authentication key
            log_config: Log configuration with 'enabled' and 'endPoint'
            session_id: Session ID from the job/room
        """
        self.meeting_id = meeting_id
        self.peer_id = peer_id
        self.jwt_key = jwt_key
        self.log_config = log_config
        self.logs_enabled = log_config.get('enabled', False)
        self.endpoint = log_config.get('endPoint')

        self.session_id = session_id or f"session_{peer_id}_{int(datetime.now().timestamp())}"

        if sdk_metadata and 'sdk' in sdk_metadata and 'sdk_version' in sdk_metadata:
            sdk_name = sdk_metadata['sdk'].upper()
            sdk_version = sdk_metadata['sdk_version']

        service_name = "videosdk-otel-telemetry-agents"

        self.sdk_info = {
            "sdk.name": sdk_name,
            "sdk.version": sdk_version,
            "service.name": service_name,
        }

        self._initialize_logs()

    def _initialize_logs(self):
        """Initialize logs configuration"""
        if not self.logs_enabled:
            return

        if not self.endpoint:
            return

    def push_logs(self, log_type: str, log_text: str, attributes: Dict[str, Any] = None):
        """
        Non-blocking push_logs that creates an async task

        Args:
            log_type: Type of log (INFO, ERROR, DEBUG, WARN)
            log_text: Log message text
            attributes: Additional attributes for the log
        """
        if not self.logs_enabled or not self.endpoint:
            return

        try:
            asyncio.create_task(self._push_logs_async(
                log_type, log_text, attributes))
        except RuntimeError:
            pass

    async def _push_logs_async(self, log_type: str, log_text: str, attributes: Dict[str, Any] = None):
        """
        Asynchronously push logs to the endpoint using aiohttp.
        """
        if not self.logs_enabled or not self.endpoint:
            return

        try:
            trace_id = attributes.get("traceId") if attributes else None
            span_id = attributes.get("spanId") if attributes else None
            log_attributes = {
                "roomId": self.meeting_id,
                "peerId": self.peer_id,
                "sessionId": self.session_id,
                "traceId": trace_id,
                "spanId": span_id,
                **self.sdk_info,
            }

            if attributes:
                log_attributes.update(attributes)

            body = {
                "logType": log_type,
                "logText": log_text,
                "attributes": log_attributes,
                "debugMode": False,
                "dashboardLog": False,
                "serviceName": "videosdk-otel-telemetry-agents"
            }

            headers = {
                "Authorization": self.jwt_key,
                "Content-Type": "application/json",
            }

            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.endpoint,
                    json=body,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        response_text = await response.text()
                        print(
                            f"[LOGS ERROR] HTTP {response.status}: {response_text}")
                        return {}

        except Exception as e:
            print(f"[LOGS EXCEPTION] Error pushing log: {e}")
            return {}

    def create_log(self, message: str, log_level: str, attributes: Dict[str, Any] = None):
        """
        Create a log entry (compatibility method)

        Args:
            message: Log message
            log_level: Log level (DEBUG, INFO, WARN, ERROR)
            attributes: Additional attributes
        """

        span = get_current_span()
        span_context = span.get_span_context() if span else None

        if attributes is None:
            attributes = {}

        if span_context and span_context.is_valid:
            attributes["traceId"] = format(span_context.trace_id, "032x")
            attributes["spanId"] = format(span_context.span_id, "016x")

        self.push_logs(
            log_type=log_level,
            log_text=message,
            attributes=attributes
        )

    def shutdown(self):
        """Shutdown logs (no-op for direct API calls)"""
        pass

    def flush(self):
        """Flush logs (no-op for direct API calls)"""
        pass


# Global logs instance
_logs_instance: Optional[VideoSDKLogs] = None


def get_logs() -> Optional[VideoSDKLogs]:
    """Get the global logs instance"""
    return _logs_instance


def initialize_logs(meeting_id: str, peer_id: str, jwt_key: str = None,
                    log_config: Dict[str, Any] = None, session_id: str = None, sdk_metadata: Dict[str, Any] = None):
    """
    Initialize global logs instance

    Args:
        meeting_id: Meeting/room ID
        peer_id: Peer/participant ID
        jwt_key: JWT authentication key
        log_config: Log configuration with 'endPoint' and 'enabled'
        session_id: Session ID from the job/room
    """
    global _logs_instance

    if not jwt_key:
        jwt_key = ""

    if not log_config:
        log_config = {"enabled": False, "endPoint": ""}

    _logs_instance = VideoSDKLogs(
        meeting_id=meeting_id,
        peer_id=peer_id,
        jwt_key=jwt_key,
        log_config=log_config,
        session_id=session_id,
        sdk_metadata=sdk_metadata
    )


def shutdown_logs():
    """Shutdown the global logs instance"""
    global _logs_instance
    if _logs_instance:
        _logs_instance.shutdown()
        _logs_instance = None
