"""
Job-scoped logging with Producer-Consumer pattern.

Architecture:
    ChildProcess (Job)
    └── QueueHandler on 'videosdk.agents' logger  (producer – intercepts ALL
        existing logger.info / .debug / .warning / .error calls)
            ↓
    multiprocessing Manager().Queue  (transport)
            ↓
    QueueListener in Worker process  (consumer thread)
            ↓
    LogConsumerHandler  → HTTP POST to audit-log endpoint
"""

import logging
import multiprocessing
import threading
import time
import datetime
from logging.handlers import QueueHandler, QueueListener
from typing import Dict, Any, Optional, List


# ---------------------------------------------------------------------------
# 1. JobLogger – Producer (runs inside each Job / child process)
# ---------------------------------------------------------------------------

class _JobContextFilter(logging.Filter):
    """
    Attaches job context (roomId, sessionId, …) to every LogRecord
    that passes through the QueueHandler.
    """

    def __init__(self, context: Dict[str, Any]):
        super().__init__()
        self.context = context

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in self.context.items():
            setattr(record, key, value)
        return self.context.get("send_logs_to_dashboard", False)


class JobLogger:
    """
    Intercepts ALL existing ``logger.info()`` / ``.debug()`` / ``.warning()``
    etc. calls under the ``videosdk.agents`` namespace by attaching a
    ``QueueHandler`` to the parent logger.

    * Does ZERO I/O – records are written to a ``multiprocessing.Queue``.
    * Attaches job context (roomId, sessionId, peerId, …) via a Filter.
    """

    def __init__(
        self,
        queue,
        room_id: str,
        peer_id: str,
        auth_token: str,
        session_id: Optional[str] = None,
        dashboard_log_level: str = "INFO",
        sdk_metadata: Optional[Dict[str, Any]] = None,
        send_logs_to_dashboard: bool = False,
    ):
        self._queue = queue  # Keep ref for set_endpoint() config records
        self._context: Dict[str, Any] = {
            "roomId": room_id,
            "peerId": peer_id,
            "authToken": auth_token,
            "sessionId": session_id or "",
            "sdk_name": "",
            "sdk_version": "",
            "service_name": "videosdk-otel-telemetry-agents",
            "send_logs_to_dashboard": send_logs_to_dashboard,
        }

        if sdk_metadata:
            self._context["sdk_name"] = sdk_metadata.get("sdk", "AGENTS").upper()
            self._context["sdk_version"] = sdk_metadata.get("sdk_version", "0.0.62")

        # ---- Attach to the parent logger for the SDK namespace ----
        self._parent_logger = logging.getLogger("videosdk.agents")

        # In a spawned child process the logging hierarchy starts fresh.
        # The effective level defaults to WARNING (root), which would
        # silently drop INFO records BEFORE they reach any handler.
        # Lower the parent logger's level so records can flow through.
        target_level = getattr(logging, dashboard_log_level.upper(), logging.INFO)
        if self._parent_logger.getEffectiveLevel() > target_level:
            self._parent_logger.setLevel(target_level)

        # QueueHandler: forwards records to the shared queue
        self._queue_handler = QueueHandler(queue)
        # Only forward records at or above the dashboard level
        self._queue_handler.setLevel(target_level)

        # Context filter: attaches roomId/peerId/etc to each record
        self._context_filter = _JobContextFilter(self._context)
        self._queue_handler.addFilter(self._context_filter)

        # Add the handler to the parent logger
        self._parent_logger.addHandler(self._queue_handler)

    # -- Public API ----------------------------------------------------------

    def update_context(self, **kwargs) -> None:
        """Update context fields (e.g. sessionId once available)."""
        self._context.update(kwargs)

    def set_endpoint(self, endpoint: str, jwt_key: str = "") -> None:
        """
        Send the log endpoint and observability JWT to the consumer
        (Worker process) via the queue.
        Called from integration.py when room attributes become available.
        """
        # Create a special config record that the consumer will recognize
        record = logging.LogRecord(
            name="videosdk.agents._config",
            level=logging.INFO,
            pathname="", lineno=0,
            msg="", args=(), exc_info=None,
        )
        record._config_endpoint = endpoint
        record._config_jwt_key = jwt_key
        # Put directly on the queue (bypass QueueHandler)
        self._queue.put_nowait(record)

    def cleanup(self) -> None:
        """Remove the QueueHandler from the parent logger (called on job end)."""
        self._parent_logger.removeHandler(self._queue_handler)


# ---------------------------------------------------------------------------
# 2. LogConsumerHandler – Consumer (runs in Worker process, QueueListener thread)
# ---------------------------------------------------------------------------

class LogConsumerHandler(logging.Handler):
    """
    Receives ``LogRecord`` objects via ``QueueListener`` and POSTs them
    to the audit-log endpoint.

    Runs in the QueueListener's daemon thread (NOT an asyncio loop),
    so uses synchronous ``requests.post``.
    """

    def __init__(self, auth_token: str):
        super().__init__()
        self.endpoint: str = ""
        self.jwt_key: str = ""  # observability JWT from room attributes
        self.auth_token = auth_token
        self.auth_token = auth_token
        
        # Buffer for background flushing
        self._buffer: List[logging.LogRecord] = []
        self._buffer_lock = threading.Lock()
        
        # Pending records (waiting for endpoint)
        self._pending_records: List[logging.LogRecord] = []
        self._pending_lock = threading.Lock()
        
        # Background flush thread
        self._stop_event = threading.Event()
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True, name="LogFlushThread")
        self._flush_thread.start()

        self.log_count = 0
    def update_endpoint(self, endpoint: str, jwt_key: str = "") -> None:
        """Called when room attributes provide the real endpoint."""
        self.endpoint = endpoint
        if jwt_key:
            self.jwt_key = jwt_key
        # Flush any buffered records now that we have an endpoint
        # Flush any buffered records from pending to active buffer
        if self.endpoint:
            with self._pending_lock:
                pending = self._pending_records[:]
                self._pending_records.clear()
            
            if pending:
                with self._buffer_lock:
                    self._buffer.extend(pending)

    def emit(self, record: logging.LogRecord) -> None:
        """
        Called by QueueListener for every log record pulled from the queue.
        Runs in the QueueListener's daemon thread.
        """
        # Check for config records (endpoint update from child process)
        if hasattr(record, '_config_endpoint'):
            jwt_key = getattr(record, '_config_jwt_key', '')
            self.update_endpoint(record._config_endpoint, jwt_key)
            return

        if not self.endpoint:
            # Keep in pending until endpoint is known
            with self._pending_lock:
                self._pending_records.append(record)
            return

        # Add to active buffer for background flushing
        should_flush = False
        with self._buffer_lock:
            self.log_count += 1
            if self.log_count % 10 == 0:
                print("****"*100)
                print(f"Log count: {self.log_count}")
                print("****"*100)
            self._buffer.append(record)
            if len(self._buffer) >= 50:
                should_flush = True
        
        if should_flush:
            self._flush_buffer()

    def _flush_loop(self) -> None:
        """Background thread loop: flush logs every 5 seconds."""
        while not self._stop_event.is_set():
            # Wait 5 seconds or until stopped
            if self._stop_event.wait(5.0):
                break
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Send all records currently in the buffer."""
        # Swap buffer to minimize lock time
        with self._buffer_lock:
            if not self._buffer:
                return
            batch = self._buffer[:]
            self._buffer.clear()

        # Send records (sequentially for now as per API format)
        for record in batch:
            self._sync_push_log(record)

    def close(self) -> None:
        """Stop the background thread and flush remaining logs."""
        print("****"*100)
        print(f"Log count: {self.log_count}")
        print("****"*100)
        self._stop_event.set()
        if self._flush_thread.is_alive():
            self._flush_thread.join(timeout=2.0)
        self._flush_buffer()
        super().close()

    def _sync_push_log(self, record: logging.LogRecord) -> None:
        """Synchronous HTTP POST of a log record."""
        try:
            import requests as req_lib

            room_id = getattr(record, "roomId", "")
            peer_id = getattr(record, "peerId", "")
            session_id = getattr(record, "sessionId", "")
            sdk_name = getattr(record, "sdk_name", "") or "AGENTS"
            sdk_version = getattr(record, "sdk_version", "") or "0.0.62"
            service_name = getattr(record, "service_name", "videosdk-otel-telemetry-agents")
            log_attributes = getattr(record, "log_attributes", {})

            # Format timestamp for log message and attributes
            created_dt = datetime.datetime.fromtimestamp(record.created)
            time_str = created_dt.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
            
            attributes = {
                "roomId": room_id,
                "peerId": peer_id,
                "sessionId": session_id,
                "sdk.name": sdk_name,
                "sdk.version": sdk_version,
                "service.name": service_name,
                # New attributes requested by user
                "dateTime": time_str,
                "origin": record.name,
                "level": record.levelname,
            }

            if log_attributes:
                attributes.update(log_attributes)

            body = {
                "logType": record.levelname,
                "logText": record.getMessage(),
                "attributes": attributes,
                "debugMode": False,
                "dashboardLog": False,
                "serviceName": service_name,
            }

            headers = {
                "Authorization": self.jwt_key,
                "Content-Type": "application/json",
            }
            res = req_lib.post(self.endpoint, json=body, headers=headers, timeout=10)
            if res.status_code != 200:
                print("Failed to send log to endpoint")
            else:
                print("Log sent successfully", res)
        except Exception as e:
            pass


# ---------------------------------------------------------------------------
# 3. LogManager – ties QueueListener + Handler together (Worker process)
# ---------------------------------------------------------------------------

class LogManager:
    """
    Runs in the **Worker** process.

    Owns:
    * A ``Manager().Queue()`` proxy (picklable, shareable across processes).
    * A ``QueueListener`` that consumes from the queue and dispatches to
      ``LogConsumerHandler``.
    """

    def __init__(self):
        self._manager = multiprocessing.Manager()
        self.queue = self._manager.Queue()
        self._listener: Optional[QueueListener] = None
        self._consumer_handler: Optional[LogConsumerHandler] = None

    def start(self, auth_token: str) -> None:
        """Start consuming from the queue."""
        self._consumer_handler = LogConsumerHandler(auth_token=auth_token)
        self._listener = QueueListener(
            self.queue, self._consumer_handler, respect_handler_level=True
        )
        self._listener.start()

    def update_endpoint(self, endpoint: str) -> None:
        """Set/update the endpoint and flush buffered records."""
        if self._consumer_handler:
            self._consumer_handler.update_endpoint(endpoint)

    def stop(self) -> None:
        """Stop the listener and manager gracefully."""
        if self._listener:
            self._listener.stop()
            self._listener = None
        try:
            self._manager.shutdown()
        except Exception:
            pass

    def get_queue(self):
        """Return the shared queue for child Jobs."""
        return self.queue