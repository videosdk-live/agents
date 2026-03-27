
import logging
import queue
import threading
import datetime
from logging.handlers import QueueHandler, QueueListener
from typing import Dict, Any, Optional, List

class _JobContextFilter(logging.Filter):
    """
    Attaches job context (roomId, sessionId, …) to every LogRecord
    that passes through the QueueHandler.

    Returns True only when send_logs_to_dashboard is enabled, so records
    are forwarded to the queue only when the user has opted in.
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

    * Does ZERO I/O – records are written to a ``queue.Queue``.
    * Attaches job context (roomId, sessionId, peerId, …) via a Filter.
    * Thread-safe: queue.Queue is thread-safe; multiple jobs each own their
      own queue, so no cross-agent interference.
    """

    def __init__(
        self,
        queue: "queue.Queue",
        room_id: str,
        peer_id: str,
        auth_token: str,
        session_id: Optional[str] = None,
        dashboard_log_level: str = "INFO",
        sdk_metadata: Optional[Dict[str, Any]] = None,
        send_logs_to_dashboard: bool = False,
    ):
        self._queue = queue 
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

        self._parent_logger = logging.getLogger("videosdk.agents")

        target_level = getattr(logging, dashboard_log_level.upper(), logging.INFO)
        if self._parent_logger.getEffectiveLevel() > target_level:
            self._parent_logger.setLevel(target_level)

        self._queue_handler = QueueHandler(queue)
        self._queue_handler.setLevel(target_level)

        self._context_filter = _JobContextFilter(self._context)
        self._queue_handler.addFilter(self._context_filter)

        self._parent_logger.addHandler(self._queue_handler)

    def update_context(self, **kwargs) -> None:
        """Update context fields (e.g. sessionId once available)."""
        self._context.update(kwargs)

    def set_endpoint(self, endpoint: str, jwt_key: str = "", custom_headers: Optional[Dict[str, str]] = None) -> None:
        """
        Send the log endpoint and optional custom headers to the consumer thread
        via the queue so the LogConsumerHandler can start posting logs.

        Called from room.py when room attributes become available.
        """
        record = logging.LogRecord(
            name="videosdk.agents._config",
            level=logging.INFO,
            pathname="", lineno=0,
            msg="", args=(), exc_info=None,
        )
        record._config_endpoint = endpoint
        record._config_jwt_key = jwt_key
        record._config_custom_headers = custom_headers
        self._queue.put_nowait(record)

    def cleanup(self) -> None:
        """Remove the QueueHandler from the parent logger (called on job end)."""
        self._parent_logger.removeHandler(self._queue_handler)


class LogConsumerHandler(logging.Handler):
    """
    Receives ``LogRecord`` objects via ``QueueListener`` and POSTs them
    to the audit-log endpoint in configurable batches.

    Runs in the QueueListener's daemon thread (NOT an asyncio loop),
    so uses synchronous ``requests.post``.

    Batching strategy:
      * Buffer up to ``batch_size`` records, then flush immediately.
      * Also flush on a periodic ``flush_interval_seconds`` timer.
    """

    def __init__(
        self,
        auth_token: str = "",
        batch_size: int = 50,
        flush_interval_seconds: float = 5.0,
    ):
        super().__init__()
        self.endpoint: str = ""
        self.jwt_key: str = ""
        self.auth_token = auth_token
        self._batch_size = batch_size
        self._flush_interval = flush_interval_seconds
        self._buffer: List[logging.LogRecord] = []
        self._buffer_lock = threading.Lock()
        self._pending_records: List[logging.LogRecord] = []
        self._pending_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._flush_thread = threading.Thread(
            target=self._flush_loop,
            daemon=True,
            name="LogFlushThread",
        )
        self._flush_thread.start()
        self.custom_headers = None

    def update_endpoint(self, endpoint: str, jwt_key: str = "", custom_headers: Optional[Dict[str, str]] = None) -> None:
        """Called when room attributes provide the real endpoint."""
        self.endpoint = endpoint
        if jwt_key:
            self.jwt_key = jwt_key
        if custom_headers:
            self.custom_headers = custom_headers

        if self.endpoint:
            with self._pending_lock:
                pending = self._pending_records[:]
                self._pending_records.clear()

            if pending:
                with self._buffer_lock:
                    self._buffer.extend(pending)

    def emit(self, record: logging.LogRecord) -> None:
        """
        Called by QueueListener for every record pulled from the queue.
        Runs in the QueueListener's daemon thread.
        """
        if hasattr(record, "_config_endpoint"):
            jwt_key = getattr(record, "_config_jwt_key", "")
            custom_headers = getattr(record, "_config_custom_headers", None)
            self.update_endpoint(record._config_endpoint, jwt_key, custom_headers)
            return

        if not self.endpoint:
            with self._pending_lock:
                self._pending_records.append(record)
            return

        should_flush = False
        with self._buffer_lock:
            self._buffer.append(record)
            if len(self._buffer) >= self._batch_size:
                should_flush = True

        if should_flush:
            self._flush_buffer()

    def _flush_loop(self) -> None:
        """Background thread: flush buffered logs every flush_interval_seconds."""
        while not self._stop_event.is_set():
            if self._stop_event.wait(self._flush_interval):
                break
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Drain the buffer and POST each record to the endpoint."""
        with self._buffer_lock:
            if not self._buffer:
                return
            batch = self._buffer[:]
            self._buffer.clear()

        for record in batch:
            self._sync_push_log(record)

    def close(self) -> None:
        """Stop the background thread and flush any remaining logs."""
        self._stop_event.set()
        if self._flush_thread.is_alive():
            self._flush_thread.join(timeout=5.0)
        self._flush_buffer()
        super().close()

    def _sync_push_log(self, record: logging.LogRecord) -> None:
        """Synchronous HTTP POST of a single log record."""
        try:
            import requests as req_lib

            room_id = getattr(record, "roomId", "")
            peer_id = getattr(record, "peerId", "")
            session_id = getattr(record, "sessionId", "")
            sdk_name = getattr(record, "sdk_name", "") or "AGENTS"
            sdk_version = getattr(record, "sdk_version", "") or "0.0.62"
            service_name = getattr(record, "service_name", "videosdk-otel-telemetry-agents")
            log_attributes = getattr(record, "log_attributes", {})

            created_dt = datetime.datetime.fromtimestamp(record.created)
            time_str = created_dt.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]

            attributes = {
                "roomId": room_id,
                "peerId": peer_id,
                "sessionId": session_id,
                "sdk.name": sdk_name,
                "sdk.version": sdk_version,
                "service.name": service_name,
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
                "Content-Type": "application/json",
            }
            if self.custom_headers:
                headers.update(self.custom_headers)
            elif self.jwt_key:
                headers["Authorization"] = self.jwt_key

            req_lib.post(self.endpoint, json=body, headers=headers, timeout=10)
        except Exception:
            pass

class LogManager:
    """
    One instance per **Job** (not per Worker).

    Owns:
    * A ``queue.Queue`` (thread-safe, no extra manager process needed).
    * A ``QueueListener`` that consumes from the queue and dispatches to
      ``LogConsumerHandler``.

    Lifecycle::

        mgr = LogManager()
        mgr.start(auth_token="...")       # starts background listener thread
        # … job runs, logs are captured …
        mgr.stop()                        # flushes & tears down

    The ``queue`` attribute is passed to ``JobLogger`` (producer side).
    """

    def __init__(
        self,
        batch_size: int = 50,
        flush_interval_seconds: float = 5.0,
    ):
        self.queue: queue.Queue = queue.Queue()
        self._listener: Optional[QueueListener] = None
        self._consumer_handler: Optional[LogConsumerHandler] = None
        self._batch_size = batch_size
        self._flush_interval = flush_interval_seconds

    def start(self, auth_token: str = "") -> None:
        """Start consuming from the queue."""
        self._consumer_handler = LogConsumerHandler(
            auth_token=auth_token,
            batch_size=self._batch_size,
            flush_interval_seconds=self._flush_interval,
        )
        self._listener = QueueListener(
            self.queue, self._consumer_handler, respect_handler_level=True
        )
        self._listener.start()

    def update_endpoint(self, endpoint: str, jwt_key: str = "", custom_headers: Optional[Dict[str, str]] = None) -> None:
        """Set/update the endpoint and flush buffered records."""
        if self._consumer_handler:
            self._consumer_handler.update_endpoint(endpoint, jwt_key, custom_headers)

    def stop(self) -> None:
        """Stop the listener and flush remaining logs."""
        if self._listener:
            self._listener.stop()
            self._listener = None
        if self._consumer_handler:
            self._consumer_handler.close()
            self._consumer_handler = None

    def get_queue(self) -> queue.Queue:
        """Return the shared queue for the JobLogger (producer)."""
        return self.queue