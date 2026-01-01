import logging
import threading
import contextvars
import time
import os
from typing import Optional
import json
import importlib.metadata
import asyncio
from .integration import create_log

_target_loop = None
_is_sending_log = contextvars.ContextVar("is_sending_log", default=False)
details = {}

ALLOWED_NAMESPACES = [
    "videosdk.agents", 
    "__main__", 
]

NOT_ALLOWED_NAMESPACES = [
    "videosdk.agents.room.audio_stream",
    "videosdk.agents.room.video_stream",
]

class BufferedLogHandler(logging.Handler):
    def __init__(
        self, 
        flush_interval=1, 
    ):
        super().__init__()
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
        
        self._buffer = []
        self._buffer_lock = threading.Lock()
        self.flush_interval = flush_interval

        try:
            self._main_loop = asyncio.get_event_loop()
        except RuntimeError:
            self._main_loop = None

        self._stop_event = threading.Event()
        self._flusher_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flusher_thread.start()

    def emit(self, record):
        if _is_sending_log.get():
            return

        if any(record.name.startswith(ns) for ns in NOT_ALLOWED_NAMESPACES):
            return

        if not any(record.name.startswith(ns) for ns in ALLOWED_NAMESPACES):
            return

        try:
            log_entry = {
                "level": record.levelname,
                "logs": self.format(record)
            }
            
            with self._buffer_lock:
                self._buffer.append(log_entry)
                
        except Exception as e:
            self.handleError(record)
            print("Error in log handler", e)

    def _flush_loop(self):
        """Runs in background: sleeps -> checks details -> sends batch."""
        while not self._stop_event.is_set():
            time.sleep(self.flush_interval) 
            
            if not details:
                continue 

            batch_to_send = []
            with self._buffer_lock:
                if self._buffer:
                    batch_to_send = self._buffer[:]
                    self._buffer.clear()
            
            if batch_to_send:
                self._send_batch(batch_to_send)

    def _send_batch(self, batch):
        token = _is_sending_log.set(True)
        try:
            current_details = details.copy()
            
            loop = _target_loop if (_target_loop and not _target_loop.is_closed()) else self._main_loop

            if loop and loop.is_running():
                for entry in batch:
                    loop.call_soon_threadsafe(
                        create_log,
                        entry["logs"],
                        entry["level"],
                        current_details
                    )
            else:
                pass

        except Exception as e:
            pass
        finally:
            _is_sending_log.reset(token)
    
    def close(self):
        self._stop_event.set()
        self._flusher_thread.join(timeout=1)
        super().close()

def attach_analytics():
    """Attach the log handler to the root logger."""
    root = logging.getLogger()

    current_effective_level = root.getEffectiveLevel()
    for h in root.handlers:
        if isinstance(h, BufferedLogHandler): continue
        if h.level == logging.NOTSET:
            h.setLevel(current_effective_level)

    root.setLevel(logging.DEBUG)

    if not any(isinstance(h, BufferedLogHandler) for h in root.handlers):
        api_h = BufferedLogHandler(flush_interval=1)
        api_h.setLevel(logging.DEBUG) 
        root.addHandler(api_h)

def get_details(session_details: dict, loop: Optional[asyncio.AbstractEventLoop] = None):
    """
    Populates the global details. 
    Once this is called, the BufferHandler will release the held logs.
    """
    global details, _target_loop
    if loop:
        _target_loop = loop
    # Clean and format the details before saving
    formatted_details = format_session_details(session_details)
    details.update(formatted_details)

def format_session_details(raw_options: dict) -> dict:
    """
    Filters raw room options into the exact camelCase attributes 
    expected by the OtelController backend.
    """
    
    # Attempt to get version, fallback if package not installed (e.g. dev mode)
    try:
        sdk_version = importlib.metadata.version("videosdk-agents")
    except importlib.metadata.PackageNotFoundError:
        sdk_version = "0.0.0-dev"

    attributes = {
        # 1. Required by SessionErrorLog.create
        "roomId": raw_options.get("room_id"),
        "peerId": raw_options.get("agent_participant_id"),
        "sessionId": raw_options.get("session_id"),
        "SDK": "python-agent",
        "SDK_VERSION": sdk_version,
        "error": None, 

        # 2. Optional / Metadata
        "agentName": raw_options.get("name"), 
        "playground": raw_options.get("playground", False),
    }

    return {k: v for k, v in attributes.items() if v is not None}