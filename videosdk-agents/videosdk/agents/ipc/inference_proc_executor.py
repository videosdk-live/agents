"""
Inference Process Executor for VideoSDK Agents IPC.

This module handles inference process creation, management, and communication,
similar to implementation but adapted for VideoSDK.
"""

import asyncio
import multiprocessing
import os
import signal
import sys
import time
from typing import Any, Callable, Dict, Optional
import logging
import json
from multiprocessing.connection import Connection

logger = logging.getLogger(__name__)


class InferenceProcExecutor:
    """
    Manages a single inference process for VideoSDK agents.

    Similar to InferenceProcExecutor but adapted for VideoSDK.
    """

    def __init__(
        self,
        *,
        initialize_timeout: float = 30.0,  # Longer timeout for AI model loading
        close_timeout: float = 5.0,
        memory_warn_mb: float = 1000.0,  # Higher threshold for AI models
        memory_limit_mb: float = 0.0,
        ping_interval: float = 30.0,
    ):
        self.initialize_timeout = initialize_timeout
        self.close_timeout = close_timeout
        self.memory_warn_mb = memory_warn_mb
        self.memory_limit_mb = memory_limit_mb
        self.ping_interval = ping_interval

        # Process management
        self.process: Optional[multiprocessing.Process] = None
        self.parent_conn: Optional[Connection] = None
        self.child_conn: Optional[Connection] = None
        self._available = True
        self._initialized = False
        self._shutdown = False

        # Health monitoring
        self.last_ping = time.time()
        self.memory_usage = 0.0

    async def initialize(self):
        """Initialize the inference process."""
        if self._initialized:
            return

        logger.info("Initializing inference process")

        # Create pipe for communication
        self.parent_conn, self.child_conn = multiprocessing.Pipe()

        # Start the process
        self.process = multiprocessing.Process(
            target=self._run_inference_process, args=(self.child_conn,), daemon=True
        )
        self.process.start()

        # Wait for initialization
        try:
            await asyncio.wait_for(
                self._wait_for_ready(), timeout=self.initialize_timeout
            )
            self._initialized = True
            logger.info(f"Inference process initialized (PID: {self.process.pid})")
        except asyncio.TimeoutError:
            logger.error("Inference process initialization timeout")
            await self.shutdown()
            raise

    def _run_inference_process(self, conn: Connection):
        """Run the inference process in a separate process."""
        try:
            # Set up signal handlers
            signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))

            # Import and run the inference process main
            from .inference_proc_lazy_main import run_inference_process_main

            run_inference_process_main(conn)

        except Exception as e:
            logger.error(f"Error in inference process: {e}")
            conn.send({"type": "error", "error": str(e)})
            sys.exit(1)

    async def _wait_for_ready(self):
        """Wait for the process to be ready."""
        while True:
            if self.parent_conn.poll():
                message = self.parent_conn.recv()
                if message.get("type") == "ready":
                    return
                elif message.get("type") == "error":
                    raise Exception(f"Inference process error: {message.get('error')}")
            await asyncio.sleep(0.1)

    def is_available(self) -> bool:
        """Check if the process is available for work."""
        if not self._initialized or self._shutdown:
            return False

        # Check if process is still alive
        if self.process and not self.process.is_alive():
            self._available = False
            return False

        # Check memory usage
        if self.memory_limit_mb > 0 and self.memory_usage > self.memory_limit_mb:
            logger.warning(
                f"Inference process memory limit exceeded: {self.memory_usage}MB"
            )
            self._available = False
            return False

        return self._available

    async def execute_inference(self, inference_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an inference task in this process."""
        if not self.is_available():
            raise Exception("Inference process not available")

        self._available = False

        try:
            # Send inference task to process
            self.parent_conn.send({"type": "inference", "data": inference_data})

            # Wait for result
            while True:
                if self.parent_conn.poll():
                    message = self.parent_conn.recv()
                    if message.get("type") == "result":
                        self._available = True
                        return message.get("data", {})
                    elif message.get("type") == "error":
                        self._available = True
                        raise Exception(
                            f"Inference execution error: {message.get('error')}"
                        )
                await asyncio.sleep(0.1)

        except Exception as e:
            self._available = True
            raise

    async def ping(self):
        """Send ping to check process health."""
        if not self._initialized or self._shutdown:
            return

        try:
            self.parent_conn.send({"type": "ping"})
            self.last_ping = time.time()

            # Wait for pong
            if self.parent_conn.poll(timeout=5.0):
                message = self.parent_conn.recv()
                if message.get("type") == "pong":
                    # Update memory usage
                    self.memory_usage = message.get("memory_usage", 0.0)

                    # Check memory warning
                    if (
                        self.memory_warn_mb > 0
                        and self.memory_usage > self.memory_warn_mb
                    ):
                        logger.warning(
                            f"Inference process memory usage high: {self.memory_usage}MB"
                        )

        except Exception as e:
            logger.error(f"Error pinging inference process: {e}")

    async def shutdown(self):
        """Shutdown the inference process."""
        if self._shutdown:
            return

        logger.info("Shutting down inference process")
        self._shutdown = True
        self._available = False

        try:
            # Send shutdown signal
            if self.parent_conn:
                self.parent_conn.send({"type": "shutdown"})

            # Wait for process to terminate
            if self.process:
                self.process.join(timeout=self.close_timeout)

                # Force kill if still alive
                if self.process.is_alive():
                    logger.warning("Force killing inference process")
                    self.process.terminate()
                    self.process.join(timeout=5.0)
                    if self.process.is_alive():
                        self.process.kill()

            # Close connections
            if self.parent_conn:
                self.parent_conn.close()
            if self.child_conn:
                self.child_conn.close()

        except Exception as e:
            logger.error(f"Error shutting down inference process: {e}")

        self._initialized = False
        logger.info("Inference process shutdown complete")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about this inference process."""
        return {
            "pid": self.process.pid if self.process else None,
            "available": self._available,
            "initialized": self._initialized,
            "memory_usage": self.memory_usage,
            "last_ping": self.last_ping,
            "alive": self.process.is_alive() if self.process else False,
        }
