"""
Dedicated inference resource for AI model processing.

This module provides a single shared inference process that handles all AI model
tasks (STT, LLM, TTS, VAD) similar to the old IPC system architecture.
"""

import asyncio
import logging
import multiprocessing
import os
import signal
import sys
import time
import uuid
from typing import Any, Dict, Optional, Callable
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection

from .base_resource import BaseResource
from .types import ResourceType, TaskResult, TaskStatus, ResourceStatus

logger = logging.getLogger(__name__)


class DedicatedInferenceResource(BaseResource):
    """
    Dedicated inference resource that runs AI models in a separate process.

    This mimics the old IPC system's single shared inference process that
    handles all STT, LLM, TTS, and VAD tasks for all agent jobs.
    """

    def __init__(self, resource_id: str, config: Dict[str, Any]):
        super().__init__(resource_id, config)
        self.process: Optional[Process] = None
        self.parent_conn: Optional[Connection] = None
        self.child_conn: Optional[Connection] = None
        self._process_ready = False
        self._models_cache: Dict[str, Any] = {}

        # Inference-specific configuration
        self.initialize_timeout = config.get("inference_process_timeout", 30.0)
        self.memory_warn_mb = config.get("inference_memory_warn_mb", 1000.0)
        self.ping_interval = config.get("ping_interval", 30.0)

    @property
    def resource_type(self) -> ResourceType:
        return ResourceType.PROCESS

    async def _initialize_impl(self) -> None:
        """Initialize the dedicated inference process."""
        logger.info(f"Initializing dedicated inference process: {self.resource_id}")

        # Create pipe for communication
        self.parent_conn, self.child_conn = Pipe()

        # Start the inference process
        self.process = Process(
            target=self._run_inference_process,
            args=(self.resource_id, self.child_conn, self.config),
            daemon=True,
        )
        self.process.start()

        # Wait for process to be ready
        start_time = time.time()
        while (
            not self._process_ready
            and (time.time() - start_time) < self.initialize_timeout
        ):
            try:
                if self.parent_conn.poll():
                    message = self.parent_conn.recv()
                    if message.get("type") == "ready":
                        self._process_ready = True
                        break
                    elif message.get("type") == "error":
                        raise Exception(
                            f"Inference process error: {message.get('error')}"
                        )

                await asyncio.sleep(0.1)
            except Exception as e:
                logger.warning(f"Error checking inference process readiness: {e}")

        if not self._process_ready:
            raise TimeoutError(
                f"Inference process {self.resource_id} failed to initialize within {self.initialize_timeout}s"
            )

        logger.info(
            f"Dedicated inference process initialized: {self.resource_id} (PID: {self.process.pid})"
        )

    async def _execute_task_impl(
        self, task_id: str, config, entrypoint: Callable, args: tuple, kwargs: dict
    ) -> Any:
        """Execute inference task in the dedicated process."""
        if not self._process_ready:
            raise RuntimeError(f"Inference process {self.resource_id} is not ready")

        # Prepare inference data
        inference_data = {
            "task_id": task_id,
            "task_type": config.task_type.value,
            "model_config": config.data.get("model_config", {}),
            "input_data": config.data.get("input_data", {}),
            "timeout": config.timeout,
        }

        # Send inference request to process
        self.parent_conn.send({"type": "inference", "data": inference_data})

        # Wait for result
        start_time = time.time()
        while (time.time() - start_time) < config.timeout:
            try:
                if self.parent_conn.poll():
                    message = self.parent_conn.recv()
                    if (
                        message.get("type") == "result"
                        and message.get("task_id") == task_id
                    ):
                        if message.get("status") == "success":
                            return message.get("result")
                        else:
                            raise RuntimeError(
                                message.get("error", "Unknown inference error")
                            )
                    elif message.get("type") == "error":
                        raise RuntimeError(
                            message.get("error", "Inference process error")
                        )

                await asyncio.sleep(0.1)
            except Exception as e:
                logger.warning(f"Error checking inference result: {e}")

        raise TimeoutError(
            f"Inference task {task_id} timed out after {config.timeout}s"
        )

    async def _shutdown_impl(self) -> None:
        """Shutdown the dedicated inference process."""
        if self.process and self.process.is_alive():
            # Send shutdown signal
            self.parent_conn.send({"type": "shutdown"})

            # Wait for graceful shutdown
            timeout = self.config.get("close_timeout", 60.0)
            start_time = time.time()

            while self.process.is_alive() and (time.time() - start_time) < timeout:
                await asyncio.sleep(0.1)

            # Force terminate if still alive
            if self.process.is_alive():
                logger.warning(
                    f"Force terminating inference process {self.resource_id}"
                )
                self.process.terminate()
                self.process.join(timeout=5.0)

                if self.process.is_alive():
                    self.process.kill()

    async def health_check(self) -> bool:
        """Perform a health check on the dedicated inference process."""
        try:
            if self._shutdown or not self.process or not self.process.is_alive():
                return False

            # Send ping to inference process
            self.parent_conn.send({"type": "ping"})

            # Wait for ping response
            start_time = time.time()
            timeout = 5.0  # 5 second timeout for health check

            while (time.time() - start_time) < timeout:
                try:
                    if self.parent_conn.poll():
                        message = self.parent_conn.recv()
                        if message.get("type") == "ping_response":
                            # Update last heartbeat
                            self.last_heartbeat = time.time()
                            return True
                        elif message.get("type") == "error":
                            logger.error(
                                f"Inference process error: {message.get('error')}"
                            )
                            return False

                    await asyncio.sleep(0.1)
                except Exception as e:
                    logger.warning(f"Error checking inference process health: {e}")

            # Timeout - process is unresponsive
            logger.warning(f"Inference process {self.resource_id} health check timeout")
            return False

        except Exception as e:
            logger.error(
                f"Health check failed for inference process {self.resource_id}: {e}"
            )
            return False

    @staticmethod
    def _run_inference_process(
        resource_id: str, conn: Connection, config: Dict[str, Any]
    ):
        """Run the inference process in a separate process."""
        try:
            # Set up logging
            logging.basicConfig(level=logging.INFO)
            logger.info(
                f"Inference process started: {resource_id} (PID: {os.getpid()})"
            )

            # Set up signal handlers
            def signal_handler(signum, frame):
                logger.info("Received shutdown signal")
                conn.send({"type": "shutdown_ack"})
                sys.exit(0)

            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)

            # Send ready signal
            conn.send({"type": "ready"})

            # Model cache for reuse
            models_cache: Dict[str, Any] = {}

            async def main_loop():
                while True:
                    try:
                        if conn.poll(timeout=1.0):
                            message = conn.recv()
                            message_type = message.get("type")

                            if message_type == "inference":
                                await _handle_inference(
                                    conn, message.get("data", {}), models_cache
                                )
                            elif message_type == "ping":
                                await _handle_ping(conn)
                            elif message_type == "shutdown":
                                logger.info("Received shutdown request")
                                conn.send({"type": "shutdown_ack"})
                                break
                            else:
                                logger.warning(f"Unknown message type: {message_type}")
                    except Exception as e:
                        logger.error(f"Error in inference process main loop: {e}")
                        conn.send({"type": "error", "error": str(e)})

            asyncio.run(main_loop())

        except Exception as e:
            logger.error(f"Fatal error in inference process: {e}")
            conn.send({"type": "error", "error": str(e)})
            sys.exit(1)
        finally:
            logger.info("Inference process shutting down")
            conn.close()


async def _handle_inference(
    conn: Connection, inference_data: Dict[str, Any], models_cache: Dict[str, Any]
):
    """Handle an inference request in the dedicated process."""
    try:
        task_id = inference_data.get("task_id")
        task_type = inference_data.get("task_type")  # stt, llm, tts, vad
        model_config = inference_data.get("model_config", {})
        input_data = inference_data.get("input_data", {})

        logger.info(f"Executing inference: {task_id} ({task_type})")

        # Get or create model
        model_key = f"{task_type}_{model_config.get('model_name', 'default')}"

        if model_key not in models_cache:
            logger.info(f"Loading model: {model_key}")
            models_cache[model_key] = await _load_model(task_type, model_config)

        model = models_cache[model_key]

        # Execute inference
        result = await _execute_inference(task_type, model, input_data)

        # Send result back
        conn.send(
            {
                "type": "result",
                "task_id": task_id,
                "status": "success",
                "result": result,
            }
        )

    except Exception as e:
        logger.error(f"Error in inference: {e}")
        conn.send(
            {
                "type": "result",
                "task_id": inference_data.get("task_id"),
                "status": "error",
                "error": str(e),
            }
        )


async def _load_model(task_type: str, model_config: Dict[str, Any]) -> Any:
    """Load an AI model based on task type."""
    try:
        if task_type == "stt":
            # Load STT model (Deepgram, OpenAI Whisper, etc.)
            logger.info(f"Loading STT model: {model_config}")
            # Placeholder - in real implementation, load actual STT model
            return {"type": "stt", "model": model_config.get("model_name", "deepgram")}

        elif task_type == "llm":
            # Load LLM model (OpenAI GPT, Anthropic Claude, etc.)
            logger.info(f"Loading LLM model: {model_config}")
            # Placeholder - in real implementation, load actual LLM model
            return {"type": "llm", "model": model_config.get("model_name", "gpt-4")}

        elif task_type == "tts":
            # Load TTS model (ElevenLabs, OpenAI TTS, etc.)
            logger.info(f"Loading TTS model: {model_config}")
            # Placeholder - in real implementation, load actual TTS model
            return {
                "type": "tts",
                "model": model_config.get("model_name", "elevenlabs"),
            }

        elif task_type == "vad":
            # Load VAD model (Silero, etc.)
            logger.info(f"Loading VAD model: {model_config}")
            # Placeholder - in real implementation, load actual VAD model
            return {"type": "vad", "model": model_config.get("model_name", "silero")}

        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


async def _execute_inference(
    task_type: str, model: Any, input_data: Dict[str, Any]
) -> Any:
    """Execute inference using the loaded model."""
    try:
        if task_type == "stt":
            # Execute STT inference
            audio_data = input_data.get("audio_data", b"")
            # Placeholder - in real implementation, run actual STT
            return {"transcript": "Hello, this is a test transcript"}

        elif task_type == "llm":
            # Execute LLM inference
            prompt = input_data.get("prompt", "")
            # Placeholder - in real implementation, run actual LLM
            return {"response": "This is a test response from the language model"}

        elif task_type == "tts":
            # Execute TTS inference
            text = input_data.get("text", "")
            # Placeholder - in real implementation, run actual TTS
            return {"audio": b"fake_audio_data"}

        elif task_type == "vad":
            # Execute VAD inference
            audio_data = input_data.get("audio_data", b"")
            # Placeholder - in real implementation, run actual VAD
            return {"speech_detected": True, "confidence": 0.95}

        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    except Exception as e:
        logger.error(f"Error executing inference: {e}")
        raise


async def _handle_ping(conn: Connection):
    """Handle ping request."""
    try:
        # Get memory usage
        import psutil

        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024

        conn.send(
            {
                "type": "ping_response",
                "memory_usage_mb": memory_mb,
                "timestamp": time.time(),
            }
        )
    except Exception as e:
        logger.error(f"Error in ping: {e}")
        conn.send({"type": "error", "error": str(e)})
