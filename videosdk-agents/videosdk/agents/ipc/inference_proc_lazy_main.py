import asyncio
import multiprocessing
import os
import signal
import sys
import time
import psutil
from typing import Any, Dict, Optional
import logging
from multiprocessing.connection import Connection

logger = logging.getLogger(__name__)


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0.0


def run_inference_process_main(conn: Connection):
    """
    Main function that runs inside the inference process.

    This function handles communication with the parent process
    and executes AI model inference as needed.
    """
    try:
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        logger.info(f"Inference process started (PID: {os.getpid()})")

        # Set up signal handlers
        def signal_handler(signum, frame):
            logger.info("Received shutdown signal")
            conn.send({"type": "shutdown_ack"})
            sys.exit(0)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        # Send ready signal
        conn.send({"type": "ready"})

        async def main_loop():
            while True:
                try:
                    if conn.poll(timeout=1.0):
                        message = conn.recv()
                        message_type = message.get("type")
                        if message_type == "inference":
                            await _handle_inference(conn, message.get("data", {}))
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
    conn: Connection,
    inference_data: Dict[str, Any],
    models: Dict[str, Any],
):
    """Handle an inference request."""
    try:
        logger.info(f"Executing inference: {inference_data.get('task_id', 'unknown')}")

        # Extract inference information
        task_id = inference_data.get("task_id")
        task_type = inference_data.get("task_type")  # stt, tts, llm, etc.
        model_config = inference_data.get("model_config", {})
        input_data = inference_data.get("input_data", {})

        # Get or create model
        model_key = f"{task_type}_{model_config.get('model_name', 'default')}"

        if model_key not in models:
            logger.info(f"Loading model: {model_key}")
            models[model_key] = await _load_model(task_type, model_config)

        model = models[model_key]

        # Execute inference
        result = await _execute_inference(task_type, model, input_data)

        # Send result back
        conn.send(
            {
                "type": "result",
                "data": {"task_id": task_id, "result": result, "status": "completed"},
            }
        )

    except Exception as e:
        logger.error(f"Error executing inference: {e}")
        conn.send(
            {"type": "error", "error": str(e), "task_id": inference_data.get("task_id")}
        )


async def _load_model(task_type: str, model_config: Dict[str, Any]) -> Any:
    """Load an AI model based on task type and configuration."""
    try:
        if task_type == "stt":
            # Load STT model
            from videosdk.agents.stt import STTModel

            return STTModel.from_config(model_config)

        elif task_type == "tts":
            # Load TTS model
            from videosdk.agents.tts import TTSModel

            return TTSModel.from_config(model_config)

        elif task_type == "llm":
            # Load LLM model
            from videosdk.agents.llm import LLMModel

            return LLMModel.from_config(model_config)

        else:
            raise ValueError(f"Unknown task type: {task_type}")

    except Exception as e:
        logger.error(f"Error loading model for {task_type}: {e}")
        raise


async def _execute_inference(
    task_type: str, model: Any, input_data: Dict[str, Any]
) -> Any:
    """Execute inference with the given model and input data."""
    try:
        if task_type == "stt":
            # Speech-to-text inference
            audio_data = input_data.get("audio_data")
            return await model.transcribe(audio_data)

        elif task_type == "tts":
            # Text-to-speech inference
            text = input_data.get("text")
            voice_config = input_data.get("voice_config", {})
            return await model.synthesize(text, voice_config)

        elif task_type == "llm":
            # Language model inference
            prompt = input_data.get("prompt")
            context = input_data.get("context", {})
            return await model.generate(prompt, context)

        else:
            raise ValueError(f"Unknown task type: {task_type}")

    except Exception as e:
        logger.error(f"Error executing inference for {task_type}: {e}")
        raise


async def _handle_ping(conn: Connection):
    """Handle a ping request."""
    try:
        memory_usage = get_memory_usage_mb()

        conn.send(
            {"type": "pong", "memory_usage": memory_usage, "timestamp": time.time()}
        )

    except Exception as e:
        logger.error(f"Error handling ping: {e}")
        conn.send({"type": "error", "error": str(e)})
