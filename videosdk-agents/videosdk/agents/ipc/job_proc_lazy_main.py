"""
Job Process Lazy Main for VideoSDK Agents IPC.

This module runs inside the job process and handles job execution,
similar to implementation but adapted for VideoSDK.
"""

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


def run_job_process_main(conn: Connection):
    """
    Main function that runs inside the job process.

    This function handles communication with the parent process
    and executes jobs as they come in.
    """
    try:
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        logger.info(f"Job process started (PID: {os.getpid()})")

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
                    # Wait for message from parent
                    if conn.poll(timeout=1.0):
                        message = conn.recv()
                        message_type = message.get("type")

                        if message_type == "job":
                            await _handle_job(conn, message.get("data", {}))
                        elif message_type == "ping":
                            await _handle_ping(conn)
                        elif message_type == "shutdown":
                            logger.info("Received shutdown request")
                            conn.send({"type": "shutdown_ack"})
                            break
                        else:
                            logger.warning(f"Unknown message type: {message_type}")

                except Exception as e:
                    logger.error(f"Error in job process main loop: {e}")
                    conn.send({"type": "error", "error": str(e)})

        asyncio.run(main_loop())

    except Exception as e:
        logger.error(f"Fatal error in job process: {e}")
        conn.send({"type": "error", "error": str(e)})
        sys.exit(1)
    finally:
        logger.info("Job process shutting down")
        conn.close()


async def _handle_job(conn: Connection, job_data: Dict[str, Any]):
    """Handle a job execution request."""
    try:
        job_type = job_data.get("type", "regular")
        logger.info(
            f"Executing job: {job_data.get('job_id', 'unknown')} (type: {job_type})"
        )

        if job_type == "launch_job":
            await _handle_launch_job(conn, job_data)
        else:
            await _handle_regular_job(conn, job_data)

    except Exception as e:
        logger.error(f"Error executing job: {e}")
        conn.send({"type": "error", "error": str(e), "job_id": job_data.get("job_id")})


async def _handle_launch_job(conn: Connection, job_data: Dict[str, Any]):
    """Handle a launch job with running info."""
    try:
        running_info = job_data.get("running_info")
        if not running_info:
            raise ValueError("No running_info provided for launch_job")

        # Import here to avoid circular imports
        from videosdk.agents.job import (
            JobContext,
            _set_current_job_context,
            _reset_current_job_context,
        )

        # Create job context from running info
        room_options = running_info.job.get("room")
        if not room_options:
            raise ValueError("No room options in running info")

        job_context = JobContext(room_options=room_options)

        # Set current job context
        token = _set_current_job_context(job_context)

        try:
            # Create pipeline components inside the child process to avoid pickling issues
            from videosdk.agents.cascading_pipeline import CascadingPipeline
            import os

            # Create pipeline components inside the child process to avoid pickling issues
            from videosdk.plugins.deepgram import DeepgramSTT
            from videosdk.plugins.openai import OpenAILLM
            from videosdk.plugins.elevenlabs import ElevenLabsTTS
            from videosdk.plugins.silero import SileroVAD

            # Create pipeline with components created in this process
            pipeline = CascadingPipeline(
                # Speech-to-Text
                stt=DeepgramSTT(api_key=os.getenv("DEEPGRAM_API_KEY")),
                # Language Model
                llm=OpenAILLM(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o"),
                # Text-to-Speech
                tts=ElevenLabsTTS(api_key=os.getenv("ELEVENLABS_API_KEY")),
                # Voice Activity Detection
                vad=SileroVAD(),
            )

            # Create a simple agent for the session
            from videosdk.agents import Agent

            class SimpleAgent(Agent):
                def __init__(self):
                    super().__init__(
                        instructions="You are a helpful voice assistant that can answer questions and help with tasks.",
                        tools=[],
                    )

                async def on_enter(self) -> None:
                    await self.session.say(
                        "Hello! I'm your voice assistant. How can I help you today?"
                    )

                async def on_exit(self) -> None:
                    await self.session.say("Goodbye! Have a great day!")

            # Create agent and session
            agent = SimpleAgent()
            from videosdk.agents import AgentSession

            session = AgentSession(agent=agent, pipeline=pipeline)

            # Set up logging callbacks for transcription and events
            from videosdk.agents import global_event_emitter

            def log_input_transcription(data):
                logger.info(f"🎤 INPUT TRANSCRIPT: '{data.get('text', '')}'")

            def log_output_transcription(data):
                logger.info(f"🎤 OUTPUT TRANSCRIPT: '{data.get('text', '')}'")

            def log_speech_started():
                logger.info("🔊 SPEECH STARTED (VAD detected)")

            def log_speech_stopped():
                logger.info("🔊 SPEECH STOPPED (VAD detected)")

            def log_text_response(data):
                logger.info(f"🤖 LLM RESPONSE: '{data.get('text', '')}'")

            # Register callbacks with the global event emitter
            global_event_emitter.on("input_transcription", log_input_transcription)
            global_event_emitter.on("output_transcription", log_output_transcription)
            global_event_emitter.on("speech_started", log_speech_started)
            global_event_emitter.on("speech_stopped", log_speech_stopped)
            global_event_emitter.on("text_response", log_text_response)

            # Connect to room
            await job_context.connect()

            # Start the session
            await session.start()

            # Keep the agent running until shutdown
            logger.info("Agent session started, keeping alive...")
            try:
                # Keep alive until the job is terminated
                while True:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                logger.info("Agent session cancelled, shutting down...")
            except Exception as e:
                logger.error(f"Error in agent session: {e}")

        finally:
            # Cleanup
            try:
                await job_context.shutdown()
            except Exception as e:
                logger.error(f"Error during job shutdown: {e}")

            # Reset job context
            _reset_current_job_context(token)

            # Send success result
            conn.send(
                {
                    "type": "result",
                    "data": {"status": "completed"},
                    "job_id": job_data.get("job_id"),
                }
            )

    except Exception as e:
        logger.error(f"Error in launch job: {e}")
        conn.send({"type": "error", "error": str(e), "job_id": job_data.get("job_id")})


async def _handle_regular_job(conn: Connection, job_data: Dict[str, Any]):
    """Handle a regular job execution request."""
    try:
        # Extract job information
        job_id = job_data.get("job_id")
        room_id = job_data.get("room_id")
        agent_name = job_data.get("agent_name")
        pipeline_config = job_data.get("pipeline", {})

        # Create job context
        from videosdk.agents.job import JobContext, RoomOptions

        job_ctx = JobContext(
            room_options=RoomOptions(
                room_id=room_id,
                auth_token=job_data.get("auth_token"),
                name=agent_name,
                playground=job_data.get("playground", True),
                vision=job_data.get("vision", False),
            )
        )

        # Create and configure pipeline
        from videosdk.agents.cascading_pipeline import CascadingPipeline

        # For now, create a simple pipeline without components
        pipeline = CascadingPipeline()
        job_ctx._set_pipeline_internal(pipeline)

        # Connect to room
        await job_ctx.connect()

        # Run the pipeline
        result = await pipeline.run(job_ctx)

        # Send result back
        conn.send(
            {
                "type": "result",
                "data": {"job_id": job_id, "result": result, "status": "completed"},
            }
        )

        # Cleanup
        await job_ctx.shutdown()

    except Exception as e:
        logger.error(f"Error executing regular job: {e}")
        conn.send({"type": "error", "error": str(e), "job_id": job_data.get("job_id")})


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
