"""
Use Case: Manufacturing line QA agent — passively monitors a video feed and flags defects on demand.
Pipeline: P1 — DeepgramSTT + GoogleLLM + ElevenLabsTTS + SileroVAD + TurnDetector + vision_frame hook
Demonstrates: pipeline.on("vision_frame") for frame sampling (every 50th frame), session.reply(frames=[frame]).
Env Vars: VIDEOSDK_AUTH_TOKEN, DEEPGRAM_API_KEY, GOOGLE_API_KEY, ELEVENLABS_API_KEY
"""

import os
import asyncio
import logging
from videosdk import PubSubSubscribeConfig
from videosdk.agents import Agent, AgentSession, Pipeline, WorkerJob, JobContext, RoomOptions
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.google import GoogleLLM
from videosdk.plugins.elevenlabs import ElevenLabsTTS
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
pre_download_model()

FRAMES_DIR = "qa_frames"
os.makedirs(FRAMES_DIR, exist_ok=True)


class QAInspectorAgent(Agent):
    def __init__(self, ctx: JobContext):
        super().__init__(
            instructions="""You are a manufacturing QA inspector for PrecisionParts Ltd.
            When given a frame from the production line, inspect it for defects:
            - Scratches, dents, or surface damage
            - Misalignment or incorrect assembly
            - Missing components or labels
            - Color anomalies or contamination
            Report findings with:
            - Location: left/center/right, top/bottom of the frame
            - Defect type: specific description
            - Severity: LOW / MEDIUM / HIGH
            - Recommended action: accept / rework / reject
            If no defects are found, state "QA PASS" clearly.""",
        )
        self.ctx = ctx

    async def on_enter(self) -> None:
        await self.session.say(
            "QA Inspector activated on the production line. "
            "Passive frame monitoring is running — I'll sample every 50th frame. "
            "Send 'analyze_frame' via control panel to trigger a manual inspection."
        )

    async def on_exit(self) -> None:
        await self.session.say("QA Inspector deactivated. Line monitoring has stopped.")


async def entrypoint(ctx: JobContext):
    agent = QAInspectorAgent(ctx)

    pipeline = Pipeline(
        stt=DeepgramSTT(),
        llm=GoogleLLM(),
        tts=ElevenLabsTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
    )

    frame_counter = 0
    latest_frame = None

    @pipeline.on("vision_frame")
    async def sample_frames_hook(frame_stream):
        """Passively sample every 50th frame from the production line camera."""
        nonlocal frame_counter, latest_frame
        async for frame in frame_stream:
            frame_counter += 1
            latest_frame = frame  # Keep the most recent frame in memory

            if frame_counter % 50 == 0:
                # Save the sampled frame to disk for audit trail.
                try:
                    img = frame.to_image()
                    path = os.path.join(FRAMES_DIR, f"frame_{frame_counter}.jpg")
                    img.save(path)
                    logging.info("[HOOK] Sampled frame %d → %s", frame_counter, path)
                except Exception as e:
                    logging.warning("Failed to save frame: %s", e)

            yield frame  # Always pass frames through — never drop them

    session = AgentSession(agent=agent, pipeline=pipeline)
    shutdown_event = asyncio.Event()

    async def on_control_message(message):
        """Trigger manual QA inspection on the latest sampled frame."""
        nonlocal latest_frame
        msg = message.get("message", "") if isinstance(message, dict) else str(message)
        logging.info("Control message: %s", msg)

        if msg.strip().lower() == "analyze_frame":
            if latest_frame is not None:
                logging.info("Triggering QA inspection on frame %d.", frame_counter)
                await session.reply(
                    "Inspect this production line frame for defects. "
                    "Report location, defect type, severity, and recommended action. "
                    "If no defects found, state QA PASS.",
                    frames=[latest_frame],
                )
            else:
                await agent.session.say("No frame available yet. The camera may not be streaming.")

    def on_control_message_wrapper(message):
        asyncio.create_task(on_control_message(message))

    async def cleanup_session():
        await session.close()
        shutdown_event.set()

    ctx.add_shutdown_callback(cleanup_session)

    def on_session_end(reason: str):
        asyncio.create_task(ctx.shutdown())

    try:
        await ctx.connect()
        ctx.room.setup_session_end_callback(on_session_end)
        logging.info("Waiting for production line camera...")
        await ctx.room.wait_for_participant()
        logging.info("Camera feed connected.")

        subscribe_config = PubSubSubscribeConfig(topic="CHAT", cb=on_control_message_wrapper)
        await ctx.room.subscribe_to_pubsub(subscribe_config)

        await session.start()
        await shutdown_event.wait()
    except KeyboardInterrupt:
        pass
    finally:
        await session.close()
        await ctx.shutdown()


def make_context() -> JobContext:
    room_options = RoomOptions(room_id="<room_id>", name="Production Line QA Agent", vision=True)
    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()
