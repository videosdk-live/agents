import asyncio
import os
from typing import Optional
from PIL import Image as PILImage
import av

from videosdk import PubSubSubscribeConfig
from videosdk.agents import Agent, AgentSession, Pipeline, WorkerJob, JobContext, RoomOptions
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.elevenlabs import ElevenLabsTTS
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.google import GoogleLLM
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])

pre_download_model()

FRAMES_DIR = "captured_frames"
os.makedirs(FRAMES_DIR, exist_ok=True)

latest_saved_frame_path: str | None = None
frame_counter = 0


class VisionAgent(Agent):

    def __init__(self, ctx: Optional[JobContext] = None):
        super().__init__(
            instructions="YOU CAN ONLY SPEAK IN ENGLISH. You are a helpful voice assistant with vision capabilities.",
        )
        self.ctx = ctx

    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you today?")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")


async def entrypoint(ctx: JobContext):
    global latest_saved_frame_path, frame_counter

    agent = VisionAgent(ctx)

    pipeline = Pipeline(
        stt=DeepgramSTT(),
        llm=GoogleLLM(),
        tts=ElevenLabsTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector()
    )

    @pipeline.on("vision_frame")
    async def save_frames_hook(frame_stream):
        global latest_saved_frame_path, frame_counter
        async for frame in frame_stream:
            frame_counter += 1

            if frame_counter % 50 == 0:
                img = frame.to_image()
                path = os.path.join(FRAMES_DIR, f"frame_{frame_counter}.jpg")
                img.save(path)
                latest_saved_frame_path = path
                print(f"[hook] Saved frame → {path}")

            yield frame

    session = AgentSession(
        agent=agent,
        pipeline=pipeline,
    )

    shutdown_event = asyncio.Event()

    async def on_pubsub_message(message):
        global latest_saved_frame_path
        print("Pubsub message received:", message)

        if not isinstance(message, dict):
            return

        msg = message.get("message", "")

        if msg == "send_saved_frame":
            if latest_saved_frame_path and os.path.exists(latest_saved_frame_path):
                print(f"Loading saved frame from {latest_saved_frame_path}")
                pil_img = PILImage.open(latest_saved_frame_path).convert("RGB")
                frame = av.VideoFrame.from_image(pil_img)
                await session.reply(
                    "Describe what you see in this image in one line.",
                    frames=[frame]
                )
            else:
                print("No saved frame available yet. Wait for the hook to capture one.")

    def on_pubsub_message_wrapper(message):
        asyncio.create_task(on_pubsub_message(message))

    async def cleanup_session():
        print("Cleaning up session...")
        await session.close()
        shutdown_event.set()

    ctx.add_shutdown_callback(cleanup_session)

    def on_session_end(reason: str):
        print(f"Session ended: {reason}")
        asyncio.create_task(ctx.shutdown())

    try:
        await ctx.connect()
        ctx.room.setup_session_end_callback(on_session_end)
        print("Waiting for participant...")
        await ctx.room.wait_for_participant()
        print("Participant joined")
        subscribe_config = PubSubSubscribeConfig(
            topic="CHAT",
            cb=on_pubsub_message_wrapper
        )
        await ctx.room.subscribe_to_pubsub(subscribe_config)
        await session.start()
        await shutdown_event.wait()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    finally:
        await session.close()
        await ctx.shutdown()


def make_context() -> JobContext:
    room_options = RoomOptions(room_id="<room_id>", name="Vision Hook Agent", vision=True)
    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()
