"""
Use Case: Hands-free warehouse inspection assistant — answers questions about what the camera sees.
Pipeline: P1 — DeepgramSTT + GoogleLLM + ElevenLabsTTS + SileroVAD + TurnDetector + RoomOptions(vision=True)
Demonstrates: agent.capture_frames(), session.reply(frames=frames), PubSub trigger for on-demand inspection.
Env Vars: VIDEOSDK_AUTH_TOKEN, DEEPGRAM_API_KEY, GOOGLE_API_KEY, ELEVENLABS_API_KEY

Alternative (Realtime mode):
    Uncomment the P2 block and comment out the P1 block to use GeminiRealtime with vision.
"""

import asyncio
import logging
from videosdk import PubSubSubscribeConfig
from videosdk.agents import Agent, AgentSession, Pipeline, WorkerJob, JobContext, RoomOptions
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.google import GoogleLLM
from videosdk.plugins.elevenlabs import ElevenLabsTTS
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
pre_download_model()


class WarehouseInspectorAgent(Agent):
    def __init__(self, ctx: JobContext):
        super().__init__(
            instructions="""You are a warehouse quality inspector assistant for LogiCo Warehousing.
            When shown an image from the warehouse camera:
            - Identify visible products, bins, or shelves
            - Flag any damage, missing labels, fallen items, or incorrect placement
            - Note the location within the frame (left/center/right, top/bottom)
            - Rate urgency: LOW (cosmetic), MEDIUM (restock needed), HIGH (safety risk)
            Keep each report under 3 sentences. Be factual and specific.
            Between inspections, answer questions about warehouse operations.""",
        )
        self.ctx = ctx

    async def on_enter(self) -> None:
        await self.session.say(
            "Warehouse inspector online. Camera feed is active. "
            "Send 'inspect' via the control panel or speak your question. "
            "I'm ready to analyze the floor."
        )

    async def on_exit(self) -> None:
        await self.session.say("Inspector signing off. Stay safe on the floor!")


async def entrypoint(ctx: JobContext):
    agent = WarehouseInspectorAgent(ctx)

    # --- P1: Standard voice pipeline with vision (primary) ---
    pipeline = Pipeline(
        stt=DeepgramSTT(),
        llm=GoogleLLM(),
        tts=ElevenLabsTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
    )

    # --- P2: Realtime pipeline with vision (alternative — lower latency) ---
    # from videosdk.plugins.google import GeminiRealtime, GeminiLiveConfig
    # model = GeminiRealtime(
    #     model="gemini-2.5-flash-native-audio-preview-12-2025",
    #     config=GeminiLiveConfig(voice="Puck", response_modalities=["AUDIO"]),
    # )
    # pipeline = Pipeline(llm=model)

    session = AgentSession(agent=agent, pipeline=pipeline)
    shutdown_event = asyncio.Event()

    async def on_control_message(message):
        """Handle control commands sent via PubSub."""
        msg = message.get("message", "") if isinstance(message, dict) else str(message)
        logging.info("Control message: %s", msg)

        if msg.strip().lower() == "inspect":
            try:
                frames = agent.capture_frames(num_of_frames=2)
                if frames:
                    logging.info("Captured %d frame(s) for inspection.", len(frames))
                    await session.reply(
                        "Analyze this warehouse frame. Identify any issues with products, labels, "
                        "placement, or safety. Report location and urgency level.",
                        frames=frames,
                    )
                else:
                    logging.warning("No frames available — is vision=True set in RoomOptions?")
                    await agent.session.say("Camera feed unavailable. Please check the camera connection.")
            except ValueError as e:
                logging.error("Frame capture error: %s", e)

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
        logging.info("Waiting for participant...")
        await ctx.room.wait_for_participant()
        logging.info("Participant joined.")

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
    room_options = RoomOptions(room_id="<room_id>", name="Warehouse Inspector Agent", vision=True)
    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()
