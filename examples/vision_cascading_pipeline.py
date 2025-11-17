import asyncio
from typing import Optional, Any, Dict
import logging

from videosdk import PubSubSubscribeConfig
from videosdk.agents import (Agent,AgentSession,CascadingPipeline,ConversationFlow,WorkerJob,JobContext,RoomOptions,function_tool)
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.elevenlabs import ElevenLabsTTS
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.google import GoogleLLM
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

pre_download_model()

class VisionAgent(Agent):

    def __init__(self, ctx: Optional[JobContext] = None):
        super().__init__(
            instructions=(
                "You are a helpful voice assistant that can answer questions "
                "and help with tasks. Call tool [capture_frame] whenever user says 'capture frame' or 'describe image' or some synonym related to it."
            ),
            # tools=[self.capture_frame],
        )
        self.ctx = ctx
        self.session: Optional[AgentSession] = None

    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you today?")
    
    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")

    @function_tool
    async def capture_frame(self, message: str) -> Dict[str, Any]:
        logging.info("capture_frame tool called with message: %s", message)

        if not self.session:
            logging.error("capture_frame called but no session attached to agent.")
            return {"success": False, "error": "no session attached"}
        try:
            frames = self.capture_frames(num_of_frames=4)
        except Exception as e:
            logging.exception("Error while capturing frames:")
            return {"success": False, "error": str(e)}

        if not frames:
            logging.warning("No frames captured; ensure vision is enabled in RoomOptions.")
            return {"success": False, "error": "no frames captured"}
        try:
            await self.session.reply(
                "Please analyze this frame and describe what you see in detail within one line.",
                frames=frames,
            )
        except Exception as e:
            logging.exception("Failed to send frames to session:")
            return {"success": False, "error": f"failed to reply with frames: {e}"}

        logging.info("capture_frame completed and reply sent.")
        return {"success": True, "res": "image captured and sent for analysis"}


async def entrypoint(ctx: JobContext):

    agent = VisionAgent(ctx)
    conversation_flow = ConversationFlow(agent)

    pipeline = CascadingPipeline(
        stt=DeepgramSTT(),
        llm=GoogleLLM(),
        tts=ElevenLabsTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector()
    )
    session = AgentSession(
        agent=agent,
        pipeline=pipeline,
        conversation_flow=conversation_flow,
    )
    agent.session = session

    shutdown_event = asyncio.Event()

    async def cleanup_session():
        logging.info("Cleaning up session...")
        try:
            await session.close()
        except Exception:
            logging.exception("Error while closing session")
        shutdown_event.set()

    ctx.add_shutdown_callback(cleanup_session)

    def on_session_end(reason: str):
        logging.info("Session ended: %s", reason)
        asyncio.create_task(ctx.shutdown())

    try:
        await ctx.connect()
        ctx.room.setup_session_end_callback(on_session_end)
        logging.info("Waiting for participant...")
        await ctx.room.wait_for_participant()
        logging.info("Participant joined")
        await session.start()
        await shutdown_event.wait()
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received, shutting down...")
    finally:
        try:
            await session.close()
        except Exception:
            logging.exception("Error during final session.close()")
        try:
            await ctx.shutdown()
        except Exception:
            logging.exception("Error during final ctx.shutdown()")


def make_context() -> JobContext:
    room_options = RoomOptions(
        room_id="n7cu-4iqc-nw0t", name="Vision Agent", playground=True, vision=True
    )
    return JobContext(room_options=room_options)

if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context())
    job.start()