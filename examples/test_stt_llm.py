# This example demonstrates STT + LLM only configuration
# User speaks (voice input), agent responds with text via pubsub (no voice output)

import asyncio
import logging
from videosdk import PubSubPublishConfig
from videosdk.agents import Agent, AgentSession, Pipeline, WorkerJob, JobContext, RoomOptions
from videosdk.plugins.google import GoogleLLM
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
pre_download_model()

class VoiceToTextAgent(Agent):
    def __init__(self, ctx: JobContext):
        super().__init__(
            instructions="You are a helpful assistant. Respond concisely to user questions.",
        )
        self.ctx = ctx
        
    async def on_enter(self) -> None:
        await self.send_text_response("Agent ready! Speak to me and I'll respond with text via pubsub.")

    async def on_exit(self) -> None:
        await self.send_text_response("Goodbye!")

    async def send_text_response(self, message: str):
        """Send text response via pubsub"""
        try:
            publish_config = PubSubPublishConfig(
                topic="CHAT",
                message=message
            )
            await self.ctx.room.publish_to_pubsub(publish_config)
        except Exception as e:
            logging.error(f"Failed to publish text response: {e}")

async def entrypoint(ctx: JobContext):
    agent = VoiceToTextAgent(ctx)

    pipeline = Pipeline(
        stt=DeepgramSTT(),
        llm=GoogleLLM(),
        vad=SileroVAD(),
        turn_detector=TurnDetector()
    )
    
    session = AgentSession(agent=agent, pipeline=pipeline)
    
    shutdown_event = asyncio.Event()
    
    @pipeline.on("content_generated")
    async def on_content_generated(data: dict):
        """Send agent's text response via pubsub"""
        text = data.get("text", "")
        if text.strip():
            print(f"Sending agent response via pubsub: {text}")
            
            publish_config = PubSubPublishConfig(
                topic="CHAT",
                message=text
            )
            await ctx.room.publish_to_pubsub(publish_config)
            print("Response sent to pubsub topic 'AGENT_RESPONSE'")

    
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
        
        
        await session.start()
        await shutdown_event.wait()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    finally:
        await session.close()
        await ctx.shutdown()


def make_context() -> JobContext:
    room_options = RoomOptions(
        room_id="<room_id>",
        name="VideoSDK's Voice-to-Text Agent (STT+LLM)",
        playground=True
    )
    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()
