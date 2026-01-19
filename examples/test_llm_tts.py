# This example demonstrates LLM + TTS only configuration
# User sends text via pubsub, agent responds with voice

import asyncio
import logging
from videosdk import PubSubSubscribeConfig
from videosdk.agents import Agent, AgentSession, Pipeline, WorkerJob, JobContext, RoomOptions
from videosdk.plugins.google import GoogleLLM
from videosdk.plugins.cartesia import CartesiaTTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

class TextToVoiceAgent(Agent):
    def __init__(self, ctx: JobContext):
        super().__init__(
            instructions="You are a helpful assistant. Respond naturally to user messages.",
        )
        self.ctx = ctx
        
    async def on_enter(self) -> None:
        await self.session.say("Hello! Send me a text message and I'll respond with voice.")
        print("Agent is ready! Send text messages via pubsub topic 'CHAT'")
    
    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")


async def entrypoint(ctx: JobContext):
    agent = TextToVoiceAgent(ctx)

    pipeline = Pipeline(
        llm=GoogleLLM(),
        tts=CartesiaTTS()
    )
    
    session = AgentSession(agent=agent, pipeline=pipeline)
    
    shutdown_event = asyncio.Event()
    
    async def on_pubsub_message(message):
        """Handle incoming text messages from pubsub"""
        text = message.get('message', '') if isinstance(message, dict) else str(message)
        print(f"Received text: {text}")
        
        if text.strip():
            await pipeline.process_text(text)
    
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
    room_options = RoomOptions(
        room_id="<room_id>",
        name="VideoSDK's Text-to-Voice Agent (LLM+TTS)",
        playground=True
    )
    return JobContext(room_options=room_options)


if __name__ == "__main__":

    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()
