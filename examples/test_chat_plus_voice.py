# This example demonstrates Chat + Voice combined
# Agent accepts both voice input (STT) and text input (pubsub)
# Agent responds with voice (TTS) and text (pubsub)

import asyncio
import logging
from videosdk import PubSubPublishConfig, PubSubSubscribeConfig
from videosdk.agents import Agent, AgentSession, Pipeline,WorkerJob, JobContext, RoomOptions
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.google import GoogleLLM
from videosdk.plugins.cartesia import CartesiaTTS
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
pre_download_model()

class MultimodalAgent(Agent):
    def __init__(self, ctx: JobContext):
        super().__init__(
            instructions="You are a helpful assistant. You can communicate via voice. Respond naturally to all questions.",
        )
        self.ctx = ctx
        
    async def on_enter(self) -> None:
        await self.session.say("Hello! You can talk to me or send text messages.")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")
        
        await self.send_text_response("Agent ready! You can use voice or text to communicate.")
    
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
    agent = MultimodalAgent(ctx)

    pipeline = Pipeline(
        stt=DeepgramSTT(),
        llm=GoogleLLM(),
        tts=CartesiaTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector()
    )
    
    session = AgentSession(agent=agent, pipeline=pipeline)
    
    shutdown_event = asyncio.Event()
    
    async def on_pubsub_text_message(message):
        """Process text messages from users"""
        text = message.get('message', '') if isinstance(message, dict) else str(message)
        if isinstance(message, dict) and message.get("message") == "interrupt":
            print("Interrupting...")
            session.interrupt()      
            return
        sender = message.get('senderName', 'User') if isinstance(message, dict) else 'User'
        
        if text.strip():
            print(f"Text from {sender}: {text}")
            await pipeline.process_text(text)
    
    def on_pubsub_text_message_wrapper(message):
        asyncio.create_task(on_pubsub_text_message(message))
    
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
            cb=on_pubsub_text_message_wrapper
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
        name="Multimodal Agent (Voice+Chat)",
        playground=True
    )
    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()
