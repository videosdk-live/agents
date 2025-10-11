import asyncio
from typing import Optional
from videosdk import PubSubSubscribeConfig
from videosdk.agents import Agent, AgentSession, CascadingPipeline,WorkerJob,ConversationFlow,JobContext, RoomOptions
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.elevenlabs import ElevenLabsTTS
from videosdk.plugins.anthropic import AnthropicLLM
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
pre_download_model()

class PubSubAgent(Agent):
    def __init__(self, ctx: Optional[JobContext] = None):
        super().__init__(
            instructions="You are a helpful voice assistant that can answer questions and help with tasks.",
        )
        self.ctx = ctx
        
    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you today?")
    
    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")
    

async def entrypoint(ctx: JobContext):
    
    agent = PubSubAgent(ctx)
    conversation_flow = ConversationFlow(agent)

    pipeline = CascadingPipeline(
        stt= DeepgramSTT(),
        llm=AnthropicLLM(),
        tts=ElevenLabsTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector()
    )
    session = AgentSession(
        agent=agent, 
        pipeline=pipeline,
        conversation_flow=conversation_flow,
    )
    
    shutdown_event = asyncio.Event()

    async def on_pubsub_message(message):
        print("Pubsub message received:", message)
        if isinstance(message, dict) and message.get("message") == "reply":
            print("Replying...")
            await session.reply("Create a random number between 1 and 100. tell joke to user using that number.")
        if isinstance(message, dict) and message.get("message") == "interrupt":
            print("Interrupting...")
            session.interrupt()

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
    room_options = RoomOptions(room_id="<room_id>", name="Videosdk's Agent", playground=True)
    
    return JobContext(
        room_options=room_options
        )

if __name__ == "__main__":

    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()