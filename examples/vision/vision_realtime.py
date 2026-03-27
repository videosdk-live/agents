import asyncio
from videosdk import PubSubSubscribeConfig
from typing import Optional
from videosdk.agents import Agent, AgentSession, WorkerJob,JobContext, RoomOptions, Pipeline
from videosdk.plugins.openai import OpenAIRealtime, OpenAIRealtimeConfig
from openai.types.beta.realtime.session import InputAudioTranscription, TurnDetection


import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])


class VisionAgent(Agent):
    def __init__(self, ctx: Optional[JobContext] = None):
        super().__init__(
            instructions="You are a helpful voice assistant with vision capabilities. that can answer questions and help with tasks.",
        )
        self.ctx = ctx
        
    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you today?")
    
    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")
    

async def entrypoint(ctx: JobContext):
    
    agent = VisionAgent(ctx)

    model = OpenAIRealtime(
        model="gpt-realtime-2025-08-28",
        config=OpenAIRealtimeConfig(
            voice="fable",
            modalities=["text", "audio"],
            input_audio_transcription=InputAudioTranscription(model="whisper-1"),
            turn_detection=TurnDetection(
                type="server_vad",
                threshold=0.5,
                prefix_padding_ms=300,
                silence_duration_ms=200,
            ),
            tool_choice="auto",
        )
    )
    pipeline = Pipeline(llm=model)


    session = AgentSession(
        agent=agent, 
        pipeline=pipeline,
    )
    
    shutdown_event = asyncio.Event()

    async def on_pubsub_message(message):
        print("Pubsub message received:", message)
        if isinstance(message, dict) and message.get("message") == "capture_frames":
            print("Capturing frame....")
            try:
                frames = agent.capture_frames(num_of_frames=2)
                if frames:
                    print(f"Captured {len(frames)} frame(s)")
                    await session.reply(
                        "Please analyze this frame and describe what you see in details.within one line.",
                        frames=frames
                    )
                else:
                    print("No frames available. Make sure vision is enabled in RoomOptions.")
            except ValueError as e:
                print(f"Error: {e}")


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
    room_options = RoomOptions(room_id="<room_id>", name="Vision Agent",vision=True)
    
    return JobContext(room_options=room_options)

if __name__ == "__main__":

    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()