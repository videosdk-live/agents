import asyncio
import os
from videosdk.agents import Agent, AgentSession, CascadingPipeline, WorkerJob, ConversationFlow, JobContext, RoomOptions
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
from videosdk.plugins.anthropic import AnthropicLLM
from videosdk.plugins.google import GoogleTTS
import logging

logging.getLogger().setLevel(logging.CRITICAL)
# pre_download_model()
class VoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful voice assistant that can answer questions and help with tasks and help with horoscopes and weather.",
        )
        
    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you today?")
    
    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")

async def entrypoint(ctx: JobContext):
    
    agent = VoiceAgent()
    conversation_flow = ConversationFlow(agent)

    pipeline = CascadingPipeline(
        stt= DeepgramSTT(api_key=os.getenv("DEEPGRAM_API_KEY")),
        llm=AnthropicLLM(api_key=os.getenv("ANTHROPIC_API_KEY")),
        tts=GoogleTTS(api_key=os.getenv("GOOGLE_API_KEY")),
        vad=SileroVAD(),
        turn_detector=TurnDetector(threshold=0.8)
    )
    session = AgentSession(
        agent=agent, 
        pipeline=pipeline,
        conversation_flow=conversation_flow,
        wake_up=45
    )
    
    async def on_wake_up():
        await session.say("Hello, are you there?")
    
    session.on_wake_up = on_wake_up

    await ctx.run_until_shutdown(session=session,wait_for_participant=True)
    
def make_context() -> JobContext:
    room_options = RoomOptions(room_id="<room_id>", name="Sandbox Agent", playground=True)
    
    return JobContext(
        room_options=room_options
        )

if __name__ == "__main__":

    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()
