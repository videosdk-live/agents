# This test script is used to test cascading pipeline.
import asyncio
import logging
from token import OP
import aiohttp
from videosdk.agents import Agent, AgentSession, CascadingPipeline, DTMFHandler, VoiceMailDetector, WorkerJob, MCPServerStdio, ConversationFlow, JobContext, RoomOptions, Options
from videosdk.plugins.openai import OpenAILLM
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
from videosdk.plugins.elevenlabs import ElevenLabsTTS

logging.getLogger().setLevel(logging.CRITICAL)
pre_download_model()

class VoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful voice assistant that can answer questions and help with tasks and help with horoscopes and weather."
        )
        
    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you today?")
    
    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")
        
async def entrypoint(ctx: JobContext):
    
    agent = VoiceAgent()
    conversation_flow = ConversationFlow(agent)

    pipeline = CascadingPipeline(
        stt= DeepgramSTT(),
        llm=OpenAILLM(),
        tts=ElevenLabsTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector()
    )
    
    async def custom_callback_voicemail():
        print("Voice Mail detected, Shutting down the agent")
        await agent.hangup()

    async def custom_callback_dtmf(message):
        print("Callback message received:", message)

    dtmf_handler = DTMFHandler(custom_callback_dtmf)
    voice_mail_detector = VoiceMailDetector(llm=OpenAILLM(), duration=7.0, callback = custom_callback_voicemail)

    session = AgentSession(
        agent=agent, 
        pipeline=pipeline,
        conversation_flow=conversation_flow,
        dtmf_handler = dtmf_handler,
        voice_mail_detector = voice_mail_detector
    )

    await session.start(wait_for_participant=True, run_until_shutdown=True)

def make_context() -> JobContext:
    room_options = RoomOptions(room_id="<room_id>", name="Sandbox Agent", playground=True)
    
    return JobContext(
        room_options=room_options 
        ) 
 
options = Options(agent_id="<agent-id>", max_processes=5, register=True, log_level="INFO", host="localhost", port=5000) 
 
if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context, options=options)
    job.start()

