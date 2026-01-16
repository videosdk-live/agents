import asyncio
import aiohttp
import os
from typing import AsyncIterator

from videosdk.agents import Agent, AgentSession, CascadingPipeline, function_tool, JobContext, RoomOptions, WorkerJob, ConversationFlow, ChatRole
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
from videosdk.plugins.simli import SimliAvatar, SimliConfig
from videosdk.plugins.openai import OpenAILLM
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.elevenlabs import ElevenLabsTTS
from videosdk.plugins.openai import OpenAILLM,OpenAITTS
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
from videosdk.plugins.google import GoogleLLM
from videosdk.plugins.sarvamai import SarvamAITTS
import logging
logging.basicConfig(level=logging.INFO)


# Pre-downloading the Turn Detector model
pre_download_model()



class MyVoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are VideoSDK's AI Avatar Voice Agent with real-time capabilities. You are a helpful virtual assistant with a visual avatar that can answer questions about weather help with other tasks in real-time.",
        )
        # self.set_thinking_audio()


    async def on_enter(self) -> None:
        await self.play_background_audio(looping=True)
        await self.session.say("Hello! I'm your AI avatar assistant powered by VideoSDK. How can I help you today?")
    
    async def on_exit(self) -> None:
        await self.session.say("Goodbye! It was nice talking with you!")
        

async def start_session(context: JobContext):
    
    # Initialize VAD and Turn Detector
    vad = SileroVAD()
    turn_detector = TurnDetector(threshold=0.8)

    # # Initialize Simli Avatar
    # simli_config = SimliConfig(
    #     apiKey=os.getenv("SIMLI_API_KEY"),
    #     faceId="d2a5c7c6-fed9-4f55-bcb3-062f7cd20103",
    #     maxSessionLength=1800,
    #     maxIdleTime=600,
    # )

    # simli_avatar = SimliAvatar(
    #     config=simli_config,
    #     is_trinity_avatar=True,
    # )

    # Create agent and conversation flow
    agent = MyVoiceAgent()
    conversation_flow = ConversationFlow(agent)

    # Create pipeline with avatar
    pipeline = CascadingPipeline(
        stt=DeepgramSTT(),
        llm=GoogleLLM(),
        tts=SarvamAITTS(),
        vad=vad, 
        turn_detector=turn_detector,
        # avatar=simli_avatar
    )

    session = AgentSession(
        agent=agent,
        pipeline=pipeline,
        conversation_flow=conversation_flow
    )

    await session.start(wait_for_participant=True, run_until_shutdown=True)

def make_context() -> JobContext:
    room_options = RoomOptions(
        room_id="e6bm-1c5g-ty1q",
        name="Simli Avatar Cascading Agent",
        playground=True,
        background_audio=False,
    )

    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=start_session, jobctx=make_context)
    job.start() 