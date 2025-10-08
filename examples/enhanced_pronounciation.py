import asyncio
import os
import re
from typing import AsyncIterator, Optional
from videosdk.agents import Agent, AgentSession, CascadingPipeline, WorkerJob, ConversationFlow, JobContext, RoomOptions
from videosdk.plugins.openai import OpenAILLM
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
from videosdk.plugins.elevenlabs import ElevenLabsTTS

import logging

logging.getLogger().setLevel(logging.CRITICAL)

pre_download_model()

class VoiceAgent(Agent):
    def __init__(self, ctx: Optional[JobContext] = None):
        super().__init__(
            instructions="You are a helpful voice assistant that can answer questions."
        )
        
    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you today?")
    
    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")  
    
class CustomConversationFlow(ConversationFlow):
    def __init__(self, agent):
        super().__init__(agent)
        self.pronunciation_map = {
            "nginx": "engine x",
            "URL": "U R L",
            "API": "A P I",
            "VideoSDK": "Video SDK",
        }
        
    def pronounce_text(self, text: str) -> str:
        """Pronounce the text"""
        for word, pronunciation in self.pronunciation_map.items():
            text = re.sub(
                rf'\b{word}\b',
                pronunciation,
                text,
                flags=re.IGNORECASE
            )
        return text

    async def run(self, transcript: str) -> AsyncIterator[str]:
        async for response_chunk in self.process_with_llm():
            processed_chunk = self.pronounce_text(response_chunk)
            yield processed_chunk

async def entrypoint(ctx: JobContext):
    
    agent = VoiceAgent()
    conversation_flow = CustomConversationFlow(agent)

    pipeline = CascadingPipeline(
        stt= DeepgramSTT(api_key=os.getenv("DEEPGRAM_API_KEY")),
        llm=OpenAILLM(api_key=os.getenv("OPENAI_API_KEY")),
        tts=ElevenLabsTTS(api_key=os.getenv("ELEVENLABS_API_KEY")),
        vad=SileroVAD(),
        turn_detector=TurnDetector(threshold=0.8)
    )
    session = AgentSession(
        agent=agent, 
        pipeline=pipeline,
        conversation_flow=conversation_flow,
    )

    await ctx.run_until_shutdown(session=session,wait_for_participant=True)

def make_context() -> JobContext:
    room_options = RoomOptions(room_id="<meeting_id>", name="Sandbox Agent", playground=True)
    
    return JobContext(
        room_options=room_options
        )

if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()
