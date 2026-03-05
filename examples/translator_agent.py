import asyncio
import os
from typing import AsyncIterator, Optional
from sarvamai import SarvamAI
from videosdk.agents import Agent, AgentSession, Pipeline, WorkerJob, JobContext, RoomOptions
from videosdk.plugins.openai import OpenAILLM
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.sarvamai import SarvamAISTT, SarvamAITTS
from videosdk.plugins.cartesia import CartesiaTTS

import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

class TranslatorAgent(Agent):
    def __init__(self, ctx: Optional[JobContext] = None):
        super().__init__(
            instructions="You are a helpful translator assistant that can speak to user in their language.",
        )
        self.ctx = ctx
        
    async def on_enter(self) -> None:
        pass
    
    async def on_exit(self) -> None:
        pass
    

        
async def entrypoint(ctx: JobContext):
    
    agent = TranslatorAgent(ctx)

    pipeline = Pipeline(
        stt=SarvamAISTT(),
        llm=OpenAILLM(), 
        tts=CartesiaTTS(),
        vad=SileroVAD()
    )

    current_language_code = "en-IN"

    async def detect_language(transcript: str) -> str:
        """Detect the language of the transcript."""
        api_key = os.getenv("SARVAMAI_API_KEY")
        if not api_key:
            raise ValueError("SARVAMAI_API_KEY is not set")

        client = SarvamAI(
            api_subscription_key=api_key
        )
        response = client.text.identify_language(
            input=transcript
        )
        print(f"Detected language: {response.language_code}")
        return response.language_code

    @pipeline.on("user_turn_start")
    async def on_user_turn_start(transcript: str):
        nonlocal current_language_code
        detected_language = await detect_language(transcript)
        
        if detected_language != current_language_code:
            current_language_code = detected_language
            await pipeline.change_component(
                tts = SarvamAITTS(api_key=os.getenv("SARVAMAI_API_KEY"), language=current_language_code)
            )

    session = AgentSession(
        agent=agent, 
        pipeline=pipeline,
    )

    await session.start(wait_for_participant=True, run_until_shutdown=True)

def make_context() -> JobContext:
    room_options = RoomOptions(room_id="<room_id>", name="Translator Agent", playground=True)
    
    return JobContext(
        room_options=room_options
        )

if __name__ == "__main__":

    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()
