import asyncio
import os
from typing import AsyncIterator, Optional
from sarvamai import SarvamAI
from videosdk.agents import Agent, AgentSession, CascadingPipeline, WorkerJob, ConversationFlow, JobContext, RoomOptions
from videosdk.plugins.openai import OpenAILLM
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.sarvamai import SarvamAISTT, SarvamAITTS

import logging

logging.getLogger().setLevel(logging.CRITICAL)

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
    
class TranslatorConversationFlow(ConversationFlow):
    def __init__(self, agent):
        super().__init__(agent)
        self.language_code = "en-IN"

    async def run(self, transcript: str) -> AsyncIterator[str]:
        async for response_chunk in self.process_with_llm():
            yield response_chunk

    async def on_turn_start(self, transcript: str) -> None:
        """Called at the start of a user turn."""
        self.is_turn_active = True
        detected_language = await self.detect_language(transcript)
        
        if detected_language != self.language_code:
            self.language_code = detected_language
            await self.agent.session.pipeline.change_component(
                tts = SarvamAITTS(api_key=os.getenv("SARVAMAI_API_KEY"), target_language_code=self.language_code)
            )

    async def on_turn_end(self) -> None:
        """Called at the end of a user turn."""
        self.is_turn_active = False
        
    async def detect_language(self, transcript: str) -> str:
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
        
async def entrypoint(ctx: JobContext):
    
    agent = TranslatorAgent(ctx)
    conversation_flow = TranslatorConversationFlow(agent)

    pipeline = CascadingPipeline(
        stt = DeepgramSTT(api_key=os.getenv("DEEPGRAM_API_KEY"), language="multi"),
        llm=OpenAILLM(api_key=os.getenv("OPENAI_API_KEY")), 
        tts=SarvamAITTS(api_key=os.getenv("SARVAMAI_API_KEY")),
        vad=SileroVAD()
    )
    session = AgentSession(
        agent=agent, 
        pipeline=pipeline,
        conversation_flow=conversation_flow,
    )

    await ctx.run_until_shutdown(session=session,wait_for_participant=True)

def make_context() -> JobContext:
    room_options = RoomOptions(room_id="<room_id>", name="Translator Agent", playground=True)
    
    return JobContext(
        room_options=room_options
        )

if __name__ == "__main__":

    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()
