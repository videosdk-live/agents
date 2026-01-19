import logging
from videosdk.agents import Agent, AgentSession,Pipeline,WorkerJob, JobContext, RoomOptions,FallbackSTT,FallbackLLM,FallbackTTS
from videosdk.plugins.openai import OpenAISTT,OpenAILLM,OpenAITTS
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.cartesia import CartesiaTTS
from videosdk.plugins.cerebras import CerebrasLLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
pre_download_model()

class ResilientAgent(Agent):
    def __init__(self):
        super().__init__(instructions="You are a helpful voice assistant that can answer questions and help with tasks.")
        
    async def on_enter(self) -> None:
        await self.session.say("Hello Buddy, Welcome to Videosdk's Voice AI Agent Framework.")
    
    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")

async def entrypoint(ctx: JobContext):
    
    agent = ResilientAgent()

    # Fallback configuration:
    # 1. Define a list of providers (in priority order).
    # 2. temporary_disable_sec: Time to wait before retrying a failed primary provider.
    # 3. permanent_disable_after_attempts: Disable a provider permanently after N failed recovery attempts.

    stt_provider = FallbackSTT([OpenAISTT(),DeepgramSTT()],temporary_disable_sec=30.0, permanent_disable_after_attempts=3)
    llm_provider = FallbackLLM([OpenAILLM(model="gpt-4o-mini"),CerebrasLLM()],temporary_disable_sec=30.0, permanent_disable_after_attempts=3)
    tts_provider = FallbackTTS([OpenAITTS(voice="alloy"),CartesiaTTS()],temporary_disable_sec=30.0, permanent_disable_after_attempts=3)


    pipeline = Pipeline(
        stt= stt_provider,
        llm=llm_provider,
        tts=tts_provider,
        vad=SileroVAD(),
        turn_detector=TurnDetector()
    )
    session = AgentSession(
        agent=agent, 
        pipeline=pipeline,
    )

    await session.start(wait_for_participant=True, run_until_shutdown=True)

def make_context() -> JobContext:
    room_options = RoomOptions(room_id="<room_id>", name="Resilient Agent", playground=True)
    return JobContext(room_options=room_options)

if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context).start()