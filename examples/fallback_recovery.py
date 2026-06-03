import logging
from videosdk.agents import Agent, AgentSession,Pipeline,WorkerJob, JobContext, RoomOptions,FallbackSTT,FallbackLLM,FallbackTTS
from videosdk.agents.plugins import OpenAISTT, OpenAILLM, OpenAITTS, SileroVAD, TurnDetector, pre_download_model, DeepgramSTT, CartesiaTTS, CerebrasLLM

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
    # 4. latency_threshold_ms: Per-component latency budget in ms (STT stt_latency / LLM llm_ttft / TTS ttfb).
    #    Off by default — pass a value to enable latency-based fallback.
    # 5. consecutive_latency_hits: Switch only after this many consecutive turns above the threshold (default 3).
    #    Recovery/cooldown use the same temporary_disable_sec / permanent_disable_after_attempts as the error path.

    stt_provider = FallbackSTT(
        [OpenAISTT(), DeepgramSTT()],
        temporary_disable_sec=30.0,
        permanent_disable_after_attempts=3,
        latency_threshold_ms=350,
        consecutive_latency_hits=3,
    )

    llm_provider = FallbackLLM(
        [OpenAILLM(model="gpt-4o-mini"), CerebrasLLM()],
        temporary_disable_sec=30.0,
        permanent_disable_after_attempts=3,
        latency_threshold_ms=800,
        consecutive_latency_hits=3,
    )

    tts_provider = FallbackTTS(
        [OpenAITTS(voice="alloy"), CartesiaTTS()],
        temporary_disable_sec=30.0,
        permanent_disable_after_attempts=3,
        latency_threshold_ms=250,
        consecutive_latency_hits=3,
    )


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