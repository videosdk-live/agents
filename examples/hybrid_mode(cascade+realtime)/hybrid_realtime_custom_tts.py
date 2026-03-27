import logging
from videosdk.agents import Agent, AgentSession, Pipeline, JobContext, RoomOptions, WorkerJob
from videosdk.plugins.xai import XAIRealtime, XAIRealtimeConfig, XAITurnDetection
from videosdk.plugins.cartesia import CartesiaTTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


class HybridVoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful voice assistant. Speak naturally and conversationally."
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello! I'm using a realtime model with a custom voice. How can I help you?")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")


async def entrypoint(ctx: JobContext):
    # Realtime model handles STT + LLM (low latency, web search capable)
    llm = XAIRealtime(
        model="grok-4-1-fast-non-reasoning",
        # When XAI_API_KEY is set in .env - DON'T pass api_key parameter
        # api_key="xi-proj-...",
        config=XAIRealtimeConfig(
            voice="Eve",  # Ara, Rex, Sal, Eve, Leo — model voice is overridden by TTS below
            enable_web_search=True,
            turn_detection=XAITurnDetection(
                type="server_vad",
                threshold=0.5,
                prefix_padding_ms=300,
                silence_duration_ms=200,
            ),
        )
    )

    # Hybrid Mode: Realtime LLM + custom TTS voice
    # The pipeline intercepts the realtime model's audio output and replaces it
    # with synthesized speech from CartesiaTTS, giving full control over the voice.
    pipeline = Pipeline(
        llm=llm,
        tts=CartesiaTTS(),  # Custom voice — overrides the realtime model's built-in voice
    )

    session = AgentSession(agent=HybridVoiceAgent(), pipeline=pipeline)
    await session.start(wait_for_participant=True, run_until_shutdown=True)


def make_context() -> JobContext:
    room_options = RoomOptions(
        room_id="<room_id>",  # Replace with your actual room_id
        name="Hybrid Mode Agent (Realtime + Custom TTS)",
        playground=True,
    )
    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()
