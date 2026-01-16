"""
Minimal test for VideoSDK Inference STT, LLM & TTS with CascadingPipeline.

STT: VideoSDK Inference Gateway (Sarvam saarika:v2.5)
LLM: VideoSDK Inference Gateway (Google Gemini)
TTS: VideoSDK Inference Gateway (Sarvam bulbul:v2)

Required environment variables:
    VIDEOSDK_AUTH_TOKEN - VideoSDK authentication token

Run:
    python examples/test_inference_cascading.py
"""

import logging
from videosdk.agents import (
    Agent,
    AgentSession,
    CascadingPipeline,
    ConversationFlow,
    JobContext,
    RoomOptions,
    WorkerJob,
)
from videosdk.agents.inference import STT, TTS
from videosdk.plugins.sarvamai import SarvamAITTS
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.google import GoogleLLM
# Minimal logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class SimpleAgent(Agent):
    """Simple voice agent for testing inference STT."""

    def __init__(self):
        super().__init__(
            instructions="You are a helpful voice assistant. Keep responses brief and conversational.",
        )

    async def on_enter(self) -> None:
        await self.session.say(
            "Hello! I'm using VideoSDK Inference for speech recognition. How can I help you?"
        )

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the agent."""

    agent = SimpleAgent()
    conversation_flow = ConversationFlow(agent)

    # Create pipeline with Inference STT, LLM & TTS (all via VideoSDK Gateway)
    pipeline = CascadingPipeline(
        # Inference STT (via VideoSDK Gateway)
        stt=STT.sarvam(model_id="saarika:v2.5", language="en-IN"),
        # Use direct Google LLM plugin
        llm=GoogleLLM(model="gemini-3-flash-preview"),
        # Inference TTS (via VideoSDK Gateway)
        # tts=TTS.sarvam(model_id="bulbul:v2", speaker="anushka", language="en-IN"),
        # tts=TTS.google(
        #     model_id="Chirp3-HD",
        #     voice_id="Achernar",
        #     language="en-US",
        # ),
        # Or use direct Sarvam TTS:
        tts=SarvamAITTS(model="bulbul:v2", language="en-IN", speaker="anushka"),
        vad=SileroVAD(),
    )

    session = AgentSession(
        agent=agent,
        pipeline=pipeline,
        conversation_flow=conversation_flow,
    )

    await session.start(wait_for_participant=True, run_until_shutdown=True)


def make_context() -> JobContext:
    """Create job context for playground mode."""
    room_options = RoomOptions(
        # room_id="<ROOM_ID>",
        name="Inference STT Test Agent",
        playground=True
    )
    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()
