import logging
from videosdk.agents import (
    Agent,
    AgentSession,
    Pipeline,
    JobContext,
    RoomOptions,
    WorkerJob,
)
from videosdk.agents.inference import STT, TTS, LLM, Turn
from videosdk.plugins.silero import SileroVAD
from dotenv import load_dotenv

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


class VideoSDKCascadeInferenceAgent(Agent):
    """VideoSDK Inference Agent for voice interaction."""

    def __init__(self):
        super().__init__(
            instructions="""
                You are VideoSDK Inference Agent, a professional voice assistant.

                Guidelines:
                - Keep responses concise and conversational.
                - Speak clearly and naturally.
                - Be polite, helpful, and precise.
                - Avoid long or complex explanations unless necessary.
                - Ask for clarification if the user input is unclear.
                """
        )

    async def on_enter(self) -> None:
        await self.session.say(
            "Hello. This is VideoSDK Inference Agent. "
            "I am ready to process your voice input in real time. "
            "How can I assist you today?"
        )

    async def on_exit(self) -> None:
        await self.session.say(
            "Session ended. Thank you for using VideoSDK Inference Agent."
        )


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the agent."""

    agent = VideoSDKCascadeInferenceAgent()

    pipeline = Pipeline(
        stt=STT.sarvam(
            language="en-IN",
        ),
        llm=LLM.google(
            model_id="gemini-3-flash-preview",
        ),
        tts=TTS.sarvam(model_id="bulbul:v3", speaker="shubh", language="en-IN"),
        vad=SileroVAD(),
        turn_detector=Turn.namo(language="en"),
    )

    session = AgentSession(
        agent=agent,
        pipeline=pipeline,
    )

    await session.start(wait_for_participant=True, run_until_shutdown=True)


def make_context() -> JobContext:
    """Create job context for playground mode."""

    room_options = RoomOptions(
        # room_id="<room_id>",
        name="VideoSDK's Cascade Inference Agent",
        playground=True,
    )

    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context())
    job.start()
