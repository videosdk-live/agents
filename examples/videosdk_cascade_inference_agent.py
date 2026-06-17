import logging
from videosdk.agents import (
    Agent,
    AgentSession,
    Pipeline,
    JobContext,
    RoomOptions,
    WorkerJob,
)
from videosdk.agents.inference import SarvamAISTT, GoogleLLM, SarvamAITTS, SileroVAD, TurnV2
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
        stt=SarvamAISTT(
            language="en-IN",
        ),
        llm=GoogleLLM(
            model_id="gemini-3-flash-preview",
        ),
        tts=SarvamAITTS(model_id="bulbul:v3", speaker="shubh", language="en-IN"),
        vad=SileroVAD(),
        # TurnV2 (VideoSDK Inference Gateway) — 4-state turn detection:
        # Complete / Incomplete / Backchannel / Wait. Backchannels are ignored
        # while the agent speaks; an explicit "wait/stop" interrupts. Needs
        # VIDEOSDK_AUTH_TOKEN.
        turn_detector=TurnV2.echo_large(),   # or TurnV2.echo_small() for lower latency
    )

    # Observe the turn detector's classification of each user utterance.
    # This hook is TurnV2-only — it fires for backchannel-aware detectors that
    # emit the full 4-state classification (Complete / Incomplete / Backchannel /
    # Wait).
    @pipeline.on("turn_state")
    async def on_turn_state(data: dict):
        # data = {"text": str, "state": "Complete" | "Incomplete" |
        #         "Backchannel" | "Wait" }
        print(f"[TURN] state={data['state']} text={data['text']!r}")

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
