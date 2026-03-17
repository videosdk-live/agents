"""
Use Case: Regional language support agent for Hindi/Indian English speakers (RuralPay digital payments).
Pipeline: P1 — VideoSDK Inference Gateway STT + GoogleLLM + VideoSDK Inference Gateway TTS + SileroVAD + TurnDetector
Demonstrates: VideoSDK Inference Gateway as a unified STT/TTS provider — single auth token, no third-party keys.
Env Vars: VIDEOSDK_AUTH_TOKEN (all STT/TTS routed through VideoSDK inference; only GOOGLE_API_KEY for LLM)
"""

import logging
from videosdk.agents import Agent, AgentSession, Pipeline, WorkerJob, JobContext, RoomOptions
from videosdk.agents.inference import STT, TTS
from videosdk.plugins.google import GoogleLLM
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
pre_download_model()


class RuralPayAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are Kavya, a customer support agent for RuralPay, a digital payments platform
            designed for small-town India. Help users with:
            - Checking account balance
            - Sending money to another user or phone number
            - Troubleshooting failed transactions (refund ETA: 3-5 business days)
            - Understanding UPI payment limits (daily limit: ₹1,00,000)
            Respond in simple Hindi or Indian English, depending on the user's language.
            Use simple words — many users may not be tech-savvy.
            Always confirm transaction details before processing.
            Never share OTPs or request them from users.""",
        )

    async def on_enter(self) -> None:
        await self.session.say(
            "Namaste! I'm Kavya from RuralPay support. "
            "Aap Hindi ya English mein baat kar sakte hain. "
            "How can I help you today?"
        )

    async def on_exit(self) -> None:
        await self.session.say(
            "Thank you for calling RuralPay. Dhanyavaad! Have a great day!"
        )


async def entrypoint(ctx: JobContext):
    agent = RuralPayAgent()

    # STT and TTS routed through the VideoSDK Inference Gateway — no third-party STT/TTS API keys needed.
    pipeline = Pipeline(
        stt=STT.sarvam(model_id="saarika:v2.5", language="en-IN"),
        llm=GoogleLLM(model="gemini-3.1-flash-lite-preview"),
        tts=TTS.sarvam(model_id="bulbul:v2", speaker="anushka", language="en-IN"),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
    )
    session = AgentSession(agent=agent, pipeline=pipeline)
    await session.start(wait_for_participant=True, run_until_shutdown=True)


def make_context() -> JobContext:
    room_options = RoomOptions(name="RuralPay Support - Kavya", playground=True)
    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()
