"""
Use Case: Multilingual support agent for a global SaaS product — auto-detects user language and responds in kind.
Pipeline: P3 — SarvamAISTT + GeminiRealtime + SileroVAD (Hybrid STT mode)
Demonstrates: pipeline.change_component(tts=...) on language switch, user_turn_start hook for detection.
Env Vars: VIDEOSDK_AUTH_TOKEN, SARVAMAI_API_KEY, GOOGLE_API_KEY
"""

import os
import logging
from sarvamai import SarvamAI
from videosdk.agents import Agent, AgentSession, Pipeline, WorkerJob, JobContext, RoomOptions
from videosdk.plugins.google import GeminiRealtime, GeminiLiveConfig
from videosdk.plugins.sarvamai import SarvamAISTT, SarvamAITTS
from videosdk.plugins.silero import SileroVAD

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MultilingualSupportAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are a multilingual support agent for GlobalApp, a SaaS productivity platform.
            Detect the language the user is speaking from their transcription and always respond in that same language.
            Help users with account issues, feature questions, billing, and integrations.
            If you are unsure of the language, default to English.
            Be concise — users expect quick answers.""",
        )

    async def on_enter(self) -> None:
        await self.session.say(
            "Hi, welcome to GlobalApp support. I can help you in your preferred language. "
            "How can I assist you today?"
        )

    async def on_exit(self) -> None:
        await self.session.say("Thank you for contacting GlobalApp. Goodbye!")


async def entrypoint(ctx: JobContext):
    llm = GeminiRealtime(
        model="gemini-2.5-flash-native-audio-preview-12-2025",
        config=GeminiLiveConfig(
            voice="Puck",
            response_modalities=["AUDIO"],
        ),
    )

    pipeline = Pipeline(
        stt=SarvamAISTT(),
        llm=llm,
        vad=SileroVAD(),
    )

    current_language_code = "en-IN"

    async def detect_language(transcript: str) -> str:
        """Detect the language of the user's transcript using Sarvam AI."""
        api_key = os.getenv("SARVAMAI_API_KEY")
        if not api_key:
            return current_language_code
        try:
            client = SarvamAI(api_subscription_key=api_key)
            response = client.text.identify_language(input=transcript)
            logger.info("Detected language: %s", response.language_code)
            return response.language_code
        except Exception as e:
            logger.warning("Language detection failed: %s", e)
            return current_language_code

    @pipeline.on("user_turn_start")
    async def on_user_turn_start(transcript: str):
        nonlocal current_language_code
        detected = await detect_language(transcript)
        if detected and detected != current_language_code:
            current_language_code = detected
            await pipeline.change_component(
                tts=SarvamAITTS(
                    api_key=os.getenv("SARVAMAI_API_KEY"),
                    language=current_language_code,
                )
            )
            logger.info("Switched TTS to language: %s", current_language_code)

    agent = MultilingualSupportAgent()
    session = AgentSession(agent=agent, pipeline=pipeline)
    await session.start(wait_for_participant=True, run_until_shutdown=True)


def make_context() -> JobContext:
    room_options = RoomOptions(room_id="<room_id>", name="GlobalApp Multilingual Support", playground=True)
    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()
