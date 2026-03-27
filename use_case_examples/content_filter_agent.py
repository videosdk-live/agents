"""
Use Case: Children's learning assistant (Buddy) for KidsLearn — filters input and output for age-appropriate content.
Pipeline: P1 — DeepgramSTT + GoogleLLM + CartesiaTTS + SileroVAD + TurnDetector + pipeline hooks
Demonstrates: pipeline.on("stt") hook to sanitize transcripts, pipeline.on("tts") hook to block unsafe output.
Env Vars: VIDEOSDK_AUTH_TOKEN, DEEPGRAM_API_KEY, GOOGLE_API_KEY, CARTESIA_API_KEY
"""

import re
import logging
from videosdk.agents import Agent, AgentSession, Pipeline, WorkerJob, JobContext, RoomOptions, run_stt, run_tts
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.google import GoogleLLM
from videosdk.plugins.cartesia import CartesiaTTS
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
pre_download_model()

# Words to redact from both input and output.
BLOCKED_WORDS = {"violence", "weapon", "kill", "die", "death", "drug", "alcohol"}

# Topics the agent should refuse to discuss.
BLOCKED_TOPICS = ("politics", "war", "terrorism", "adult", "gore")


def contains_blocked_content(text: str) -> bool:
    """Check if text contains any blocked words or topics."""
    text_lower = text.lower()
    return any(word in text_lower for word in BLOCKED_WORDS | set(BLOCKED_TOPICS))


def redact_text(text: str) -> str:
    """Replace blocked words with a safe placeholder."""
    for word in BLOCKED_WORDS:
        text = re.sub(rf"\b{word}\b", "***", text, flags=re.IGNORECASE)
    return text


class KidsLearningAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are Buddy, the friendly learning assistant for KidsLearn.
            Help children aged 6-12 with math, reading, science, and general curiosity questions.
            Always use simple words and short sentences. Be encouraging and enthusiastic.
            Never discuss violence, politics, adult topics, or anything inappropriate for children.
            If asked about a blocked topic, gently redirect: 'That's a topic for a grown-up to explain!
            Let's explore something fun instead — want to learn about dinosaurs or space?'
            End every session with a fun fact or encouraging word.""",
        )

    async def on_enter(self) -> None:
        await self.session.say(
            "Hi there! I'm Buddy, your learning pal! "
            "What are we going to discover together today? "
            "Ask me anything about math, reading, or science!"
        )

    async def on_exit(self) -> None:
        await self.session.say(
            "Great job today! Remember: learning is an adventure, and you're a superstar! "
            "Fun fact: a day on Venus is longer than a year on Venus! "
            "See you next time, explorer!"
        )


async def entrypoint(ctx: JobContext):
    agent = KidsLearningAgent()

    pipeline = Pipeline(
        stt=DeepgramSTT(),
        llm=GoogleLLM(),
        tts=CartesiaTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
    )

    @pipeline.on("stt")
    async def stt_content_filter(audio_stream):
        """Sanitize STT transcripts — redact blocked words before they reach the LLM."""
        async def filtered_audio():
            async for audio in audio_stream:
                yield audio

        async for event in run_stt(filtered_audio()):
            if event.data and event.data.text:
                original = event.data.text
                if contains_blocked_content(original):
                    logging.warning("[STT Filter] Blocked content detected: %s", original[:60])
                    event.data.text = redact_text(original)
                else:
                    logging.info("[STT] %s", original)
            yield event

    @pipeline.on("tts")
    async def tts_content_filter(text_stream):
        """Block unsafe text from being spoken — replace with a safe redirect if needed."""
        async def safe_text():
            async for text in text_stream:
                if contains_blocked_content(text):
                    logging.warning("[TTS Filter] Blocked output: %s", text[:60])
                    yield "That's not something I can talk about! Let's explore something fun instead."
                else:
                    yield text

        async for audio in run_tts(safe_text()):
            yield audio

    @pipeline.on("user_turn_start")
    async def on_user_turn_start(transcript: str):
        logging.info("[USER] %s", transcript)

    @pipeline.on("agent_turn_end")
    async def on_agent_turn_end():
        logging.info("[AGENT] Response complete.")

    session = AgentSession(agent=agent, pipeline=pipeline)
    await session.start(wait_for_participant=True, run_until_shutdown=True)


def make_context() -> JobContext:
    room_options = RoomOptions(room_id="<room_id>", name="KidsLearn - Buddy", playground=True)
    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()
