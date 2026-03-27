"""
Use Case: Wellness meditation guide (Serenity) — plays ambient audio during sessions, guides with breathing exercises.
Pipeline: P1 — DeepgramSTT + OpenAILLM + OpenAITTS + SileroVAD + TurnDetector
Demonstrates: set_thinking_audio(), play_background_audio(looping=True), stop_background_audio() via function tools.
Env Vars: VIDEOSDK_AUTH_TOKEN, DEEPGRAM_API_KEY, OPENAI_API_KEY
"""

import logging
from videosdk.agents import Agent, AgentSession, Pipeline, function_tool, WorkerJob, JobContext, RoomOptions
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.openai import OpenAILLM, OpenAITTS
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
pre_download_model()


class MeditationGuideAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are Serenity, a calm and gentle meditation guide.
            Help users with breathing exercises, mindfulness check-ins, and short guided meditation sessions.
            When a session starts (via start_session tool), play calming background music automatically.
            When a session ends or the user speaks, stop the music.
            Keep your voice slow, warm, and unhurried. Use short pauses between instructions.
            Never rush the user. Celebrate small moments of calm.
            Common session types: breathing (4-7-8 technique), body scan, gratitude, visualization.""",
        )
        # Play a soft sound while the agent is "thinking" between responses.
        self.set_thinking_audio()

    async def on_enter(self) -> None:
        await self.session.say(
            "Welcome. I'm Serenity, your meditation guide. "
            "Find a comfortable seat, close your eyes when you're ready, and breathe. "
            "Say 'start session' to begin a guided meditation, or ask me anything about mindfulness."
        )

    async def on_exit(self) -> None:
        await self.stop_background_audio()
        await self.session.say(
            "Take a moment to return to your surroundings. "
            "Carry this calm with you. Until next time — namaste."
        )

    @function_tool
    async def start_session(self, duration_minutes: int = 5, session_type: str = "breathing") -> dict:
        """Begin a guided meditation session with ambient background music.

        Args:
            duration_minutes: Length of the session in minutes (default: 5)
            session_type: Type of session — 'breathing', 'body_scan', 'gratitude', 'visualization'
        """
        await self.play_background_audio(override_thinking=False, looping=True)
        return {
            "status": "started",
            "session_type": session_type,
            "duration_minutes": duration_minutes,
            "message": f"A {duration_minutes}-minute {session_type} session has begun. Background music is playing.",
        }

    @function_tool
    async def stop_session(self) -> dict:
        """End the current meditation session and stop background music."""
        await self.stop_background_audio()
        return {
            "status": "ended",
            "message": "Session complete. Background music stopped.",
        }


async def entrypoint(ctx: JobContext):
    agent = MeditationGuideAgent()

    pipeline = Pipeline(
        stt=DeepgramSTT(),
        llm=OpenAILLM(),
        tts=OpenAITTS(voice="nova"),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
    )
    session = AgentSession(agent=agent, pipeline=pipeline)
    await session.start(wait_for_participant=True, run_until_shutdown=True)


def make_context() -> JobContext:
    room_options = RoomOptions(
        # room_id="el00-4yu2-jjng",
        name="Serenity Meditation Guide",
        playground=True,
        background_audio=True,
        # signaling_base_url="dev-api.videosdk.live"
    )
    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()
