"""
VideoSDK Avatar — Cascade Agent

This example shows how to wire the VideoSDK data-channel avatar into the
Agents framework's Pipeline.  The avatar service is a separate process (see videosdk_avatar_service.py)
that renders audio as a circular-glow video and publishes it from its own
participant slot in the meeting.

Required env vars:
    VIDEOSDK_AUTH_TOKEN   – agent's VideoSDK auth token
    VIDEOSDK_API_KEY      – VideoSDK API key (for signing the avatar JWT)
    VIDEOSDK_SECRET_KEY   – VideoSDK secret key
    VIDEOSDK_ROOM_ID      – target meeting ID
    DEEPGRAM_API_KEY
    OPENAI_API_KEY
    ELEVENLABS_API_KEY

The avatar dispatcher must be running before starting the agent:
    python examples/avatar/videosdk_avatar_launcher.py
"""

import logging
import os

from dotenv import load_dotenv

from videosdk.agents import Agent, AgentSession, Pipeline, JobContext, RoomOptions, WorkerJob
from videosdk.agents.avatar import AvatarAudioOut, generate_avatar_credentials
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
from videosdk.plugins.google import GoogleLLM, GoogleTTS

load_dotenv(override=True)
pre_download_model()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

class MyVoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are VideoSDK's AI avatar voice assistant. "
                "You have a visual avatar that renders as a glowing waterfall spectrogram. "
                "Be concise and conversational."
            )
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello! I'm your AI avatar assistant. How can I help you today?")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye! It was nice talking with you!")

async def start_session(context: JobContext):
    api_key = os.getenv("VIDEOSDK_API_KEY", "")
    secret_key = os.getenv("VIDEOSDK_SECRET_KEY", "")

    credentials = generate_avatar_credentials(api_key, secret_key)

    avatar_audio_out = AvatarAudioOut(
        credentials=credentials,
        avatar_dispatcher_url="http://localhost:8089/launch",
    )

    agent = MyVoiceAgent()

    pipeline = Pipeline(
        stt=DeepgramSTT(),
        llm=GoogleLLM(),
        tts=GoogleTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(threshold=0.8),
        avatar=avatar_audio_out,
    )

    session = AgentSession(agent=agent, pipeline=pipeline)
    await session.start(wait_for_participant=True, run_until_shutdown=True)

def make_context() -> JobContext:
    room_options = RoomOptions(
        room_id=os.getenv("VIDEOSDK_ROOM_ID", ""),
        name="VideoSDK Avatar Agent",
        playground=False,
        agent_participant_id="agent_brain",
    )
    return JobContext(room_options=room_options)

if __name__ == "__main__":
    job = WorkerJob(entrypoint=start_session, jobctx=make_context)
    job.start()