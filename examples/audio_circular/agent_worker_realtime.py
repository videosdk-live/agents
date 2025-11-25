import argparse
import logging
import sys
from dataclasses import asdict, dataclass
from functools import partial
import os
from dotenv import load_dotenv
from videosdk.agents import Agent, AgentSession, JobContext, RealTimePipeline
from videosdk.agents.avatar import AvatarDataChannel
from videosdk.plugins.openai import OpenAIRealtime, OpenAIRealtimeConfig
from openai.types.beta.realtime.session import InputAudioTranscription, TurnDetection
load_dotenv()

logger = logging.getLogger("avatar-example")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])

class MyVoiceAgent(Agent):
    def __init__(self):
        super().__init__(instructions="You are a helpful voice assistant.")

    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you today?")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")


async def entrypoint(ctx: JobContext):
    agent = MyVoiceAgent()
    model = OpenAIRealtime(
        model="gpt-realtime-2025-08-28",
        config=OpenAIRealtimeConfig(
            voice="alloy", 
            modalities=["text", "audio"],
            input_audio_transcription=InputAudioTranscription(model="whisper-1"),
            turn_detection=TurnDetection(
                type="server_vad",
                threshold=0.5,
                prefix_padding_ms=300,
                silence_duration_ms=200,
            ),
            tool_choice="auto",
        )
    )

    pipeline = RealTimePipeline(
        model=model,
        avatar=ctx.room_options.avatar 
    )

    session = AgentSession(
        agent=agent,
        pipeline=pipeline
    )
    await session.start(wait_for_participant=True, run_until_shutdown=True)


if __name__ == "__main__":
    from videosdk.agents.job import JobContext, RoomOptions, WorkerJob
    import sys

    VIDEOSDK_ROOM_ID = os.getenv("VIDEOSDK_ROOM_ID")
    # We need keys now to generate the distinct avatar token
    VIDEOSDK_API_KEY = os.getenv("VIDEOSDK_API_KEY") 
    VIDEOSDK_SECRET_KEY = os.getenv("VIDEOSDK_SECRET_KEY")
    # This token is for the AGENT (Brain) to join
    AGENT_TOKEN = os.getenv("VIDEOSDK_AUTH_TOKEN") 

    if not (VIDEOSDK_ROOM_ID and VIDEOSDK_API_KEY and VIDEOSDK_SECRET_KEY and AGENT_TOKEN):
        print("Error: Missing environment variables. Need VIDEOSDK_ROOM_ID, VIDEOSDK_API_KEY, VIDEOSDK_SECRET_KEY, and VIDEOSDK_AUTH_TOKEN")
        sys.exit(1)

    # 2. Create RoomOptions
    room_options = RoomOptions(
        room_id=VIDEOSDK_ROOM_ID,
        auth_token=AGENT_TOKEN,
        playground=False,
    )

    job_context = JobContext(room_options=room_options)
    avatar_credentials = job_context.create_avatar_credentials(
        api_key=VIDEOSDK_API_KEY,
        secret_key=VIDEOSDK_SECRET_KEY,
    )
    room_options.avatar = AvatarDataChannel(
        room_id=VIDEOSDK_ROOM_ID,
        credentials=avatar_credentials,
    )

    job = WorkerJob(entrypoint=entrypoint, jobctx=job_context)
    job.start()