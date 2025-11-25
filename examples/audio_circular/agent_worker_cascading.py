
from videosdk.agents import Agent, AgentSession, JobContext, CascadingPipeline,ConversationFlow
from videosdk.agents.avatar import AvatarDataChannel
from videosdk.plugins.google import GoogleLLM
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.elevenlabs import ElevenLabsTTS
import os
from dotenv import load_dotenv
import logging

load_dotenv()

pre_download_model()

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
    conversation_flow = ConversationFlow(agent)

    pipeline = CascadingPipeline(
        stt=DeepgramSTT(),
        llm=GoogleLLM(),
        tts=ElevenLabsTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
        avatar=ctx.room_options.avatar,
    )
    session = AgentSession(
        agent=agent, 
        pipeline=pipeline,
        conversation_flow=conversation_flow
    )

    await session.start(wait_for_participant=True, run_until_shutdown=True)



if __name__ == "__main__":
    from videosdk.agents.job import JobContext, RoomOptions, WorkerJob
    import sys
    
    VIDEOSDK_ROOM_ID = os.getenv("VIDEOSDK_ROOM_ID")
    VIDEOSDK_API_KEY = os.getenv("VIDEOSDK_API_KEY") 
    VIDEOSDK_SECRET_KEY = os.getenv("VIDEOSDK_SECRET_KEY")
    AGENT_TOKEN = os.getenv("VIDEOSDK_AUTH_TOKEN") 

    if not (VIDEOSDK_ROOM_ID and VIDEOSDK_API_KEY and VIDEOSDK_SECRET_KEY and AGENT_TOKEN):
        print("Error: Missing environment variables. Need VIDEOSDK_ROOM_ID, VIDEOSDK_API_KEY, VIDEOSDK_SECRET_KEY, and VIDEOSDK_AUTH_TOKEN")
        sys.exit(1)

    room_options = RoomOptions(
        room_id=VIDEOSDK_ROOM_ID,
        auth_token=AGENT_TOKEN,
        playground=False,
        agent_participant_id="agent_brain",
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