import logging
from videosdk.agents import Agent, AgentSession, Pipeline, JobContext, RoomOptions, WorkerJob
from videosdk.plugins.xai import XAIRealtime, XAIRealtimeConfig,XAITurnDetection
from videosdk.plugins.cartesia import CartesiaTTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

class RealtimeAndTTS(Agent):
    """Voice agent using XAI Realtime with external TTS"""
    
    def __init__(self):
        super().__init__(instructions="You are a helpful voice assistant. Speak naturally and conversationally.")

    async def on_enter(self) -> None:
        await self.session.say("Hello! I'm using XAI Realtime with a custom voice. How can I help you?")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")


async def entrypoint(ctx: JobContext):
    """
    Create a voice agent with hybrid_tts mode using XAI Realtime:
    - Realtime STT+LLM: XAI Realtime handles speech recognition and language processing
    - External TTS: Cartesia TTS for custom voice synthesis
    """

    llm = XAIRealtime(
        model="grok-4-1-fast-non-reasoning",        
        config=XAIRealtimeConfig(
            voice="Eve",
            enable_web_search=True,
            # enable_x_search=True,
            # allowed_x_handles=["elonmusk"],
            # collection_id="your-collection-id",
            turn_detection=XAITurnDetection(
                type="server_vad",
                threshold=0.5,
                prefix_padding_ms=300,
                silence_duration_ms=200,
            ),
        )
    )
  
    pipeline = Pipeline(
        llm=llm,          
        tts=CartesiaTTS(),    
    )

    agent = RealtimeAndTTS()
    session = AgentSession(agent=agent, pipeline=pipeline)
    await session.start(wait_for_participant=True, run_until_shutdown=True)

def make_context() -> JobContext:
    room_options = RoomOptions(room_id="<room_id>",name="Realtime+TTS Agent",playground=True)
    return JobContext(room_options=room_options)


if __name__ == "__main__":    
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()
