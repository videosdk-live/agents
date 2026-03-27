from videosdk.agents import Agent, AgentSession, Pipeline, JobContext, RoomOptions, WorkerJob
from videosdk.plugins.google import GeminiRealtime, GeminiLiveConfig
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])

class MyVoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You Are VideoSDK's Voice Agent.You are a helpful voice assistant that can answer questions and help with tasks.",
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you today?")
    
    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")

async def start_session(context: JobContext):
    agent = MyVoiceAgent()
    model = GeminiRealtime(
        model="gemini-3.1-flash-live-preview",
        # When GOOGLE_API_KEY is set in .env - DON'T pass api_key parameter
        # api_key="AIXXXXXXXXXXXXXXXXXXXX", 
        config=GeminiLiveConfig(
            voice="Leda", # Puck, Charon, Kore, Fenrir, Aoede, Leda, Orus, and Zephyr.
            response_modalities=["AUDIO"]
        )
    )

    pipeline = Pipeline(llm=model)
    session = AgentSession(
        agent=agent,
        pipeline=pipeline
    )

    await session.start(wait_for_participant=True, run_until_shutdown=True)

def make_context() -> JobContext:
    room_options = RoomOptions(
        # room_id="<room_id>", # Replace it with your actual room_id
        name="Gemini Realtime Agent",
        playground=True,
    )

    return JobContext(room_options=room_options)

if __name__ == "__main__":
    job = WorkerJob(entrypoint=start_session, jobctx=make_context)
    job.start()