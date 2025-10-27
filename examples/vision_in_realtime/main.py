import asyncio  
from videosdk.agents import AgentSession, WorkerJob, RoomOptions, JobContext, Agent  
from videosdk.plugins.google import GeminiRealtime, GeminiLiveConfig  
from videosdk.agents import RealTimePipeline  
  
class VisionAgent(Agent):  
    def __init__(self):  
        super().__init__(  
            instructions="You are a vision-enabled AI agent. Describe what you see in the video feed and respond to questions about visual content."  
        )  
  
    async def on_enter(self) -> None:  
        """Called when the agent session starts"""  
        await self.session.say("Hi! I'm your vision agent. Show me something and I'll describe what I see!")  
  
    async def on_exit(self) -> None:  
        """Called when the agent session ends"""  
        await self.session.say("Goodbye! Thanks for testing vision functionality with me.")  
  
async def start_session(context: JobContext):  
    # Initialize Gemini with vision capabilities  
    model = GeminiRealtime(  
        model="gemini-2.0-flash-live-001",  
        config=GeminiLiveConfig(  
            voice="Leda",  
            response_modalities=["AUDIO"]  
        )  
    )  
  
    # Create real-time pipeline with vision support  
    pipeline = RealTimePipeline(model=model)  
      
    # Create agent session  
    session = AgentSession(  
        agent=VisionAgent(),  
        pipeline=pipeline  
    )  
  
    try:  
        await context.connect()  
        await session.start()  
        await asyncio.Event().wait()  
    finally:  
        await session.close()  
        await context.shutdown()  
  
def make_context() -> JobContext:  
    room_options = RoomOptions(  
        room_id="YOUR-ROOM-ID",  
        name="Vision Test Agent",  
        playground=True,  
        vision=True,  
        recording=False  
    )  
      
    return JobContext(room_options=room_options)  
  
if __name__ == "__main__":  
    job = WorkerJob(entrypoint=start_session, jobctx=make_context)  
    job.start()