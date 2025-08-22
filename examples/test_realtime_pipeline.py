# This test script is used to test realtime pipeline.
import asyncio
import logging
from videosdk.agents import Agent, AgentSession, RealTimePipeline, function_tool, WorkerJob, JobContext, RoomOptions
from videosdk.plugins.google import GeminiRealtime, GeminiLiveConfig

logging.getLogger().setLevel(logging.CRITICAL)
class RealtimeAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=""" You are a helpful voice assistant that can answer questions and help with tasks. """,
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you today?")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")

    @function_tool
    async def get_horoscope(self, sign: str) -> dict:
        horoscopes = {
            "Aries": "Today is your lucky day!",
            "Taurus": "Focus on your goals today.",
            "Gemini": "Communication will be important today.",
        }
        return {
            "sign": sign,
            "horoscope": horoscopes.get(sign, "The stars are aligned for you today!"),
        }

async def entrypoint(ctx: JobContext):
    
    model = GeminiRealtime(
        model="gemini-2.0-flash-live-001",
        config=GeminiLiveConfig(
            voice="Leda", # Puck, Charon, Kore, Fenrir, Aoede, Leda, Orus, and Zephyr.
            response_modalities=["AUDIO"]
        )
    )
    
    pipeline = RealTimePipeline(model=model)
    
    def on_transcription(data: dict):
        role = data.get("role")
        text = data.get("text")
        print(f"[TRANSCRIPT][{role}: {text}")

        pipeline.on("realtime_model_transcription", on_transcription)

    agent = RealtimeAgent()
    
    session = AgentSession(
        agent=agent,
        pipeline=pipeline,
    )

    try:
        await ctx.connect()
        print("Waiting for participant...")
        await ctx.room.wait_for_participant()
        print("Participant joined")
        await session.start()
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        await session.close()
        await ctx.shutdown()

def make_context() -> JobContext:
    room_options = RoomOptions(room_id="4gga-v342-sfe9", name="Sandbox Agent", playground=True) 
    return JobContext(
        room_options=room_options
        )


if __name__ == "__main__":

    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()
