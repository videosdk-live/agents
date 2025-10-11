# This test script is used to test realtime pipeline.
import asyncio
import logging
from videosdk.agents import Agent, AgentSession, RealTimePipeline,WorkerJob, JobContext, RoomOptions
from videosdk.plugins.openai import OpenAIRealtime, OpenAIRealtimeConfig
from openai.types.beta.realtime.session import TurnDetection

logging.getLogger().setLevel(logging.CRITICAL)
class RealtimeAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are a high-energy game-show host guiding the caller to guess a secret number from 1 to 100 to win 1,000,000$.""",
        )

    async def on_enter(self) -> None:
        await self.session.say("Welcome to the Videosdk's AI Agent game show! I'm your host, and we're about to play for 1,000,000$. Are you ready to play?")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")

async def entrypoint(ctx: JobContext):

    # Initialize the OpenAI GPT real-time model
    model = OpenAIRealtime(
        model="gpt-realtime-2025-08-28",
        config=OpenAIRealtimeConfig(
            voice="alloy", # alloy, ash, ballad, coral, echo, fable, onyx, nova, sage, shimmer, and verse
            modalities=["audio"],
            turn_detection=TurnDetection(
                type="server_vad",
                threshold=0.5,
                prefix_padding_ms=300,
                silence_duration_ms=200,
            ),
            tool_choice="auto"
        )
    )
    
    pipeline = RealTimePipeline(model=model)
    
    agent = RealtimeAgent()
    
    session = AgentSession(
        agent=agent,
        pipeline=pipeline,
    )

    await ctx.run_until_shutdown(session=session,wait_for_participant=True)

def make_context() -> JobContext:
    room_options = RoomOptions(room_id="<room_id>", name="Sandbox Agent", playground=True) 
    return JobContext(
        room_options=room_options
        )


if __name__ == "__main__":

    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()
