VideoSDK Agents

Agents Framework on top of VideoSDK's architecture.

## Installation

```bash
pip install videosdk-agents
```

Visit https://docs.videosdk.live/ai_agents/introduction for Quickstart, Examples and Detailed Documentation.

## Usage

```py
import asyncio
from videosdk.agents import Agent, AgentSession, RealTimePipeline, function_tool, WorkerJob, RoomOptions, JobContext
from videosdk.plugins.openai import OpenAIRealtime, OpenAIRealtimeConfig
from openai.types.beta.realtime.session import InputAudioTranscription, TurnDetection


class MyVoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful voice assistant that can answer questions and help with tasks.",
        )

    async def on_enter(self) -> None:
        await self.session.say("How can i assist you today?")

async def entrypoint(ctx: JobContext):
    print("Starting connection test...")
    print(f"Job context: {jobctx}")
    
    model = OpenAIRealtime(
        model="gpt-4o-realtime-preview",
        config=OpenAIRealtimeConfig( modalities=["text", "audio"] )
    )
    pipeline = RealTimePipeline(model=model)
    session = AgentSession(
        agent=MyVoiceAgent(), 
        pipeline=pipeline,
        context=jobctx
    )

    try:
        await ctx.connect()
        await session.start()
        print("Connection established. Press Ctrl+C to exit.")
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    finally:
        await session.close()
        await ctx.shutdown()


def make_context() -> JobContext:
    room_options = RoomOptions(room_id="<meeting_id>", name="Sandbox Agent", playground=True)
    
    return JobContext(
        room_options=room_options
        )


if __name__ == "__main__":
    job = WorkerJob(job_func=entryPoint, jobctx=make_context)
    job.start()
```


