VideoSDK Agents

Agents Framework on top of VideoSDK's architecture.

## Installation

```bash
pip install videosdk-agents
```

Visit https://docs.videosdk.live for Quickstart, Examples and Detailed Documentation.

## Usage

```py
import asyncio
from videosdk.agents import Agent, AgentSession, RealTimePipeline, function_tool, WorkerJob
from videosdk.plugins.openai import OpenAIRealtime, OpenAIRealtimeConfig
from openai.types.beta.realtime.session import InputAudioTranscription, TurnDetection


class MyVoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful voice assistant that can answer questions and help with tasks.",
        )

    async def on_enter(self) -> None:
        await self.session.say("How can i assist you today?")

async def test_connection(jobctx):
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
        await session.start()
        await asyncio.Event().wait()
    except KeyboardInterrupt:
    finally:
        await session.close()


def entryPoint(jobctx):
    jobctx["pid"] = os.getpid()
    asyncio.run(test_connection(jobctx))


if __name__ == "__main__":
    def make_context():
        return {"meetingId": "<Meet-ID>", "name": "Agent"}
    job = WorkerJob(job_func=entryPoint, jobctx=make_context)
    job.start()
```


