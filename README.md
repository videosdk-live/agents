# VideoSDK Agents

Agents Framework on top of VideoSDK's architecture. The Agents Framework lets you build Voice AI agents that can hear, see*, and speak in realtime. Agents Framework is an open-source platform that helps creating server-side agentic applications.

## Installation

```bash
pip install videosdk-agents
```

## Features
- **Realtime Pipeline**: Use realtime model from providers like openai, google, aws for low latency agentic conversation/ tasks.
- **Cascading Pipeline**: Use cascading pipeline to get more control over the choice of models in your STT, LLM, TTS from available providers for better flexibility according to your use case. To make conversations more human-like, cascading also supports VAD and turn detection.
- **Semantic Turn detection**: Uses a BERT model to detect when a user is done speaking, helps to reduce interruptions.
- **Telephony Support**: Works seamlessly with VideoSDK's [telephony](https://docs.videosdk.live/react/guide/sip-connect/overview), allowing your agent to make/receive calls from phones.
- **WebRTC Client Support**: Use VideoSDK's [client SDKs](https://docs.videosdk.live/) to build client applications, supporting nearly all major platforms.
- **MCP Support**: Native support for MCP. Integrate [MCP](https://docs.videosdk.live/ai_agents/mcp-integration) tools seamlessly.
- **Open-Source**: Allows you to run the entire agent stack on your own servers, including VideoSDK's AgentCloud that has built-in support of Worker Job.

## Documentation and Guides

Visit [Agents Docs](https://docs.videosdk.live/ai_agents/introduction) for Quickstart, Examples and Detailed Documentation.

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
        await self.session.say("How can I assist you today?")

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
        return {"meetingId": "<meeting_id>", "name": "Sandbox Agent", "playground": True}

    asyncio.run(entryPoint(make_context()))
```

You'll need following environment variables for this examples:
- OPENAI_API_KEY
- [VIDEOSDK_AUTH_TOKEN](https://docs.videosdk.live/react/guide/video-and-audio-calling-api-sdk/authentication-and-token)


