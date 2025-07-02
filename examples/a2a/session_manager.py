from videosdk.agents import AgentSession, RealTimePipeline
from videosdk.plugins.openai import OpenAIRealtime, OpenAIRealtimeConfig
from openai.types.beta.realtime.session import TurnDetection
from videosdk.plugins.google import GeminiRealtime, GeminiLiveConfig
from typing import Dict

def create_pipeline(agent_type: str) -> RealTimePipeline:
    if agent_type == "customer":
        model = GeminiRealtime(
        model="gemini-2.0-flash-live-001",
        config=GeminiLiveConfig(
            voice="Leda",
            response_modalities=["AUDIO"]
        )
    )
    else:
        model = GeminiRealtime(
            model="gemini-2.0-flash-live-001",
            config=GeminiLiveConfig(response_modalities=["TEXT"])
        )

    return RealTimePipeline(model=model)


def create_session(agent, pipeline) -> AgentSession:
    return AgentSession(agent=agent, pipeline=pipeline)