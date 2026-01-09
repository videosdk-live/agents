"""
Test for VideoSDK Inference Realtime with RealTimePipeline.

Realtime: VideoSDK Inference Gateway (Gemini 2.0 Flash)

This uses the inference gateway to handle the Gemini API connection,
making the client lightweight and reducing latency.

Required environment variables:
    VIDEOSDK_AUTH_TOKEN - VideoSDK authentication token

Run:
    python examples/test_inference_realtime.py
"""

import logging
from videosdk.agents import (
    Agent,
    AgentSession,
    RealTimePipeline,
    ConversationFlow,
    JobContext,
    RoomOptions,
    WorkerJob,
)
from videosdk.agents.inference import Realtime

# Minimal logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class SimpleAgent(Agent):
    """Simple voice agent for testing inference realtime."""

    def __init__(self):
        super().__init__(
            instructions="""You are a helpful and friendly voice assistant. 
You speak in a natural, conversational tone. Keep your responses concise but informative.
When asked questions, provide clear and helpful answers.""",
        )

    async def on_enter(self) -> None:
        await self.session.say(
            "Hello! I'm using the VideoSDK Inference Gateway with Gemini. How can I help you today?"
        )

    async def on_exit(self) -> None:
        await self.session.say("Goodbye! Have a great day!")


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the agent."""

    agent = SimpleAgent()
    conversation_flow = ConversationFlow(agent)

    # Create RealTimePipeline with Inference Realtime (Gemini)
    pipeline = RealTimePipeline(
        model=Realtime.gemini(
            model="gemini-2.0-flash-exp",
            voice="Puck",  # Options: Puck, Charon, Kore, Fenrir, Aoede
            language_code="en-US",
            response_modalities=["AUDIO"],  # ["TEXT", "AUDIO"] for text+audio
            temperature=0.7
        ),
    )

    session = AgentSession(
        agent=agent,
        pipeline=pipeline,
        conversation_flow=conversation_flow,
    )

    await session.start(wait_for_participant=True, run_until_shutdown=True)


def make_context() -> JobContext:
    """Create job context for playground mode."""
    room_options = RoomOptions(
        room_id="<ROOM_ID>",
        name="Inference Realtime Test Agent",
        playground=True
    )
    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()
