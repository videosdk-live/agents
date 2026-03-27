import logging
from videosdk.agents import Agent, AgentSession, Pipeline, JobContext, RoomOptions, WorkerJob
from videosdk.agents.inference import Realtime
from dotenv import load_dotenv
load_dotenv(override=True)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])

class VideoSDKRealtimeInferenceAgent(Agent):
    """VideoSDK Inference Agent for voice interaction."""

    def __init__(self):
        super().__init__(
            instructions="""
                You are VideoSDK Inference Agent, a professional voice assistant.

                Guidelines:
                - Keep responses concise and conversational.
                - Speak clearly and naturally.
                - Be polite, helpful, and precise.
                - Avoid long or complex explanations unless necessary.
                - Ask for clarification if the user input is unclear.
                """
        )

    async def on_enter(self) -> None:
        await self.session.say(
            "Hello. This is VideoSDK Inference Agent With Realtime Model. "
            "I am ready to process your voice input in real time. "
            "How can I assist you today?"
        )

    async def on_exit(self) -> None:
        await self.session.say(
            "Session ended. Thank you for using VideoSDK Inference Agent."
        )

async def entrypoint(ctx: JobContext):
    """Main entrypoint for the agent."""

    agent = VideoSDKRealtimeInferenceAgent()

    pipeline = Pipeline(
        llm=Realtime.gemini(
            model="gemini-2.5-flash-native-audio-preview-12-2025",
            voice="Puck",  # Options: Puck, Charon, Kore, Fenrir, Aoede
            language_code="en-US",
            response_modalities=["AUDIO"],  # ["TEXT", "AUDIO"] for text+audio
            temperature=0.7
        )
    )

    session = AgentSession(
        agent=agent,
        pipeline=pipeline,
    )

    await session.start(wait_for_participant=True,run_until_shutdown=True)

def make_context() -> JobContext:
    """Create job context for playground mode."""

    room_options = RoomOptions(
        room_id="<room_id>",
        name="VideoSDK's Realtime Inference Agent",
        playground=True,
    )

    return JobContext(room_options=room_options)

if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint,jobctx=make_context())
    job.start()