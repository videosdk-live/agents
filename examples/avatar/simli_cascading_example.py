import asyncio
import aiohttp
import os
from typing import AsyncIterator

from videosdk.agents import Agent, AgentSession, CascadingPipeline, function_tool, JobContext, RoomOptions, WorkerJob, ConversationFlow, ChatRole
from videosdk.plugins.google import GoogleTTS, GoogleLLM, GoogleSTT
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
from videosdk.plugins.simli import SimliAvatar, SimliConfig

# Pre-downloading the Turn Detector model
pre_download_model()

@function_tool
async def get_weather(
    latitude: str,
    longitude: str,
):
    """Called when the user asks about the weather. This function will return the weather for
    the given location. When given a location, please estimate the latitude and longitude of the
    location and do not ask the user for them.

    Args:
        latitude: The latitude of the location
        longitude: The longitude of the location
    """
    print("###Getting weather for", latitude, longitude)
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m"
    weather_data = {}
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                print("###Weather data", data)
                weather_data = {
                    "temperature": data["current"]["temperature_2m"],
                    "temperature_unit": "Celsius",
                }
            else:
                raise Exception(
                    f"Failed to get weather data, status code: {response.status}"
                )

    return weather_data


class MyVoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are VideoSDK's AI Avatar Voice Agent with real-time capabilities. You are a helpful virtual assistant with a visual avatar that can answer questions about weather help with other tasks in real-time.",
            tools=[get_weather]
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello! I'm your AI avatar assistant. How can I help you today?")
    
    async def on_exit(self) -> None:
        await self.session.say("Goodbye! It was nice talking with you!")
        

class MyConversationFlow(ConversationFlow):
    def __init__(self, agent: Agent):
        super().__init__(agent)

    async def run(self, transcript: str) -> AsyncIterator[str]:
        """Main conversation loop: handle a user turn."""
        await self.on_turn_start(transcript)
        processed_transcript = transcript.lower().strip()
        self.agent.chat_context.add_message(
            role=ChatRole.USER, content=processed_transcript
        )
        async for response_chunk in self.process_with_llm():
            yield response_chunk
        await self.on_turn_end()

    async def on_turn_start(self, transcript: str) -> None:
        """Called at the start of a user turn."""
        self.is_turn_active = True
        print(f"User transcript: {transcript}")

    async def on_turn_end(self) -> None:
        """Called at the end of a user turn."""
        self.is_turn_active = False
        print("Agent turn ended.")


async def start_session(context: JobContext):
    # Initialize Google services
    stt = GoogleSTT(model="latest_long")
    llm = GoogleLLM(api_key=os.getenv("GOOGLE_API_KEY"))
    tts = GoogleTTS(api_key=os.getenv("GOOGLE_API_KEY"))
    
    # Initialize VAD and Turn Detector
    vad = SileroVAD()
    turn_detector = TurnDetector(threshold=0.8)

    # Initialize Simli Avatar
    simli_config = SimliConfig(
        apiKey=os.getenv("SIMLI_API_KEY")
    )

    simli_avatar = SimliAvatar(config=simli_config)

    # Create agent and conversation flow
    agent = MyVoiceAgent()
    conversation_flow = MyConversationFlow(agent)

    # Create pipeline with avatar
    pipeline = CascadingPipeline(
        stt=stt, 
        llm=llm, 
        tts=tts, 
        vad=vad, 
        turn_detector=turn_detector,
        avatar=simli_avatar
    )

    session = AgentSession(
        agent=agent,
        pipeline=pipeline,
        conversation_flow=conversation_flow
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
        room_id="YOUR_MEETING_ID",  # Replace it with your actual meetingID
        # auth_token = "<VIDEOSDK_AUTH_TOKEN>",  # When VIDEOSDK_AUTH_TOKEN is set in .env - DON'T include videosdk_auth
        name="Simli Avatar Cascading Agent",
        playground=False
    )

    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=start_session, jobctx=make_context)
    job.start() 