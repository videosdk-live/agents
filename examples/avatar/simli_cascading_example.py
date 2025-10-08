import asyncio
import aiohttp
import os
from typing import AsyncIterator

from videosdk.agents import Agent, AgentSession, CascadingPipeline, function_tool, JobContext, RoomOptions, WorkerJob, ConversationFlow, ChatRole
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
from videosdk.plugins.simli import SimliAvatar, SimliConfig
from videosdk.plugins.openai import OpenAILLM
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.elevenlabs import ElevenLabsTTS

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
        await self.session.say("Hello! I'm your AI avatar assistant powered by VideoSDK. How can I help you today?")
    
    async def on_exit(self) -> None:
        await self.session.say("Goodbye! It was nice talking with you!")
        

async def start_session(context: JobContext):

    stt = DeepgramSTT(model="nova-3", language="multi", api_key=os.getenv("DEEPGRAM_API_KEY"))
    llm = OpenAILLM(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    tts = ElevenLabsTTS(api_key=os.getenv("ELEVENLABS_API_KEY"), enable_streaming=True, speed=1.2)
    
    # Initialize VAD and Turn Detector
    vad = SileroVAD()
    turn_detector = TurnDetector(threshold=0.8)

    # Initialize Simli Avatar
    simli_config = SimliConfig(
        apiKey=os.getenv("SIMLI_API_KEY"),
        faceId="d2a5c7c6-fed9-4f55-bcb3-062f7cd20103",
        maxSessionLength=1800,
        maxIdleTime=600,
    )
   
    simli_avatar = SimliAvatar(
        config=simli_config,
        is_trinity_avatar=True,
    )

    # Create agent and conversation flow
    agent = MyVoiceAgent()
    conversation_flow = ConversationFlow(agent)

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

    await context.run_until_shutdown(session=session)

def make_context() -> JobContext:
    room_options = RoomOptions(
        room_id="<room_id>",
        name="Simli Avatar Cascading Agent",
        playground=False
    )

    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=start_session, jobctx=make_context)
    job.start() 