import asyncio
import aiohttp
import os

from videosdk.agents import Agent, AgentSession, RealTimePipeline, function_tool, JobContext, RoomOptions, WorkerJob
from videosdk.plugins.google import GeminiRealtime, GeminiLiveConfig
from videosdk.plugins.simli import SimliAvatar, SimliConfig

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
        await self.session.say("Hello! I'm your real-time AI avatar assistant powered by VideoSDK. How can I help you today?")
    
    async def on_exit(self) -> None:
        await self.session.say("Goodbye! It was great talking with you!")
        

async def start_session(context: JobContext):
    # Initialize Gemini Realtime model
    model = GeminiRealtime(
        model="gemini-2.0-flash-live-001",
        # When GOOGLE_API_KEY is set in .env - DON'T pass api_key parameter
        # api_key="AIXXXXXXXXXXXXXXXXXXXX", 
        config=GeminiLiveConfig(
            voice="Leda",  # Puck, Charon, Kore, Fenrir, Aoede, Leda, Orus, and Zephyr.
            response_modalities=["AUDIO"]
        )
    )

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

    # Create pipeline with avatar
    pipeline = RealTimePipeline(model=model, avatar=simli_avatar)

    session = AgentSession(agent=MyVoiceAgent(), pipeline=pipeline)

    await context.run_until_shutdown(session=session)

def make_context() -> JobContext:
    room_options = RoomOptions(
        room_id="<room_id>",
        name="Simli Avatar Realtime Agent",
        playground=False 
    )

    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=start_session, jobctx=make_context)
    job.start() 