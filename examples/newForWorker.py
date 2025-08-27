import asyncio
import logging
import aiohttp
from videosdk.agents import (
    Agent,
    Options,
    AgentSession,
    CascadingPipeline,
    function_tool,
    WorkerJob,
    ConversationFlow,
    JobContext,
    RoomOptions,
)
from videosdk.plugins.openai import OpenAILLM
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.silero import SileroVAD

# from videosdk.plugins.elevenlabs import ElevenLabsTTS
# from videosdk.plugins.sarvamai import SarvamTTS
from videosdk.plugins.google import GoogleTTS


logging.basicConfig(level=logging.INFO)


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
    # logger.info(f"getting weather for {latitude}, {longitude}")
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m"
    weather_data = {}
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                print("Weather data", data)
                weather_data = {
                    "temperature": data["current"]["temperature_2m"],
                    "temperature_unit": "Celsius",
                }
            else:
                raise Exception(
                    f"Failed to get weather data, status code: {response.status}"
                )

    return weather_data


class VoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful voice assistant that can answer questions and help with tasks and help with horoscopes and weather.",
            tools=[get_weather],
        )

    async def on_enter(self) -> None:
        print("DEBUG: Agent on_enter called")
        await self.session.say("Hello, how can I help you today?")

    async def on_exit(self) -> None:
        print("DEBUG: Agent on_exit called")
        await self.session.say("Goodbye!")

    # # Static test function
    @function_tool
    async def get_horoscope(self, sign: str) -> dict:
        """Get today's horoscope for a given zodiac sign.

        Args:
            sign: The zodiac sign (e.g., Aries, Taurus, Gemini, etc.)
        """
        horoscopes = {
            "Aries": "Today is your lucky day!",
            "Taurus": "Focus on your goals today.",
            "Gemini": "Communication will be important today.",
        }
        return {
            "sign": sign,
            "horoscope": horoscopes.get(sign, "The stars are aligned for you today!"),
        }


async def entrypoint(ctx: JobContext):
    print("DEBUG: Entrypoint started")

    agent = VoiceAgent()
    conversation_flow = ConversationFlow(agent)

    print("DEBUG: Creating pipeline with GoogleTTS")
    pipeline = CascadingPipeline(
        stt=DeepgramSTT(), llm=OpenAILLM(), tts=GoogleTTS(), vad=SileroVAD()
    )

    print("DEBUG: Creating agent session")
    session = AgentSession(
        agent=agent,
        pipeline=pipeline,
        conversation_flow=conversation_flow,
    )

    async def cleanup_session():
        pass

    ctx.add_shutdown_callback(cleanup_session)

    try:
        print("DEBUG: Connecting to room")
        await ctx.connect()
        print("Waiting for participant...")
        await ctx.room.wait_for_participant()
        print("Participant joined")
        print("DEBUG: Starting session")
        await session.start()
        print("DEBUG: Session started, waiting for events")
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    finally:
        await session.close()
        await ctx.shutdown()


def make_context() -> JobContext:
    room_options = RoomOptions(
        name="Sandbox Agent",
        room_id="4gga-v342-sfe9", 
        playground=True,
        signaling_base_url="dev-api.videosdk.live",
        auto_end_session=True,
        session_timeout_seconds=10,
    )

    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(
        entrypoint=entrypoint,
        jobctx=make_context,
        options=Options(
            signaling_base_url="dev-api.videosdk.live", log_level="INFO", register=True
        ),
    )
    job.start()