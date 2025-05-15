import asyncio
import os
from videosdk.plugins.openai import OpenAIRealtime, OpenAIRealtimeConfig
from videosdk.plugins.google import GeminiRealtime, GeminiLiveConfig
from videosdk.agents import Agent, AgentSession, RealTimePipeline, function_tool, WorkerJob
from google.genai.types import AudioTranscriptionConfig
import aiohttp
import logging
from openai.types.beta.realtime.session import InputAudioTranscription, TurnDetection

logger = logging.getLogger(__name__)


class MyVoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful voice assistant that can answer questions and help with tasks.",
        )
        self.register_tools([self.get_weather, self.get_horoscope])

    async def on_enter(self) -> None:
        await self.session.say(" i am a voice assistant created by google. i'm here to help you with your meeting, if you have any questions or need help, please let me know.")
    
    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")

    @function_tool
    async def get_weather(
        self,
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

    # Static test function
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


async def test_connection(jobctx):
    print("Starting connection test...")
    print(f"Job context: {jobctx}")
    
    # model = OpenAIRealtime(
    #     model="gpt-4o-realtime-preview",
    #     config=OpenAIRealtimeConfig(
    #         modalities=["text", "audio"],
    #         input_audio_transcription=InputAudioTranscription(
    #             model="whisper-1"
    #         ),
    #         turn_detection=TurnDetection(
    #             type="server_vad",
    #             threshold=0.5,
    #             prefix_padding_ms=300,
    #             silence_duration_ms=200,
    #         ),
    #         tool_choice="auto"
    #     )
    # )
    model = GeminiRealtime(
        model="gemini-2.0-flash-live-001",
        config=GeminiLiveConfig(
            response_modalities=["AUDIO"],
            output_audio_transcription=AudioTranscriptionConfig(
            )
        )
    )
    pipeline = RealTimePipeline(model=model)
    session = AgentSession(
        agent=MyVoiceAgent(), 
        pipeline=pipeline,
        context=jobctx
    )

    try:
        await session.start()
        print("Connection established. Press Ctrl+C to exit.")
        await asyncio.Event().wait()
        # await asyncio.sleep(30)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    finally:
        await session.close()


def entryPoint(jobctx):
    jobctx["pid"] = os.getpid()
    asyncio.run(test_connection(jobctx))


if __name__ == "__main__":

    def make_context():
        return {"meetingId": "pbow-6vec-vahn", "name": "Agent"}

    job = WorkerJob(job_func=entryPoint, jobctx=make_context)
    job.start()
