import asyncio
import aiohttp
from videosdk.agents import Agent, AgentSession, RealTimePipeline, function_tool
from videosdk.plugins.google import GeminiRealtime, GeminiLiveConfig
from google.genai.types import AudioTranscriptionConfig


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
            instructions="You Are VideoSDK's Voice Agent.You are a helpful voice assistant that can answer questions and help with tasks.",
            tools=[get_weather]
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you today?")
    
    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")
        
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
    
    @function_tool
    async def end_call(self) -> None:
        """End the call upon request by the user"""
        await self.session.say("Goodbye!")
        await asyncio.sleep(1)
        await self.session.leave()
        


async def main(context: dict):
    model = GeminiRealtime(
        model="gemini-2.0-flash-live-001",
        # When GOOGLE_API_KEY is set in .env - DON'T pass api_key parameter
        config=GeminiLiveConfig(
            response_modalities=["TEXT"],
            voice=None,
            language_code=None,
        )
    )

    pipeline = RealTimePipeline(model=model)
    session = AgentSession(
        agent=MyVoiceAgent(),
        pipeline=pipeline,
        context=context
    )

    try:
        await session.start()

        # Handler for text responses from the agent
        def on_text_response(data):
            print(f"Agent Response: {data.get('text')}")

        # Attach the handler to the GeminiRealtime model instance
        if hasattr(pipeline, 'model') and hasattr(pipeline.model, 'on'):
            pipeline.model.on("text_response", on_text_response)
        else:
            print("Could not attach text_response handler. Ensure pipeline.model is GeminiRealtime and has event handling.")

        # Simple text input loop
        while True:
            user_input = await asyncio.to_thread(input, "You: ")
            if user_input.lower() == "exit":
                print("Exiting chat...")
                break
            if hasattr(pipeline, 'model') and hasattr(pipeline.model, 'send_message'):
                await pipeline.model.send_message(user_input)
            else:
                print("Could not send message. Ensure pipeline.model is GeminiRealtime and has send_message.")
                break
        
        await asyncio.Event().wait() # This will now be unblocked by breaking the loop or KeyboardInterrupt

    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        await session.close()

if __name__ == "__main__":
    def make_context():
        return {"meetingId": "j7wx-r9jv-qni2", "name": "Gemini Agent"}
    
    asyncio.run(main(context=make_context()))