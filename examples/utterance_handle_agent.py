import asyncio
import logging
import aiohttp
from videosdk.agents import Agent, AgentSession, CascadingPipeline, function_tool, WorkerJob, ConversationFlow, JobContext, RoomOptions, UtteranceHandle
from videosdk.plugins.openai import OpenAILLM
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
from videosdk.plugins.elevenlabs import ElevenLabsTTS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])

pre_download_model()

class VoiceAgent(Agent):
    """A voice agent demonstrating UtteranceHandle and interruption-aware tools."""

    def __init__(self):
        super().__init__(
            instructions=(
                "You are a helpful voice assistant. You can answer questions and fetch weather "
                "information using the 'get_weather' tool."
                "You can also perform a long-running task using the 'long_running_task' tool."
            )
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you today?")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")

    @function_tool
    async def get_weather(self, latitude: str, longitude: str) -> dict:
        """
        Fetches current weather for a given location using the Open-Meteo API.

        Supports interruption:
        If the user starts speaking while the agent is responding, the task cancels gracefully.

        Args:
            latitude (str): Latitude of the location
            longitude (str): Longitude of the location
            dont ask user for latitude and longitude, estimate it.
        """
        logger.info(f"### Getting weather for lat={latitude}, lon={longitude}")
        utterance: UtteranceHandle | None = self.session.current_utterance

        try:
            url = (
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={latitude}&longitude={longitude}&current=temperature_2m"
            )

            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise Exception(f"Weather API failed with status {response.status}")
                    data = await response.json()

            temperature = data["current"]["temperature_2m"]

            # --- Correct Way: Using UtteranceHandle with await ---
            handle1 = self.session.say(f"The current temperature is {temperature}°C.")
            await handle1  # Ensures this TTS finishes before starting next speech

            handle2 = self.session.say("Do you live in this city?")
            await handle2  # Ensures sequential speech

            # --- Incorrect Way: Using create_task (commented out) ---
            # These would start concurrently and cause overlapping speech:
            # asyncio.create_task(self.session.say(f"The current temperature is {temperature}°C."))
            # asyncio.create_task(self.session.say("Do you live in this city?"))

            # Check if user interrupted mid-speech
            if utterance and utterance.interrupted:
                return {"response": "Weather request cancelled due to user interruption."}

            return {"response": f"The temperature is {temperature}°C."}

        except asyncio.CancelledError:
            return {"response": "Weather task cancelled."}
        except Exception as e:
            logger.error(f"Error fetching weather: {e}")
            return {"response": "Sorry, I couldn't fetch the weather right now."}

    @function_tool
    async def long_running_task(self) -> str:
        """Simulates a long-running task that can be interrupted."""
        logger.info("Starting a long-running task...")
        utterance: UtteranceHandle | None = self.session.current_utterance

        try:
            for i in range(10):
                if utterance and utterance.interrupted:
                    logger.info("Long-running task was interrupted by the user.")
                    return "The task was cancelled because you interrupted me."

                logger.info(f"Long-running task progress: {i+1}/10")
                await asyncio.sleep(1)

            return "The long task is finally complete."
        except asyncio.CancelledError:
            logger.info("Long-running task was cancelled.")
            return "The task was cancelled."

async def entrypoint(ctx: JobContext):
    agent = VoiceAgent()
    conversation_flow = ConversationFlow(agent)

    pipeline = CascadingPipeline(
        stt=DeepgramSTT(),
        llm=OpenAILLM(),
        tts=ElevenLabsTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector()
    )

    session = AgentSession(
        agent=agent,
        pipeline=pipeline,
        conversation_flow=conversation_flow
    )

    await ctx.run_until_shutdown(session=session, wait_for_participant=True)


def make_context() -> JobContext:
    room_options = RoomOptions(name="Sandbox Agent", playground=True)
    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context())
    job.start()