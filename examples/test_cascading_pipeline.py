# This test script is used to test cascading pipeline.
import asyncio
import os
from typing import AsyncIterator
from videosdk.plugins.openai import OpenAIRealtime, OpenAIRealtimeConfig, OpenAILLM, OpenAISTT, OpenAITTS
from videosdk.plugins.google import GeminiRealtime, GeminiLiveConfig, GoogleTTS,GoogleVoiceConfig,GoogleLLM, GoogleSTT
# from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.silero import SileroVAD
from videosdk.agents import Agent, AgentSession, CascadingPipeline, function_tool, WorkerJob, MCPServerStdio, MCPServerHTTP, ConversationFlow, ChatRole
from google.genai.types import AudioTranscriptionConfig
import aiohttp
import logging
from openai.types.beta.realtime.session import InputAudioTranscription, TurnDetection
import pathlib
import sys
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
# from videosdk.plugins.elevenlabs import ElevenLabsTTS
# from videosdk.plugins.sarvamai import SarvamAITTS, SarvamAILLM,SarvamAISTT

logger = logging.getLogger(__name__)

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
        current_dir = pathlib.Path(__file__).parent
        mcp_server_path = current_dir / "mcp_server_examples" / "mcp_server_example.py"
        mcp_current_time_path = current_dir / "mcp_server_examples" / "mcp_current_time_example.py"

        if not mcp_server_path.exists():
            print(f"MCP server example not found at: {mcp_server_path}")
            raise Exception("MCP server example not found")
        
        if not mcp_current_time_path.exists():
            print(f"MCP current time example not found at: {mcp_current_time_path}")
            raise Exception("MCP current time example not found")
        super().__init__(
            instructions="You are a helpful voice assistant that can answer questions and help with tasks and help with horoscopes and weather.",
            tools=[get_weather],
            mcp_servers=[
                MCPServerStdio(
                    command=sys.executable,
                    args=[str(mcp_server_path)],
                    client_session_timeout_seconds=30
                ),
                MCPServerStdio(
                    command=sys.executable,
                    args=[str(mcp_current_time_path)],
                    client_session_timeout_seconds=30
                ),
                MCPServerHTTP(
                    url="https://mcp.zapier.com/api/mcp/s/ODk5ODA5OTctMDM2Ny00ZDEyLTk2NjctNDQ4NDE3MDI5MjA3OjE3MzQ5NjE3LTg0MjQtNDJhZC1iOWJkLTE2OTBmMmRkYzI0ZQ==/mcp",
                    client_session_timeout_seconds=30
                )
            ]
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you today?")
    
    async def on_exit(self) -> None:
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
    
    # @function_tool
    # async def end_call(self) -> None:
    #     """End the call upon request by the user"""
    #     await self.session.say("Goodbye!")
    #     await asyncio.sleep(1)
    #     await self.session.leave()
    
class MyConversationFlow(ConversationFlow):
    def __init__(self, agent, stt=None, llm=None, tts=None):
        super().__init__(agent, stt, llm, tts)

    async def run(self, transcript: str) -> AsyncIterator[str]:
        """Main conversation loop: handle a user turn."""
        await self.on_turn_start(transcript)

        processed_transcript = transcript.lower().strip()
        self.agent.chat_context.add_message(role=ChatRole.USER, content=processed_transcript)
        
        async for response_chunk in self.process_with_llm():
            yield response_chunk

        await self.on_turn_end()

    async def on_turn_start(self, transcript: str) -> None:
        """Called at the start of a user turn."""
        self.is_turn_active = True

    async def on_turn_end(self) -> None:
        """Called at the end of a user turn."""
        self.is_turn_active = False


async def test_connection(jobctx):
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
    
    # model = GeminiRealtime(
    #     model="gemini-2.0-flash-live-001",
    #     config=GeminiLiveConfig(
    #         response_modalities=["AUDIO"],
    #         output_audio_transcription=AudioTranscriptionConfig(
    #         )
    #     )
    # )
    # pipeline = RealTimePipeline(model=model)
        #     stt = OpenAISTT(
        # api_key=os.getenv("OPENAI_API_KEY"),
        # model="whisper-1",
        # language="en",
        # turn_detection={
        #     "type": "server_vad",
        #     "threshold": 0.5,
        #     "prefix_padding_ms": 600,
        #     "silence_duration_ms": 350,
        # }
        # ),
    agent = MyVoiceAgent()
    conversation_flow = MyConversationFlow(agent)
    pipeline = CascadingPipeline(
        # stt= DeepgramSTT(api_key=os.getenv("DEEPGRAM_API_KEY")),
        stt= OpenAISTT(api_key=os.getenv("OPENAI_API_KEY")),
        llm=OpenAILLM(api_key=os.getenv("OPENAI_API_KEY")),
        tts=OpenAITTS(api_key=os.getenv("OPENAI_API_KEY")),
        # tts=ElevenLabsTTS(api_key=os.getenv("ELEVENLABS_API_KEY")),
        # stt = GoogleSTT( model="latest_long"),
        # llm=GoogleLLM(api_key=os.getenv("GOOGLE_API_KEY")),
        # tts=GoogleTTS(api_key=os.getenv("GOOGLE_API_KEY")),
        # stt=SarvamAISTT(api_key=os.getenv("SARVAMAI_API_KEY")),
        # llm=SarvamAILLM(api_key=os.getenv("SARVAMAI_API_KEY")),
        # tts=SarvamAITTS(api_key=os.getenv("SARVAMAI_API_KEY")),
        vad=SileroVAD(),
        turn_detector=TurnDetector(threshold=0.8)
    )
    session = AgentSession(
        agent=agent, 
        pipeline=pipeline,
        conversation_flow=conversation_flow,
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
        return {"meetingId": "<meeting_id>", "name": "Sandbox Agent", "playground": True}

    job = WorkerJob(job_func=entryPoint, jobctx=make_context)
    job.start()
