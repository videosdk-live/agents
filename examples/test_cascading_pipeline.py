# This test script is used to test cascading pipeline.
import asyncio
import os
from typing import AsyncIterator, Optional
from videosdk import PubSubPublishConfig, PubSubSubscribeConfig
from videosdk.agents import Agent, AgentSession, CascadingPipeline, function_tool, WorkerJob, MCPServerStdio, MCPServerHTTP, ConversationFlow, ChatRole, JobContext, RoomOptions
from videosdk.plugins.openai import OpenAILLM, OpenAISTT, OpenAITTS
from videosdk.plugins.google import GoogleTTS,GoogleVoiceConfig,GoogleLLM, GoogleSTT
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
from videosdk.plugins.elevenlabs import ElevenLabsTTS
from videosdk.plugins.sarvamai import SarvamAITTS, SarvamAILLM,SarvamAISTT
from videosdk.plugins.cartesia import CartesiaTTS, CartesiaSTT
from videosdk.plugins.smallestai import SmallestAITTS
from videosdk.plugins.resemble import ResembleTTS
from videosdk.plugins.inworldai import InworldAITTS
from videosdk.plugins.lmnt import LMNTTTS
from videosdk.plugins.cerebras import CerebrasLLM
from videosdk.plugins.aws import AWSPollyTTS
from videosdk.plugins.neuphonic import NeuphonicTTS
from videosdk.plugins.anthropic import AnthropicLLM
from videosdk.plugins.humeai import HumeAITTS
from videosdk.plugins.rime import RimeTTS
from videosdk.plugins.speechify import SpeechifyTTS
from videosdk.plugins.groq import GroqTTS
from videosdk.plugins.navana import NavanaSTT
from videosdk.plugins.papla import PaplaTTS
from videosdk.plugins.assemblyai import AssemblyAISTT

import logging
import pathlib
import sys
import aiohttp

logging.getLogger().setLevel(logging.CRITICAL)

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
    def __init__(self, ctx: Optional[JobContext] = None):
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
                    executable_path=sys.executable,
                    process_arguments=[str(mcp_server_path)],
                    session_timeout=30
                ),
                MCPServerStdio(
                    executable_path=sys.executable,
                    process_arguments=[str(mcp_current_time_path)],
                    session_timeout=30
                ),
                MCPServerHTTP(
                    endpoint_url="YOUR_ZAPIER_MCP_SERVER_URL",
                    session_timeout=30
                )
            ]
        )
        self.ctx = ctx
        
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
        
    @function_tool
    async def send_pubsub_message(self, message: str):
        """Send a message to the pubsub topic CHAT_MESSAGE"""
        publish_config = PubSubPublishConfig(
            topic="CHAT_MESSAGE",
            message=message
        )
        await self.ctx.room.publish_to_pubsub(publish_config)
        return "Message sent to pubsub topic CHAT_MESSAGE"
    
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

def on_pubsub_message(message):
    print("Pubsub message received:", message)


async def entrypoint(ctx: JobContext):
    
    agent = MyVoiceAgent(ctx)
    conversation_flow = MyConversationFlow(agent)

    pipeline = CascadingPipeline(
        # STT Based Providers 
        stt= DeepgramSTT(api_key=os.getenv("DEEPGRAM_API_KEY")),
        # stt=CartesiaSTT(api_key=os.getenv("CARTESIA_API_KEY")),
        # stt=AssemblyAISTT(api_key=os.getenv("ASSEMBLYAI_API_KEY")),
        # stt=NavanaSTT(api_key=os.getenv("NAVANA_API_KEY"), customer_id=os.getenv("NAVANA_CUSTOMER_ID")),
       
        # OpenAI - All Three 
        # stt= OpenAISTT(api_key=os.getenv("OPENAI_API_KEY")),
        # llm=OpenAILLM(api_key=os.getenv("OPENAI_API_KEY")),
        # tts=OpenAITTS(api_key=os.getenv("OPENAI_API_KEY")),

        # Google - All Three 
        # stt = GoogleSTT( model="latest_long"),
        # llm=GoogleLLM(api_key=os.getenv("GOOGLE_API_KEY")),
        # tts=GoogleTTS(api_key=os.getenv("GOOGLE_API_KEY")),
        
        # SarvamAI - All Three 
        # stt=SarvamAISTT(api_key=os.getenv("SARVAMAI_API_KEY")),
        # llm=SarvamAILLM(api_key=os.getenv("SARVAMAI_API_KEY")),
        # tts=SarvamAITTS(api_key=os.getenv("SARVAMAI_API_KEY")),

        # LLM Based Providers 
        # llm=CerebrasLLM(api_key=os.getenv("CEREBRAS_API_KEY")),
        llm=AnthropicLLM(api_key=os.getenv("ANTHROPIC_API_KEY")),

        # TTS Based Providers 
        # tts=ElevenLabsTTS(api_key=os.getenv("ELEVENLABS_API_KEY")),
        # tts=CartesiaTTS(api_key=os.getenv("CARTESIA_API_KEY")),
        # tts=SmallestAITTS(api_key=os.getenv("SMALLESTAI_API_KEY")),
        # tts=ResembleTTS(api_key=os.getenv("RESEMBLE_API_KEY")),
        # tts=AWSPollyTTS(api_key=os.getenv("AWS_API_KEY")),
        # tts=NeuphonicTTS(api_key=os.getenv("NEUPHONIC_API_KEY")),
        # tts=InworldAITTS(api_key=os.getenv("INWORLD_API_KEY")),
        # tts=LMNTTTS(api_key=os.getenv("LMNT_API_KEY")),
        # tts=HumeAITTS(api_key=os.getenv("HUMEAI_API_KEY")),
        # tts=RimeTTS(api_key=os.getenv("RIME_API_KEY")),
        tts=SpeechifyTTS(api_key=os.getenv("SPEECHIFY_API_KEY")),
        # tts=GroqTTS(api_key=os.getenv("GROQ_API_KEY")),
        # tts=PaplaTTS(api_key=os.getenv("PAPLA_API_KEY")),

        vad=SileroVAD(),
        turn_detector=TurnDetector(threshold=0.8)
    )
    session = AgentSession(
        agent=agent, 
        pipeline=pipeline,
        conversation_flow=conversation_flow,
    )
    
    async def cleanup_session():
        print("Cleaning up session...")
    
    ctx.add_shutdown_callback(cleanup_session)

    try:
        await ctx.connect()
        print("Waiting for participant...")
        await ctx.room.wait_for_participant()
        print("Participant joined")
        await session.start()
        print("Connection established. Press Ctrl+C to exit.")
        
        subscribe_config = PubSubSubscribeConfig(
            topic="CHAT",
            cb=on_pubsub_message
        )
        await ctx.room.subscribe_to_pubsub(subscribe_config)
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    finally:
        await session.close()
        await ctx.shutdown()

def make_context() -> JobContext:
    room_options = RoomOptions(room_id="<meeting_id>", name="Sandbox Agent", playground=True)
    
    return JobContext(
        room_options=room_options
        )


if __name__ == "__main__":

    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()
