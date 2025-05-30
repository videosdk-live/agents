import asyncio
import logging
import os
import pathlib
import sys

import aiohttp
from videosdk.agents import Agent, AgentSession, RealTimePipeline, function_tool
from videosdk.agents.mcp_integration import MCPToolManager
from videosdk.agents.mcp_server import MCPServerStdio,MCPServerHTTP
from videosdk.plugins.aws import NovaSonicRealtime, NovaSonicConfig
from videosdk.plugins.google import GeminiRealtime, GeminiLiveConfig
from videosdk.plugins.openai import OpenAIRealtime, OpenAIRealtimeConfig
from openai.types.beta.realtime.session import  TurnDetection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@function_tool
async def get_weather(latitude: str, longitude: str):
    """Called when the user asks about the weather.
    do not ask user for latitude and longitude, estimate it.

    Args:
        latitude: The latitude of the location
        longitude: The longitude of the location
    """
    logger.info(f"Getting weather for {latitude}, {longitude}")
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return {
                    "temperature": data["current"]["temperature_2m"],
                    "temperature_unit": "Celsius",
                }
            else:
                raise Exception(f"Failed to get weather data, status code: {response.status}")


class MyVoiceAgent(Agent):
    def __init__(self):
        current_dir = pathlib.Path(__file__).parent
        possible_paths = [
            current_dir / "examples" / "mcp_server_example.py",
            current_dir.parent / "examples" / "mcp_server_example.py",
            current_dir / "mcp_server_example.py"
        ]

        spath = [
            current_dir / "examples" / "mcp_current_time_example.py",
            current_dir.parent / "examples" / "mcp_current_time_example.py",
            current_dir / "mcp_current_time_example.py"
        ]


        mcp_server_path = next((p for p in possible_paths if p.exists()), None)
        mcp_current_time_path = next((p for p in spath if p.exists()), None)

        if not mcp_server_path:
            logger.error("MCP server example not found. Checked paths:")
            for path in possible_paths:
                logger.error(f" - {path}")
            raise Exception("MCP server example not found")
        
        if not mcp_current_time_path:
            logger.error("MCP current time example not found. Checked paths:")
            for path in spath:
                logger.error(f" - {path}")
            raise Exception("MCP current time example not found")

        logger.info(f"Connecting to MCP server at {mcp_server_path}")
        super().__init__(
            instructions=""" You are a helpful voice assistant that can answer questions and help with tasks. """,
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

    @function_tool
    async def get_horoscope(self, sign: str) -> dict:
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
        await self.session.say("Goodbye!")
        await asyncio.sleep(1)
        await self.session.leave()


async def main(context: dict):
    logger.info("Starting voice agent with MCP support...")
    

    model = OpenAIRealtime(
        model="gpt-4o-realtime-preview",
        config=OpenAIRealtimeConfig(
            voice="alloy", # alloy, ash, ballad, coral, echo, fable, onyx, nova, sage, shimmer, and verse
            modalities=["text", "audio"],
            turn_detection=TurnDetection(
                type="server_vad",
                threshold=0.5,
                prefix_padding_ms=300,
                silence_duration_ms=200,
            ),
            tool_choice="auto"
        )
    )

    # model = GeminiRealtime(
    #     model="gemini-2.0-flash-live-001",
    #     config=GeminiLiveConfig(
    #         voice="Leda", # Puck, Charon, Kore, Fenrir, Aoede, Leda, Orus, and Zephyr.
    #         response_modalities=["AUDIO"]
    #     )
    # )

    # model = NovaSonicRealtime(
    #     model="amazon.nova-sonic-v1:0",
    #     config=NovaSonicConfig(
    #         voice="tiffany",
    #         temperature=0.7,
    #         top_p=0.9,
    #         max_tokens=1024
    #     )
    # )

    pipeline = RealTimePipeline(model=model)
    agent = MyVoiceAgent()

    session = AgentSession(
        agent=agent,
        pipeline=pipeline,
        context=context
    )

    try:
        await session.start()
        logger.info("Voice session started. Awaiting interaction...")
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await session.close()
        await pipeline.cleanup()


if __name__ == "__main__":
    def make_context():
        return {"meetingId": "obsk-dfh0-qmyb", "name": "VideoSDK Voice Agent"}

    asyncio.run(main(context=make_context()))
