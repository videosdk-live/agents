# This test script is used to test realtime pipeline.

import asyncio
import os
import pathlib
import sys
import logging
import aiohttp
from typing import Optional
from videosdk import PubSubPublishConfig, PubSubSubscribeConfig
from videosdk.agents import Agent, AgentSession, RealTimePipeline, function_tool, MCPServerStdio, MCPServerHTTP, WorkerJob, JobContext, RoomOptions
from videosdk.plugins.aws import NovaSonicRealtime, NovaSonicConfig
from videosdk.plugins.google import GeminiRealtime, GeminiLiveConfig
from videosdk.plugins.openai import OpenAIRealtime, OpenAIRealtimeConfig
from openai.types.beta.realtime.session import  TurnDetection

logging.getLogger().setLevel(logging.CRITICAL)


@function_tool
async def get_weather(latitude: str, longitude: str):
    """Called when the user asks about the weather.
    do not ask user for latitude and longitude, estimate it.

    Args:
        latitude: The latitude of the location
        longitude: The longitude of the location
    """
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
                    url="YOUR_ZAPIER_MCP_SERVER_URL",
                    client_session_timeout_seconds=30
                )
            ]
        )
        self.ctx = ctx

    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you today?")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")
    
    @function_tool
    async def send_pubsub_message(self, message: str):
        """Send a message to the pubsub topic CHAT_MESSAGE"""
        publish_config = PubSubPublishConfig(
            topic="CHAT_MESSAGE",
            message=message
        )
        await self.ctx.room.publish_to_pubsub(publish_config)
        return "Message sent to pubsub topic CHAT_MESSAGE"

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

    # @function_tool
    # async def end_call(self) -> None:
    #     await self.session.say("Goodbye!")
    #     await asyncio.sleep(1)
    #     await self.session.leave()
        
def on_pubsub_message(message):
    print("Pubsub message received:", message)


async def entrypoint(ctx: JobContext):
    

    # model = OpenAIRealtime(
    #     model="gpt-4o-realtime-preview",
    #     config=OpenAIRealtimeConfig(
    #         voice="alloy", # alloy, ash, ballad, coral, echo, fable, onyx, nova, sage, shimmer, and verse
    #         modalities=["text", "audio"],
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
            voice="Leda", # Puck, Charon, Kore, Fenrir, Aoede, Leda, Orus, and Zephyr.
            response_modalities=["AUDIO"]
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
    agent = MyVoiceAgent(ctx)

    session = AgentSession(
        agent=agent,
        pipeline=pipeline,
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
        print("Voice session started. Awaiting interaction...")
        subscribe_config = PubSubSubscribeConfig(
            topic="CHAT_MESSAGE",
            cb=on_pubsub_message
        )
        await ctx.room.subscribe_to_pubsub(subscribe_config)
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("Shutting down...")
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
