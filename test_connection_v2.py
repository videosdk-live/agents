
import asyncio
import os
import pathlib
import sys
from videosdk.plugins.openai import OpenAIRealtime, OpenAIRealtimeConfig
from videosdk.plugins.google import GeminiRealtime, GeminiLiveConfig
from videosdk.agents import Agent, AgentSession, ConversationFlow, RealTimePipeline, function_tool
from videosdk.agents.mcp_integration import MCPToolManager
from videosdk.agents.mcp_server import MCPServerStdio
from videosdk.plugins.aws import NovaSonicRealtime, NovaSonicConfig
from videosdk.agents import Agent, AgentSession, RealTimePipeline, function_tool, WorkerJob
from google.genai.types import AudioTranscriptionConfig
import aiohttp
import logging
from openai.types.beta.realtime.session import InputAudioTranscription, TurnDetection
from videosdk.agents.job import WorkerJob

logger = logging.getLogger(__name__)


class MyVoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
            You are a helpful voice assistant that can answer questions and help with tasks.
            
            FINANCIAL TOOLS AVAILABLE:
            - get_current_time: Shows the current time (no parameters needed)
            - get_nifty50_price: Shows Nifty 50 index (no parameters needed)
            - get_stock_quote: Gets stock price - requires symbol parameter
            - get_exchange_rate: Gets currency exchange - requires from_currency and to_currency parameters
            - get_company_info: Gets company details - requires symbol parameter
            - search_with_time: Search with time context - requires query parameter
            
            For tools that require parameters, ALWAYS include them when calling the tool.
            Examples:
            - get_stock_quote(symbol="AAPL")
            - get_exchange_rate(from_currency="USD", to_currency="EUR")
            - get_company_info(symbol="MSFT")
            - search_with_time(query="latest stock market news")
            
            If a tool call fails, carefully check the error message and try again with
            the correct parameters. Make sure to include all required parameters.
            """,
        )
        self.mcp_manager = MCPToolManager()
        self.register_tools([self.get_weather, self.get_horoscope])
        
    async def initialize_mcp(self):
        """Initialize MCP tools"""
        # Try to find the MCP server example file in multiple paths
        current_dir = pathlib.Path(__file__).parent
        possible_paths = [
            current_dir / "examples" / "mcp_server_example.py",  # If in examples folder
            current_dir.parent / "examples" / "mcp_server_example.py",  # If in parent/examples folder
            current_dir / "mcp_server_example.py"  # If in same directory
        ]
        
        mcp_server_path = None
        for path in possible_paths:
            if path.exists():
                mcp_server_path = path
                break
                
        if not mcp_server_path:
            logger.error("MCP server example not found. Checked paths:")
            for path in possible_paths:
                logger.error(f" - {path}")
            return
            
        # Create and initialize MCP server connection
        logger.info(f"Connecting to MCP server at {mcp_server_path}")
        stdio_server = MCPServerStdio(
            command=sys.executable,  # Use the current Python executable
            args=[str(mcp_server_path)],
            client_session_timeout_seconds=30
        )
        
        try:
            # Add the server to the manager
            await self.mcp_manager.add_mcp_server(stdio_server)
            
            # Register MCP tools with the agent
            await self.mcp_manager.register_mcp_tools(self)
            
            logger.info("MCP tools initialized and registered")
        except Exception as e:
            logger.error(f"Error initializing MCP tools: {e}")
            # We don't want to fail the whole agent if MCP fails
            # Just log the error and continue without MCP tools

    async def on_enter(self) -> None:
        await self.session.say("Hi there! I can help you with financial data and more. Try asking about the weather, your horoscope, stock prices, or exchange rates.")

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
    # model = GeminiRealtime(
    #     model="gemini-2.0-flash-live-001",
    #     config=GeminiLiveConfig(
    #         response_modalities=["AUDIO"],
    #         output_audio_transcription=AudioTranscriptionConfig(
    #         )
    #     )
    # )

    model = NovaSonicRealtime(
            model="amazon.nova-sonic-v1:0",
            config=NovaSonicConfig(
                voice="tiffany",      
                temperature=0.7,      
                top_p=0.9,            
                max_tokens=1024       
            )
    )
    pipeline = RealTimePipeline(model=model)
    
    # Create agent and initialize MCP
    agent = MyVoiceAgent()
    await agent.initialize_mcp()
    
    session = AgentSession(
        agent=agent, 
        pipeline=pipeline,
        context=jobctx
    )

    try:
        await session.start()
        print("Connection established. Press Ctrl+C to exit.")
        print("\nExample queries to try:")
        print("- \"What time is it?\"")
        print("- \"What's the current Nifty 50 price?\"") 
        print("- \"What's the price of AAPL stock?\"")
        print("- \"What's the exchange rate from USD to EUR?\"")
        print("- \"Tell me about the company with symbol MSFT\"")
        print("- \"What's the weather in New York?\"")
        print("- \"What's my Taurus horoscope?\"")
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    finally:
        await pipeline.cleanup()


def entryPoint(jobctx):
    asyncio.run(test_connection(jobctx))


if __name__ == "__main__":

    def make_context():
        return {"pid": os.getpid(), "meetingId": "obsk-dfh0-qmyb", "name": "Agent"}

    job = WorkerJob(job_func=entryPoint, jobctx=make_context())
    job.start()