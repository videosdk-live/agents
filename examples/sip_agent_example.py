import asyncio
import os
import sys
import pathlib
import logging
import aiohttp
from contextlib import asynccontextmanager
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
import uvicorn
from pyngrok import ngrok
from videosdk.plugins.agentsip import create_sip_manager
from videosdk.agents import Agent, JobContext, function_tool, RealTimePipeline, MCPServerStdio
from videosdk.plugins.google import GeminiRealtime, GeminiLiveConfig
from videosdk import PubSubPublishConfig

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

@function_tool
async def get_weather(latitude: str, longitude: str):
    """Called when the user asks about the weather. Do not ask for latitude/longitude, estimate it."""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}¤t=temperature_2m"
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

def create_agent_pipeline():
    """Factory to create the specific pipeline for our agent."""
    # This is where all your customization goes!
    # Example using GeminiRealtime
    model = GeminiRealtime(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-2.0-flash-live-001",
        config=GeminiLiveConfig(
            voice="Leda",
            response_modalities=["AUDIO"],
        ),
    )
    return RealTimePipeline(model=model)

class SIPAIAgent(Agent):
    """A production-ready AI agent for handling voice calls."""

    def __init__(self, ctx: Optional[JobContext] = None):
        # Example of adding an MCP Server to the agent
        current_dir = pathlib.Path(__file__).parent
        # IMPORTANT: Make sure these paths are correct for your project structure
        mcp_server_path = current_dir / "mcp_server_examples" / "mcp_server_example.py"
        if not mcp_server_path.exists():
            # Handle case where MCP server example is not found
            # For now, we'll proceed without it, but you might want to raise an error
            logger.warning(f"MCP server example not found at: {mcp_server_path}. Agent will start without it.")
            mcp_servers = []
        else:
             mcp_servers=[
                MCPServerStdio(
                    command=sys.executable,
                    args=[str(mcp_server_path)],
                    client_session_timeout_seconds=30
                )
             ]

        super().__init__(
            instructions="You are a helpful voice assistant that can answer questions and help with tasks. Be friendly and concise.",
            tools=[get_weather, self.send_pubsub_message, self.end_call],
            mcp_servers=mcp_servers
        )
        self.ctx = ctx
        logger.info(f"🤖 SIPAIAgent created.")

    async def on_enter(self) -> None:
        logger.info("🎤 Agent on_enter() called - saying greeting...")
        await self.session.say("Hello! How can I help you today?")

    async def on_exit(self) -> None:
        logger.info("👋 Agent on_exit() called")

    @function_tool
    async def send_pubsub_message(self, message: str):
        """Send a message to the pubsub topic CHAT_MESSAGE"""
        if self.ctx and self.ctx.room:
            await self.ctx.room.publish_to_pubsub(
                PubSubPublishConfig(topic="CHAT_MESSAGE", message=message)
            )
            return f"Message sent to pubsub topic CHAT_MESSAGE: {message}"
        return "Cannot send pubsub message - no context available"

    @function_tool
    async def end_call(self) -> str:
        """End the current call gracefully"""
        await self.session.say("Thank you for calling. Have a great day!")
        await asyncio.sleep(1)
        await self.session.leave()
        return "Call ended gracefully"


sip_manager = create_sip_manager(
    provider=os.getenv("SIP_PROVIDER", "twilio"),
    videosdk_token=os.getenv("VIDEOSDK_AUTH_TOKEN"),
    provider_config={
        # Twilio config
        "account_sid": os.getenv("TWILIO_ACCOUNT_SID"),
        "auth_token": os.getenv("TWILIO_AUTH_TOKEN"),
        "phone_number": os.getenv("TWILIO_PHONE_NUMBER"),
    }
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for FastAPI app startup and shutdown."""
    port = int(os.getenv("PORT", 8000))
    try:
        ngrok.kill()
        ngrok_auth_token = os.getenv("NGROK_AUTHTOKEN")
        if ngrok_auth_token:
            ngrok.set_auth_token(ngrok_auth_token)
        tunnel = ngrok.connect(port, "http")
        sip_manager.set_base_url(tunnel.public_url)
        logger.info(f"🌐 Ngrok tunnel created: {tunnel.public_url}")
    except Exception as e:
        logger.error(f"❌ Failed to start ngrok tunnel: {e}")
    yield
    try:
        ngrok.kill()
        logger.info("🔚 Ngrok tunnel closed")
    except Exception as e:
        logger.error(f"❌ Error closing ngrok tunnel: {e}")

app = FastAPI(title="Production SIP AI Agent (Using Plugin)", lifespan=lifespan)

@app.post("/call/make")
async def make_call(to_number: str):
    if not sip_manager.base_url:
        return {"status": "error", "message": "Service not ready (no base URL)."}
    agent_config = {"room_name": "Production Call", "enable_pubsub": True}
    details = await sip_manager.make_call(
        to_number=to_number,
        agent_class=SIPAIAgent,
        pipeline_factory=create_agent_pipeline,
        agent_config=agent_config
    )
    return {"status": "success", "details": details}

@app.post("/sip/answer/{room_id}")
async def answer_webhook(room_id: str):
    logger.info(f"🔗 Answering call for room: {room_id}")
    body, status_code, headers = sip_manager.get_sip_response_for_room(room_id)
    return Response(content=body, status_code=status_code, media_type=headers.get("Content-Type"))

@app.post("/webhook/incoming")
async def incoming_webhook(request: Request):
    try:
        content_type = request.headers.get("Content-Type", "")
        if "x-www-form-urlencoded" in content_type:
            webhook_data = dict(await request.form())
        else:
            webhook_data = await request.json()
        logger.info(f"Received incoming webhook: {webhook_data}")

        agent_config = {"room_name": "Incoming Call", "enable_pubsub": True}
        body, status_code, headers = await sip_manager.handle_incoming_call(
            webhook_data=webhook_data,
            agent_class=SIPAIAgent,
            pipeline_factory=create_agent_pipeline,
            agent_config=agent_config
        )
        return Response(content=body, status_code=status_code, media_type=headers.get("Content-Type"))
    except Exception as e:
        logger.error(f"Error in incoming webhook: {e}", exc_info=True)
        return Response(content="Error processing request", status_code=500)

@app.get("/sessions")
async def get_sessions():
    return {"sessions": sip_manager.get_active_sessions()}

@app.get("/")
async def root():
    return {"message": "Production SIP AI Agent (Using videosdk-plugins-agentsip)"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"🚀 Starting Production SIP AI Agent (Plugin Version) on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port) 