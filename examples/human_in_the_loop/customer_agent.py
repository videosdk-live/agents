import asyncio
import os
from typing import Optional
from videosdk.agents import Agent, AgentSession, CascadingPipeline, WorkerJob, MCPServerStdio, ConversationFlow, JobContext, RoomOptions
from videosdk.plugins.google import GoogleTTS
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
from videosdk.plugins.anthropic import AnthropicLLM

import logging
import pathlib
import sys
import aiohttp

logging.getLogger().setLevel(logging.CRITICAL)

pre_download_model()

class CustomerAgent(Agent):
    def __init__(self, ctx: Optional[JobContext] = None):
        current_dir = pathlib.Path(__file__).parent
        discord_mcp_server_path = current_dir / "discord_mcp_server.py"

        if not discord_mcp_server_path.exists():
            print(f"Discord MCP server not found at: {discord_mcp_server_path}")
            raise Exception("MCP server example not found")
        super().__init__(
            instructions="You are a customer-facing agent for VideoSDK. You have access to various tools to assist with customer inquiries, provide support, and handle tasks. When a user asks for a discount percentage, always use the appropriate tool to retrieve and provide the accurate answer from your superior human agent.",
            mcp_servers=[
                MCPServerStdio(
                    executable_path=sys.executable,
                    process_arguments=[str(discord_mcp_server_path)],
                    session_timeout=30
                ),
            ]
        )
        self.ctx = ctx
        
    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you today?")
    
    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")
        
async def entrypoint(ctx: JobContext):
    
    agent = CustomerAgent(ctx)
    conversation_flow = ConversationFlow(agent)

    pipeline = CascadingPipeline(
        stt= DeepgramSTT(api_key=os.getenv("DEEPGRAM_API_KEY")),  
        llm=AnthropicLLM(api_key=os.getenv("ANTHROPIC_API_KEY")),
        tts=GoogleTTS(api_key=os.getenv("GOOGLE_API_KEY")),
        vad=SileroVAD(),
        turn_detector=TurnDetector(threshold=0.8)
    )
    
    session = AgentSession(
        agent=agent, 
        pipeline=pipeline,
        conversation_flow=conversation_flow,
    )

    await ctx.run_until_shutdown(session=session,wait_for_participant=True)

def make_context() -> JobContext:
    room_options = RoomOptions(room_id="<room_id>", name="Customer Agent", playground=True)
    
    return JobContext(
        room_options=room_options
        )


if __name__ == "__main__":

    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()
