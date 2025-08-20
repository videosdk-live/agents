import asyncio
import os
import pathlib
import sys
import logging
from typing import Optional
from videosdk.agents import Agent, AgentSession, RealTimePipeline,MCPServerStdio, MCPServerHTTP, WorkerJob, JobContext, RoomOptions
from videosdk.plugins.openai import OpenAIRealtime, OpenAIRealtimeConfig
from openai.types.beta.realtime.session import  TurnDetection

logging.getLogger().setLevel(logging.CRITICAL)

class MCPAgent(Agent):
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
        

    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you today?")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")


async def entrypoint(ctx: JobContext):
    
    model = OpenAIRealtime(
        model="gpt-4o-realtime-preview",
        config=OpenAIRealtimeConfig(
            voice="alloy",
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

    pipeline = RealTimePipeline(model=model)
    
    agent = MCPAgent(ctx)

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
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        await session.close()
        await ctx.shutdown()

def make_context() -> JobContext:
    room_options = RoomOptions(room_id="<room_id>", name="Sandbox Agent", playground=True)
    
    return JobContext(
        room_options=room_options
        )

if __name__ == "__main__":

    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()