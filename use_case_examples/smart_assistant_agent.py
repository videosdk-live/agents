"""
Use Case: Smart home assistant (Aria) — controls devices and runs automations via MCP tool servers.
Pipeline: P2 — OpenAIRealtime + MCPServerStdio (local tools) + MCPServerHTTP (Zapier automation)
Demonstrates: MCP server integration with a realtime pipeline; tools auto-discovered from MCP servers.
Env Vars: VIDEOSDK_AUTH_TOKEN, OPENAI_API_KEY, ZAPIER_MCP_URL (optional)
"""

import sys
import logging
import pathlib
from videosdk.agents import Agent, AgentSession, Pipeline, MCPServerStdio, MCPServerHTTP, WorkerJob, JobContext, RoomOptions
from videosdk.plugins.openai import OpenAIRealtime, OpenAIRealtimeConfig
from openai.types.beta.realtime.session import TurnDetection

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])


class SmartHomeAgent(Agent):
    def __init__(self):
        current_dir = pathlib.Path(__file__).parent
        mcp_server_path = current_dir / "mcp_server_examples" / "mcp_server_example.py"
        mcp_time_path = current_dir / "mcp_server_examples" / "mcp_current_time_example.py"

        mcp_servers = []

        # Local MCP server: device controls (lights, thermostat)
        if mcp_server_path.exists():
            mcp_servers.append(
                MCPServerStdio(
                    executable_path=sys.executable,
                    process_arguments=[str(mcp_server_path)],
                    session_timeout=30,
                )
            )

        # Local MCP server: current time (for scheduling reminders)
        if mcp_time_path.exists():
            mcp_servers.append(
                MCPServerStdio(
                    executable_path=sys.executable,
                    process_arguments=[str(mcp_time_path)],
                    session_timeout=30,
                )
            )

        # Remote MCP server: Zapier automation (optional)
        # Replace with your Zapier MCP URL to enable home automations via Zapier.
        # mcp_servers.append(
        #     MCPServerHTTP(endpoint_url="YOUR_ZAPIER_MCP_URL", session_timeout=30)
        # )

        super().__init__(
            instructions="""You are Aria, the smart home assistant.
            You control lights, thermostat, and music using the tools available to you.
            You can also check the current time and set reminders.
            Be brief — users are typically hands-free and want quick actions.
            Confirm each action after performing it (e.g., 'Done, lights are off.').
            For ambiguous commands, ask one clarifying question before acting.""",
            mcp_servers=mcp_servers,
        )

    async def on_enter(self) -> None:
        await self.session.say(
            "Hi, I'm Aria — your smart home assistant. "
            "I can control your lights, thermostat, and music. What would you like to do?"
        )

    async def on_exit(self) -> None:
        await self.session.say("Goodbye! Aria is standing by.")


async def entrypoint(ctx: JobContext):
    model = OpenAIRealtime(
        model="gpt-realtime-2025-08-28",
        config=OpenAIRealtimeConfig(
            voice="nova",
            modalities=["text", "audio"],
            turn_detection=TurnDetection(
                type="server_vad",
                threshold=0.5,
                prefix_padding_ms=300,
                silence_duration_ms=200,
            ),
            tool_choice="auto",
        ),
    )

    pipeline = Pipeline(llm=model)

    agent = SmartHomeAgent()
    session = AgentSession(agent=agent, pipeline=pipeline)
    await session.start(wait_for_participant=True, run_until_shutdown=True)


def make_context() -> JobContext:
    room_options = RoomOptions(room_id="<room_id>", name="Smart Home - Aria", playground=True)
    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()
