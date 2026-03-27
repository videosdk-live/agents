"""
Use Case: Voice-controlled Slack assistant powered by LangChain.
Pipeline: P1 — DeepgramSTT + LangChainLLM + CartesiaTTS + SileroVAD + TurnDetector

Demonstrates: LangChainLLM (Mode A) — @function_tool methods that talk to Slack's API.
The agent posts messages, — all by voice.

Env Vars: VIDEOSDK_AUTH_TOKEN, DEEPGRAM_API_KEY, OPENAI_API_KEY, CARTESIA_API_KEY, SLACK_BOT_TOKEN
Install:  pip install -r requirements.txt
"""

import logging
import os

from slack_sdk.web.async_client import AsyncWebClient

from videosdk.agents import Agent, AgentSession, Pipeline, WorkerJob, JobContext, RoomOptions, function_tool
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.cartesia import CartesiaTTS
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
from langchain_openai import ChatOpenAI
from videosdk.plugins.langchain import LangChainLLM

from dotenv import load_dotenv
load_dotenv(override=True)

pre_download_model()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])

_slack = AsyncWebClient(token=os.environ.get("SLACK_BOT_TOKEN", ""))

async def _resolve_channel_id(name: str) -> str | None:
    """Resolve a channel name to its Slack channel ID."""
    name = name.lstrip("#")
    if len(name) >= 9 and name[0].upper() in ("C", "G"):
        return name
    cursor = None
    while True:
        kwargs: dict = {"limit": 200, "types": "public_channel,private_channel"}
        if cursor:
            kwargs["cursor"] = cursor
        resp = await _slack.conversations_list(**kwargs)
        for ch in resp.get("channels", []):
            if ch.get("name", "").lower() == name.lower():
                return ch["id"]
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
    return None


async def _resolve_user_id(name: str) -> str | None:
    """Resolve a user's display name or real name to their Slack user ID."""
    name_lower = name.lower()
    cursor = None
    while True:
        kwargs: dict = {"limit": 200}
        if cursor:
            kwargs["cursor"] = cursor
        resp = await _slack.users_list(**kwargs)
        for member in resp.get("members", []):
            if member.get("deleted") or member.get("is_bot"):
                continue
            profile = member.get("profile", {})
            display = profile.get("display_name", "").lower()
            real = profile.get("real_name", "").lower()
            if name_lower in (display, real):
                return member["id"]
            if display and name_lower == display.split()[0]:
                return member["id"]
            if real and name_lower == real.split()[0]:
                return member["id"]
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
    return None

class SlackVoiceAgent(Agent):
    """Voice assistant that controls Slack: post messages, read channels, send DMs."""

    def __init__(self):
        super().__init__(
            instructions=(
                "You are Max, a voice-controlled Slack assistant. "
                "You can post messages to channels, read recent messages, and send direct messages. "
                "After executing any action, confirm it briefly. "
                "Keep all responses short and conversational — you are speaking on a call.\n\n"
                "Examples of what you can do:\n"
                "- 'Post to engineering: the deploy is done'\n"
                "- 'Read the last 5 messages in general'\n"
                "- 'Send a DM to Alex saying the PR is ready for review'"
            ),
        )

    async def on_enter(self) -> None:
        await self.session.say(
            "Hey! I'm Max, your Slack voice assistant. "
            "I can post to channels, At what channel you like to post the message?"
        )

    async def on_exit(self) -> None:
        await self.session.say("Signing off. Have a productive day!")


    @function_tool
    async def post_message(self, channel: str, message: str) -> str:
        """Post a message to a Slack channel.

        Args:
            channel: Channel name or ID (e.g. 'general', 'engineering', 'C01234ABCDE')
            message: The message text to post
        """
        channel_name = channel.lstrip("#")
        try:
            await _slack.chat_postMessage(channel=f"#{channel_name}", text=message)
            return f"Message posted to #{channel_name}."
        except Exception as exc:
            return f"Failed to post to #{channel_name}: {exc}"


async def entrypoint(ctx: JobContext):
    agent = SlackVoiceAgent()

    langchain_llm = LangChainLLM(
        llm=ChatOpenAI(model="gpt-4o-mini", streaming=True),
    )
    
    pipeline = Pipeline(
        stt=DeepgramSTT(),
        llm=langchain_llm,
        tts=CartesiaTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
    )
    session = AgentSession(agent=agent, pipeline=pipeline)
    await session.start(wait_for_participant=True, run_until_shutdown=True)


def make_context() -> JobContext:
    room_options = RoomOptions(room_id="<room_id>", name="Slack Assistant", playground=True)
    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()