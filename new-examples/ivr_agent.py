"""
Use Case: Bank automated phone IVR with DTMF menu navigation and voicemail detection (Metro Bank).
Pipeline: P1 — DeepgramSTT + GoogleLLM + CartesiaTTS + SileroVAD + TurnDetector
Demonstrates: DTMFHandler for keypress routing, VoiceMailDetector for auto-hangup, non-interruptible menus.
Env Vars: VIDEOSDK_AUTH_TOKEN, DEEPGRAM_API_KEY, GOOGLE_API_KEY, CARTESIA_API_KEY, OPENAI_API_KEY
"""

import logging
from videosdk.agents import (
    Agent, AgentSession, Pipeline, WorkerJob, JobContext, RoomOptions, Options,
    DTMFHandler, VoiceMailDetector,
)
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.google import GoogleLLM
from videosdk.plugins.cartesia import CartesiaTTS
from videosdk.plugins.openai import OpenAILLM
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
pre_download_model()


class MetroBankIVRAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are the automated phone system for Metro Bank.
            You handle keypad DTMF input and voice commands:
            - Press or say 1: Account balance
            - Press or say 2: Recent transactions
            - Press or say 3: Transfer funds
            - Press or say 0 or say 'agent': Connect to a live representative
            For voice commands, match the intent and provide the requested information.
            Menu announcements are non-interruptible — always complete them.
            Be concise — this is a phone system, not a conversation.""",
        )

    async def on_enter(self) -> None:
        # Non-interruptible menu announcement — critical for IVR usability.
        await self.session.say(
            "Welcome to Metro Bank. "
            "For account balance, press 1. "
            "For recent transactions, press 2. "
            "For fund transfers, press 3. "
            "To speak with an agent, press 0.",
            interruptible=False,
        )

    async def on_exit(self) -> None:
        await self.session.say("Thank you for banking with Metro Bank. Goodbye.")


async def entrypoint(ctx: JobContext):
    agent = MetroBankIVRAgent()

    pipeline = Pipeline(
        stt=DeepgramSTT(),
        llm=GoogleLLM(),
        tts=CartesiaTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
    )

    async def on_dtmf_press(message):
        """Route DTMF keypress to the appropriate IVR branch."""
        key = str(message).strip() if not isinstance(message, dict) else message.get("key", "")
        logging.info("DTMF key pressed: %s", key)
        dtmf_responses = {
            "1": "Checking your account balance. Your current balance is $4,250.00.",
            "2": "Your last three transactions: $35.00 at Starbucks, $120.00 at Amazon, $200.00 at Whole Foods.",
            "3": "To transfer funds, please use the Metro Bank mobile app or visit a branch.",
            "0": "Connecting you to a live representative. Please hold.",
        }
        response = dtmf_responses.get(key)
        if response:
            await agent.session.say(response, interruptible=False)
        else:
            await agent.session.say("Invalid option. " + agent._get_menu(), interruptible=False)

    async def on_voicemail_detected():
        """Auto-hang up if voicemail is detected — do not leave a message."""
        logging.info("Voicemail detected — disconnecting.")
        await agent.hangup()

    dtmf_handler = DTMFHandler(on_dtmf_press)
    voice_mail_detector = VoiceMailDetector(
        llm=OpenAILLM(),
        duration=5.0,
        callback=on_voicemail_detected,
    )

    session = AgentSession(
        agent=agent,
        pipeline=pipeline,
        dtmf_handler=dtmf_handler,
        voice_mail_detector=voice_mail_detector,
    )
    await session.start(wait_for_participant=True, run_until_shutdown=True)


def make_context() -> JobContext:
    room_options = RoomOptions(name="Metro Bank IVR", playground=True)
    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(
        entrypoint=entrypoint,
        jobctx=make_context,
        options=Options(agent_id="YOUR_AGENT_ID", max_processes=2, register=True, host="localhost", port=8081),
    )
    job.start()
