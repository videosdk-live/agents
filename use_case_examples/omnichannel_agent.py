"""
Use Case: Banking virtual assistant — accepts voice and text input in the same session (First National Bank).
Pipeline: P1 — DeepgramSTT + GoogleLLM + CartesiaTTS + SileroVAD + TurnDetector
Demonstrates: Combined voice + text input/output, session.interrupt() via pubsub command.
Env Vars: VIDEOSDK_AUTH_TOKEN, DEEPGRAM_API_KEY, GOOGLE_API_KEY, CARTESIA_API_KEY
"""

import asyncio
import logging
from videosdk import PubSubPublishConfig, PubSubSubscribeConfig
from videosdk.agents import Agent, AgentSession, Pipeline, function_tool, WorkerJob, JobContext, RoomOptions
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.google import GoogleLLM
from videosdk.plugins.cartesia import CartesiaTTS
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
pre_download_model()


class BankingAgent(Agent):
    def __init__(self, ctx: JobContext):
        super().__init__(
            instructions="""You are a virtual banker at First National Bank.
            Help customers check account balances, review recent transactions, and initiate transfers.
            Always ask for and verify the customer's account number before sharing any account details.
            Do not reveal full account numbers — only the last 4 digits.
            Be concise and professional — customers are often on the go.
            You can receive messages both via voice and via text chat.""",
        )
        self.ctx = ctx

    async def on_enter(self) -> None:
        await self.session.say(
            "Good day, welcome to First National Bank. I'm your virtual banker. "
            "You can speak to me or send a text message. "
            "May I have your account number to get started?"
        )

    async def on_exit(self) -> None:
        await self.session.say(
            "Thank you for banking with First National Bank. Have a wonderful day!"
        )

    @function_tool
    async def get_account_balance(self, account_last_four: str) -> dict:
        """Retrieve the current balance for a customer account.

        Args:
            account_last_four: Last 4 digits of the customer's account number
        """
        # In production, replace with a real banking API call.
        mock_balances = {
            "1234": {"checking": 4250.00, "savings": 12800.50, "currency": "USD"},
            "5678": {"checking": 1020.75, "savings": 5000.00, "currency": "USD"},
        }
        account = mock_balances.get(account_last_four)
        if not account:
            return {"found": False, "message": "Account not found. Please verify the account number."}
        return {"found": True, "account_ending": account_last_four, **account}


async def entrypoint(ctx: JobContext):
    agent = BankingAgent(ctx)

    pipeline = Pipeline(
        stt=DeepgramSTT(),
        llm=GoogleLLM(),
        tts=CartesiaTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
    )

    session = AgentSession(agent=agent, pipeline=pipeline)
    shutdown_event = asyncio.Event()

    async def on_text_message(message):
        """Process text messages from the customer; also handle interrupt command."""
        text = message.get("message", "") if isinstance(message, dict) else str(message)

        if text.strip().lower() == "interrupt":
            logging.info("Interrupt requested via text.")
            session.interrupt()
            return

        if text.strip():
            logging.info("Text message from customer: %s", text)
            await pipeline.process_text(text)

    def on_text_message_wrapper(message):
        asyncio.create_task(on_text_message(message))

    @pipeline.on("llm")
    async def on_llm(data: dict):
        """Mirror agent's text response to AGENT_RESPONSE pubsub topic for chat clients."""
        text = data.get("text", "").strip()
        if text:
            try:
                publish_config = PubSubPublishConfig(topic="AGENT_RESPONSE", message=text)
                await ctx.room.publish_to_pubsub(publish_config)
            except Exception as e:
                logging.error("Failed to publish agent response: %s", e)

    async def cleanup_session():
        await session.close()
        shutdown_event.set()

    ctx.add_shutdown_callback(cleanup_session)

    def on_session_end(reason: str):
        asyncio.create_task(ctx.shutdown())

    try:
        await ctx.connect()
        ctx.room.setup_session_end_callback(on_session_end)
        await ctx.room.wait_for_participant()

        subscribe_config = PubSubSubscribeConfig(topic="CHAT", cb=on_text_message_wrapper)
        await ctx.room.subscribe_to_pubsub(subscribe_config)

        await session.start()
        await shutdown_event.wait()
    except KeyboardInterrupt:
        pass
    finally:
        await session.close()
        await ctx.shutdown()


def make_context() -> JobContext:
    room_options = RoomOptions(room_id="<room_id>", name="First National Bank - Virtual Banker", playground=True)
    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()
