"""
Use Case: Fast-food drive-through ordering agent (QuickBite) — handles fast, overlapping speech.
Pipeline: P1 — DeepgramSTT + GoogleLLM + CartesiaTTS + SileroVAD + TurnDetector + EOUConfig + InterruptConfig
Demonstrates: ADAPTIVE EOUConfig for tight timeouts, HYBRID interrupt mode, non-interruptible order confirmation.
Env Vars: VIDEOSDK_AUTH_TOKEN, DEEPGRAM_API_KEY, GOOGLE_API_KEY, CARTESIA_API_KEY
"""

import logging
from google.genai.interactions import generation_config
from videosdk.agents import (
    Agent, AgentSession, Pipeline, function_tool, WorkerJob, JobContext, RoomOptions,
    EOUConfig, InterruptConfig,
)
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.openai import OpenAILLM
from videosdk.plugins.cartesia import CartesiaTTS,GenerationConfig
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
pre_download_model()


class DriveThruAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are the drive-through ordering agent for QuickBite.
            Take food orders accurately and efficiently. The menu includes:
            Burgers: Classic ($5), Cheese ($6), Double ($8)
            Sides: Fries ($2.50), Onion Rings ($3), Salad ($3.50)
            Drinks: Small ($1.50), Medium ($2), Large ($2.50)
            After collecting the full order, use confirm_order to read it back.
            The confirmation must not be interrupted — customers must hear the full order.
            Upsell a side or drink if the order total is under $10.
            Be fast and friendly — customers are in their cars.""",
        )

    async def on_enter(self) -> None:
        await self.session.say(
            "Welcome to QuickBite! What can I get started for you today?"
        )

    async def on_exit(self) -> None:
        await self.session.say(
            "Thanks for choosing QuickBite! Pull up to the first window. Have a great day!"
        )

    @function_tool
    async def confirm_order(self, items: str, total: float) -> dict:
        """Read the full order back to the customer for confirmation. This cannot be interrupted.

        Args:
            items: Comma-separated list of ordered items with sizes
            total: Total price of the order in USD
        """
        await self.session.say(
            f"To confirm your order: {items}. "
            f"Your total comes to ${total:.2f}. "
            "Does that sound right?",
            interruptible=False,
        )
        return {
            "items": items,
            "total": total,
            "status": "confirmed_pending_customer_approval",
        }

    @function_tool
    async def place_order(self, items: str, total: float) -> dict:
        """Finalize and place the customer's order after they confirm.

        Args:
            items: Final list of ordered items
            total: Final total price in USD
        """
        order_id = f"QB-{abs(hash(items)) % 10000:04d}"
        return {
            "order_id": order_id,
            "items": items,
            "total": total,
            "prep_time_minutes": 4,
            "window": "Window 1",
            "status": "placed",
        }


async def entrypoint(ctx: JobContext):
    agent = DriveThruAgent()

    pipeline = Pipeline(
        stt=DeepgramSTT(),
        # llm=GoogleLLM(),
        llm=OpenAILLM(),
        tts=CartesiaTTS(generation_config=GenerationConfig(volume=1,speed=1.3)),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
        # ADAPTIVE mode computes EOU timeout based on speech confidence — ideal for quick back-and-forth.
        eou_config=EOUConfig(
            mode="ADAPTIVE",
            min_max_speech_wait_timeout=[0.3, 0.6],
        ),
        # HYBRID mode uses both VAD and STT signals to detect real interruptions.
        interrupt_config=InterruptConfig(
            mode="HYBRID",
            interrupt_min_duration=0.2,
            interrupt_min_words=2,
            false_interrupt_pause_duration=1.5,
            resume_on_false_interrupt=True,
        ),
    )
    session = AgentSession(agent=agent, pipeline=pipeline)
    await session.start(wait_for_participant=True, run_until_shutdown=True)


def make_context() -> JobContext:
    room_options = RoomOptions(room_id="<room_id>", name="QuickBite Drive-Through", playground=True)
    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()
