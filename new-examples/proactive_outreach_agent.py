"""
Use Case: Subscription renewal outbound call agent (StreamBox) — proactively speaks if user is silent.
Pipeline: P1 — DeepgramSTT + AnthropicLLM + GoogleTTS + SileroVAD + TurnDetector
Demonstrates: AgentSession(wake_up=N) for proactive outreach, session.reply() for agent-initiated speech.
Env Vars: VIDEOSDK_AUTH_TOKEN, DEEPGRAM_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY
"""

import logging
from videosdk.agents import Agent, AgentSession, Pipeline, function_tool, WorkerJob, JobContext, RoomOptions
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.anthropic import AnthropicLLM
from videosdk.plugins.google import GoogleTTS
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
pre_download_model()


class RenewalAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are a renewal specialist for StreamBox, a streaming entertainment service.
            You are calling because the customer's subscription expires in 3 days.
            Your goal is to confirm or cancel their renewal.
            If they hesitate, offer a 20% loyalty discount valid for 48 hours.
            Be warm, conversational, and brief — this is an outbound call.
            If they confirm renewal, use confirm_renewal tool.
            If they want to cancel, use process_cancellation tool and ask for a brief reason.""",
        )

    async def on_enter(self) -> None:
        await self.session.reply(
            instructions=(
                "Greet the customer warmly, introduce yourself as a StreamBox renewal specialist, "
                "mention that their subscription expires in 3 days, and ask if they'd like to renew."
            )
        )

    async def on_exit(self) -> None:
        await self.session.say(
            "Thank you for your time. Enjoy StreamBox! Goodbye."
        )

    @function_tool
    async def confirm_renewal(self, plan: str = "current") -> dict:
        """Confirm the customer's subscription renewal.

        Args:
            plan: Plan to renew — 'current', 'upgraded', or 'discounted_20_percent'
        """
        plan_prices = {
            "current": 12.99,
            "upgraded": 17.99,
            "discounted_20_percent": 10.39,
        }
        price = plan_prices.get(plan, 12.99)
        confirmation_id = f"SB-RNW-{abs(hash(plan)) % 100000:05d}"
        return {
            "confirmation_id": confirmation_id,
            "plan": plan,
            "monthly_price": price,
            "renewal_date": "3 days from today",
            "status": "confirmed",
        }

    @function_tool
    async def process_cancellation(self, reason: str = "not specified") -> dict:
        """Process the customer's request to cancel their subscription.

        Args:
            reason: Customer's reason for cancellation
        """
        return {
            "status": "cancellation_scheduled",
            "effective_date": "end of current billing period",
            "reason_recorded": reason,
            "message": "Your subscription will remain active until the end of the billing period.",
        }


async def entrypoint(ctx: JobContext):
    agent = RenewalAgent()

    pipeline = Pipeline(
        stt=DeepgramSTT(),
        llm=AnthropicLLM(),
        tts=GoogleTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
    )

    session = AgentSession(
        agent=agent,
        pipeline=pipeline,
        wake_up=45,  # If the user is silent for 45 seconds, the agent re-engages proactively.
    )

    async def on_wake_up():
        """Re-engage the customer if they go silent mid-call."""
        await session.reply(
            instructions="The customer has been silent. Gently check if they're still there and re-state the renewal offer."
        )

    session.on_wake_up = on_wake_up

    await session.start(wait_for_participant=True, run_until_shutdown=True)


def make_context() -> JobContext:
    room_options = RoomOptions(room_id="<room_id>", name="StreamBox Renewal Call", playground=True)
    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()
