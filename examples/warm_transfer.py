import logging

from videosdk.agents import (
    Agent,
    AgentSession,
    JobContext,
    Options,
    Pipeline,
    RoomOptions,
    WorkerJob,
    function_tool,
)
from videosdk.agents.warm_transfer import SIPDestination, WarmTransferConfig
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.google import GoogleLLM
from videosdk.plugins.cartesia import CartesiaTTS
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
pre_download_model()


class CustomerServiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful customer service agent. "
                "If the caller asks to speak to a manager or supervisor, or "
                "their issue requires a human, call the escalate_to_human tool."
            )
        )

    async def on_enter(self) -> None:
        await self.session.say("Hi, how can I help you today?")

    async def on_exit(self) -> None:
        pass

    @function_tool
    async def escalate_to_human(self, reason: str) -> str:
        """Escalate this call to a human supervisor with a warm transfer.

        Args:
            reason: Short description of why the escalation is happening.
        """
        config = WarmTransferConfig(
            destination=SIPDestination(
                # The SIP routing rule to dial the supervisor through.
                routing_rule_id="rr_xxxxxxxx",
                # The supervisor's phone number to dial (E.164).
                sip_call_to="+1XXXXXXXXXX",
                # Caller-ID to present on the outbound call — must be a number
                # this routing rule / trunk is authorised to send.
                sip_call_from="+1XXXXXXXXXX",
            ),
            # Optional: override the default summary behavior with a custom
            # prompt and/or a different LLM, e.g.
            #   summary_llm=AnthropicLLM(model="claude-3-5-sonnet"),
            #   summary_prompt="Custom prompt that overrides the default 150-word briefing.",
            # Optional: build the consultation-room pipeline yourself (needed if
            # your providers take constructor args rather than reading env vars):
            #   briefing_pipeline_factory=lambda: Pipeline(
            #       stt=DeepgramSTT(), llm=GoogleLLM(), tts=CartesiaTTS(),
            #       vad=SileroVAD(), turn_detector=TurnDetector(),
            #   ),
        )

        result = await self.session.warm_transfer(config)
        if result.success:
            return "Connected to a supervisor."
        return (
            "I couldn't reach a supervisor right now. "
            "Let me keep helping you in the meantime."
        )


async def entrypoint(ctx: JobContext) -> None:
    pipeline = Pipeline(
        stt=DeepgramSTT(),
        llm=GoogleLLM(),
        tts=CartesiaTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
    )
    session = AgentSession(agent=CustomerServiceAgent(), pipeline=pipeline)

    @session.on_warm_transfer()
    def on_any_phase(payload):
        logging.info("[WARM TRANSFER] phase=%s data=%s", payload["phase"].value, payload["data"])

    @session.on_warm_transfer("transfer_complete")
    def on_done(payload):
        logging.info("[WARM TRANSFER] → complete")

    await session.start(wait_for_participant=True, run_until_shutdown=True)


def make_context() -> JobContext:
    return JobContext(
        room_options=RoomOptions(
            name="Warm Transfer Demo",
            playground=True,
        )
    )


if __name__ == "__main__":
    job = WorkerJob(
        entrypoint=entrypoint,
        jobctx=make_context,
        options=Options(
            agent_id="YOUR_AGENT_ID",
            register=True,
            host="localhost",
            port=8081,
            max_processes=3,
            num_idle_processes=2,
        ),
    )
    job.start()
