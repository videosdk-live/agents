"""
Sequential agent-to-agent handoff — Phase 1 context primitives in an agent.

Two peer agents share one conversation. When the intake agent transfers the
caller to the billing agent, it records the transfer on the shared
``ChatContext`` with ``add_handoff(...)``. The billing agent reads that handoff
marker when it takes over, so it can greet the caller with context.

Phase 1 primitives demonstrated:
  - ``ChatContext.add_handoff(...)``  — record a control transfer between agents
  - reading ``AgentHandoff`` items    — the receiving agent inspects *why*
  - ``agent_id`` attribution          — stamp who produced a message

Scope note: the agent-switch mechanism itself (returning a new ``Agent`` from a
``@function_tool`` with ``inherit_context=True``) is existing framework
behavior. Phase 1 adds the *context-layer* primitives used alongside it —
``add_handoff`` and per-item ``agent_id``. This file is a wiring pattern;
running it end-to-end needs VideoSDK credentials and a room.
"""

import logging

from videosdk.agents import (
    Agent,
    AgentSession,
    Pipeline,
    function_tool,
    JobContext,
    RoomOptions,
    WorkerJob,
    ChatRole,
    AgentHandoff,
    EOUConfig,
    InterruptConfig
)
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


class IntakeAgent(Agent):
    """First-line agent. Triages the caller, then hands off to billing."""

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are the intake agent for a support line. Greet the caller, "
                "find out what they need, and if it concerns a charge, payment, "
                "or refund, call transfer_to_billing with a short reason."
            ),
            agent_id="intake",
        )

    async def on_enter(self) -> None:
        await self.session.say("Hi, you've reached support. How can I help?")

    async def on_exit(self) -> None:
        pass

    @function_tool
    async def transfer_to_billing(self, reason: str) -> Agent:
        """Transfer the caller to the billing specialist.

        Args:
            reason: Short reason for the transfer (e.g. "disputed charge").
        """
        logging.info("Tool invoked: transfer_to_billing(reason=%s)", reason)
        # --- Phase 1 primitive ---------------------------------------------
        # Record the handoff on the shared context. AgentHandoff is a
        # structural item: it stays in the context for audit and for the
        # receiving agent to read, but provider converters never send it to
        # an LLM.
        self.chat_context.add_handoff(
            from_agent=self.id,
            to_agent="billing",
            reason=reason,
        )
        # Existing framework handoff: the new agent inherits this context
        # (including the AgentHandoff item just appended).
        return BillingAgent(inherit_context=True)


class BillingAgent(Agent):
    """Billing specialist. Reads the handoff marker to greet with context."""

    def __init__(self, inherit_context: bool = False) -> None:
        super().__init__(
            instructions=(
                "You are the billing specialist. Resolve charge disputes, "
                "payment questions, and refunds."
            ),
            agent_id="billing",
            inherit_context=inherit_context,
        )

    async def on_enter(self) -> None:
        reason = self._last_handoff_reason()
        if reason:
            await self.session.say(
                f"I'm the billing specialist — I see you're here about "
                f"{reason}. Let's get that sorted."
            )
        else:
            await self.session.say(
                "I'm the billing specialist. What can I help you with?"
            )
        # --- Phase 1 primitive ---------------------------------------------
        # Attribute this agent's own marker line to itself via agent_id, so a
        # later filter (ctx.copy(filter_agent_id="billing")) can isolate it.
        self.chat_context.add_message(
            role=ChatRole.ASSISTANT,
            content="[billing agent engaged]",
            agent_id=self.id,
        )

    async def on_exit(self) -> None:
        pass

    def _last_handoff_reason(self) -> str | None:
        """Read the most recent AgentHandoff marker from the shared context."""
        for item in reversed(self.chat_context.items):
            if isinstance(item, AgentHandoff):
                return item.reason
        return None


async def entrypoint(ctx: JobContext) -> None:
    pipeline = Pipeline(
        stt=DeepgramSTT(endpointing=-1),
        llm=GoogleLLM(model="gemini-2.5-flash-lite"),
        tts=CartesiaTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
        eou_config=EOUConfig(
            mode='ADAPTIVE',
            # EOU mode: 'DEFAULT' uses fixed min/max timeouts; 'ADAPTIVE' computes a continuous timeout based on confidence.
            min_max_speech_wait_timeout=[0.1, 0.2],  # Tuple (min_duration, max_duration) for EOU.
        ),
        interrupt_config=InterruptConfig(
            mode="HYBRID",
            # Interruption mode: 'VAD_ONLY' (voice activity), 'STT_ONLY' (speech-to-text), or 'HYBRID' (both).
            interrupt_min_duration=0.2,  # Minimum continuous speech duration (VAD-based) to trigger an interruption.
            interrupt_min_words=2,  # Minimum number of transcribed words (STT-based) to trigger an interruption.
            false_interrupt_pause_duration=2.0,  # Duration to pause TTS for false interruption detection.
            resume_on_false_interrupt=True,  # Automatically resume TTS after a false interruption timeout.
        )

    )
    session = AgentSession(agent=IntakeAgent(), pipeline=pipeline)
    await session.start(wait_for_participant=True, run_until_shutdown=True)


def make_context() -> JobContext:
    return JobContext(
        room_options=RoomOptions(name="Sequential Handoff Agent", playground=True)
    )


if __name__ == "__main__":
    WorkerJob(entrypoint=entrypoint, jobctx=make_context).start()
