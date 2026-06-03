"""
Long-conversation context management — Phase 1 ContextWindow + active config.

An agent built for long calls. The Pipeline is given a ``ContextWindow`` so
that as the conversation grows, old turns are automatically compressed into a
summary and the context is truncated to a budget — the agent keeps long-term
memory without unbounded token growth.

It also shows an agent changing its *own* configuration mid-call: when the
caller escalates, the agent records an ``AgentConfigUpdate`` and resolves the
now-effective instructions with ``active_config_at()``.

Phase 1 primitives demonstrated:
  - ``ContextWindow(...)``                  — wired into the Pipeline
  - ``ChatContext.summarize(...)``          — invoked automatically by ContextWindow
  - ``ChatContext.add_config_update(...)``  — record a mid-call instruction change
  - ``ChatContext.active_config_at()``      — resolve the currently-effective config

Scope note: ``ContextWindow.manage()`` is invoked automatically by the pipeline
before each LLM turn — wired, runnable behaviour. The summarize/truncate work
happens on the live ``ChatContext``. Running end-to-end needs VideoSDK
credentials and a room.
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
    ContextWindow,
)
from videosdk.agents.plugins import DeepgramSTT, GoogleLLM, CartesiaTTS, SileroVAD, TurnDetector, pre_download_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
pre_download_model()


class LongCallAgent(Agent):
    """Support agent for long troubleshooting calls; can escalate mid-call."""

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a patient support agent for long troubleshooting "
                "calls. Work the problem step by step."
            ),
            agent_id="support",
        )

    async def on_enter(self) -> None:
        await self.session.say("Support here — walk me through what's happening.")

    async def on_exit(self):
        await self.session.say("Thank you for calling, Have a nice day.")

    @function_tool
    async def escalate_to_manager(self) -> str:
        """Escalate the call to manager authority for the rest of the call."""
        logging.info("Tool invoked: escalate_to_manager")
        # --- Phase 1 primitive --------------------------------------------
        # Record a mid-conversation config change. AgentConfigUpdate is a
        # structural item: it feeds active_config_at() and is excluded from
        # provider calls. summarize()/truncate() preserve it, so the escalation
        # survives context compression later in a long call.
        self.chat_context.add_config_update(
            instructions=(
                "You now have manager authority: you may approve refunds up to "
                "$500 and waive fees. Be decisive."
            ),
                tools=["approve_refund", "waive_fee"],
            agent_id=self.id,
        )
        # --- Phase 1 primitive --------------------------------------------
        # Resolve the effective instructions/tools as of now. Walks the
        # SYSTEM/DEVELOPER messages + AgentConfigUpdate items in order.
        instructions, tools = self.chat_context.active_config_at()
        logging.info("Effective config after escalation — tools=%s", tools)
        await self.session.reply(instructions=instructions)
        return "Escalated — manager authority is now active."

    @function_tool
    async def approve_refund(self, order_id: str, amount: float) -> str:
        """Approve a refund on an order. Manager authority — unlocked after
        escalation; refunds are capped at $500.

        Args:
            order_id: The order receiving the refund.
            amount: Refund amount in USD; anything above 500 is rejected.
        """
        logging.info(
            "Tool invoked: approve_refund(order_id=%s, amount=%.2f)",
            order_id,
            amount,
        )
        if amount > 500:
            return (
                f"Refund of ${amount:.2f} exceeds the $500 manager limit on "
                f"order {order_id} — needs director approval."
            )
        return f"Approved a ${amount:.2f} refund on order {order_id}."

    @function_tool
    async def waive_fee(self, order_id: str, fee_type: str) -> str:
        """Waive a fee on an order. Manager authority — unlocked after
        escalation.

        Args:
            order_id: The order the fee is on.
            fee_type: The fee to waive (e.g. "late payment", "restocking").
        """
        logging.info(
            "Tool invoked: waive_fee(order_id=%s, fee_type=%s)",
            order_id,
            fee_type,
        )
        return f"Waived the {fee_type} fee on order {order_id}."


async def entrypoint(ctx: JobContext) -> None:
    # --- Phase 1 primitive ------------------------------------------------
    # ContextWindow is the policy layer. Before each LLM turn the pipeline
    # calls window.manage(), which compresses old turns via
    # ChatContext.summarize() and then truncates to the budget. Structural
    # items (system/developer messages, handoffs, config updates) are
    # preserved through both.
    context_window = ContextWindow(
        max_tokens=500,        # compress + truncate once the context exceeds this
        keep_recent_turns=4,    # keep the last 4 user turns verbatim
    )
    pipeline = Pipeline(
        stt=DeepgramSTT(),
        llm=GoogleLLM(),
        tts=CartesiaTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
        context_window=context_window,
    )
    session = AgentSession(agent=LongCallAgent(), pipeline=pipeline)
    await session.start(wait_for_participant=True, run_until_shutdown=True)


def make_context() -> JobContext:
    return JobContext(
        room_options=RoomOptions(name="Long Conversation Agent", playground=True)
    )


if __name__ == "__main__":
    WorkerJob(entrypoint=entrypoint, jobctx=make_context).start()
