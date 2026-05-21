"""
Supervisor delegating to a sub-agent — Phase 1 fork/merge primitives.

A supervisor agent answers general questions itself, but for a focused
sub-task (checking refund eligibility) it spins off a sub-agent: it ``fork``s a
scoped context for the sub-agent to work on, then ``merge``s the sub-agent's
result back into its own context.

Phase 1 primitives demonstrated:
  - ``ChatContext.fork_brief(...)``     — a fresh, scoped context for the sub-agent
  - ``ChatContext.fork_filtered(...)``  — alternative: instructions + recent turns
  - ``ChatContext.merge_result(...)``   — pull only the sub-agent's conclusion back
  - ``ChatContext.merge_summary(...)``  — alternative: pull an LLM summary back
  - ``ReadOnlyChatContext``             — hand the sub-agent a non-mutating view

Scope note: Phase 1 ships the *context-layer* primitives (fork/merge). The
runtime that actually spawns and runs a sub-agent on the forked context is
Phase 2. The ``# --- Phase 2 ---`` block below marks where that orchestration
plugs in; the fork/merge calls around it are real Phase 1 API. Running this
file end-to-end needs VideoSDK credentials, a room, and the Phase 2 runtime.
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
    ReadOnlyChatContext,
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


class SupervisorAgent(Agent):
    """Handles the call directly, delegating focused sub-tasks to sub-agents."""

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a customer-support supervisor. Answer general questions "
                "yourself. For refund-eligibility questions, call "
                "check_refund_eligibility with the order id."
            ),
            agent_id="supervisor",
        )

    async def on_enter(self) -> None:
        await self.session.say("Support supervisor here — how can I help?")

    async def on_exit(self) -> None:
        pass

    @function_tool
    async def check_refund_eligibility(self, order_id: str) -> dict:
        """Delegate a refund-eligibility check to a focused sub-agent.

        Args:
            order_id: The order to evaluate (e.g. "4471").
        """
        logging.info("Tool invoked: check_refund_eligibility(order_id=%s)", order_id)
        # --- Phase 1 primitive --------------------------------------------
        # Fork a FRESH, scoped context for the sub-agent: just its own
        # instructions + a one-line task brief, no parent conversation
        # history. fork_brief returns a new, independent ChatContext.
        sub_ctx = self.chat_context.fork_brief(
            instructions=(
                "You are a refund-eligibility checker. Decide only whether the "
                "given order qualifies for a refund and answer in one sentence."
            ),
            task_brief=f"Is order {order_id} eligible for a refund?",
            agent_id="refund-checker",
        )
        # Alternative — give the sub-agent recent context instead of a blank
        # slate (instructions + the last 2 turns, tool calls scoped out):
        #   sub_ctx = self.chat_context.fork_filtered(recent_turns=2, tools=[])

        # --- Phase 2: orchestration (not in this phase) -------------------
        # A sub-agent would now run on `sub_ctx` via its own pipeline/session.
        # To let it READ the supervisor's live history without being able to
        # mutate it, hand it a read-only view:
        #   parent_view = ReadOnlyChatContext(self.chat_context.items)
        #   sub_agent = RefundCheckerAgent(); sub_agent.chat_context = sub_ctx
        #   await run_subagent(sub_agent, parent_view)
        # For this Phase 1 pattern file we simulate the sub-agent's reply:
        sub_ctx.add_message(
            role=ChatRole.ASSISTANT,
            content=f"Order {order_id} is within the 30-day window — refundable.",
            agent_id="refund-checker",
        )
        # --- end Phase 2 placeholder --------------------------------------

        # --- Phase 1 primitive --------------------------------------------
        # Merge the sub-agent's work back into the supervisor's context.
        # merge_result pulls ONLY the final answer (cheapest). To instead pull
        # an LLM-condensed summary of everything the sub-agent did, use:
        #   await self.chat_context.merge_summary(
        #       sub_ctx, llm=self.session.pipeline.llm, agent_id="refund-checker")
        await self.chat_context.merge_result(sub_ctx, agent_id="refund-checker")

        conclusion = str(sub_ctx.messages()[-1].content)
        return {"order_id": order_id, "conclusion": conclusion}


async def entrypoint(ctx: JobContext) -> None:
    pipeline = Pipeline(
        stt=DeepgramSTT(),
        llm=GoogleLLM(),
        tts=CartesiaTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
    )
    session = AgentSession(agent=SupervisorAgent(), pipeline=pipeline)
    await session.start(wait_for_participant=True, run_until_shutdown=True)


def make_context() -> JobContext:
    return JobContext(
        room_options=RoomOptions(name="Supervisor + Sub-agent", playground=True)
    )


if __name__ == "__main__":
    WorkerJob(entrypoint=entrypoint, jobctx=make_context).start()
