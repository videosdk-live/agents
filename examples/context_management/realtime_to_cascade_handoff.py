"""
Realtime → Cascade handoff — carrying chat context across a pipeline switch.

One agent, one conversation, one ``AgentSession``. The call starts on a
**realtime** speech-to-speech model (lowest latency). When the caller needs the
deterministic, provider-flexible cascade stack, a ``@function_tool`` switches
the *pipeline itself* to a **cascade** pipeline (STT → LLM → TTS) via
``pipeline.change_pipeline(...)``.

While the realtime half runs, the SDK keeps ``agent.chat_context`` faithful: it
records the realtime model's finalized transcript turns AND its tool calls
(``FunctionCall`` + ``FunctionCallOutput``). After the switch, the cascade LLM
reads that same ``chat_context`` — so it continues with the full history of the
realtime half, tool calls included.

Realtime provider: pick one in ``make_realtime_model()`` below. Exactly one
block must be uncommented — swap which one to try a different provider.

Scope note: switching the pipeline mid-call and recording realtime activity
into ``chat_context`` is existing/this-feature SDK behavior — this file is a
wiring pattern. Running it end-to-end needs VideoSDK credentials and a room.
"""

import asyncio
import logging

from videosdk.agents import (
    Agent,
    AgentSession,
    Pipeline,
    function_tool,
    JobContext,
    RoomOptions,
    WorkerJob,
)
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.google import GoogleLLM  # cascade LLM — always used
from videosdk.plugins.cartesia import CartesiaTTS
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model

# --- Realtime provider (pick one) -------------------------------------------
# Gemini Live is active. To try another provider, comment the Gemini import +
# block in make_realtime_model() and uncomment one of the others.
from videosdk.plugins.google import GeminiRealtime, GeminiLiveConfig
# from videosdk.plugins.openai import OpenAIRealtime, OpenAIRealtimeConfig
# from videosdk.plugins.xai import XAIRealtime, XAIRealtimeConfig
# from videosdk.plugins.ultravox import UltravoxRealtime, UltravoxLiveConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
pre_download_model()


def make_realtime_model():
    """Return the realtime model the call starts on. Swap providers here.

    Each provider records its transcripts and tool calls into the agent's
    ``chat_context`` (the realtime context-management feature), so the cascade
    LLM inherits them after the switch. Exactly one block must be live.
    """
    # --- Gemini Live (active) ---
    return GeminiRealtime(
        model="gemini-3.1-flash-live-preview",
        config=GeminiLiveConfig(voice="Leda", response_modalities=["AUDIO"]),
    )

    # --- OpenAI Realtime ---
    # return OpenAIRealtime(
    #     model="gpt-realtime",
    #     config=OpenAIRealtimeConfig(voice="alloy", modalities=["text", "audio"]),
    # )

    # --- xAI Realtime ---
    # return XAIRealtime(
    #     model="grok-4-1-fast-non-reasoning",
    #     config=XAIRealtimeConfig(voice="Eve"),
    # )

    # --- Ultravox Realtime ---
    # return UltravoxRealtime(config=UltravoxLiveConfig())


class SupportAgent(Agent):
    """A support agent that begins on realtime and can fall back to cascade."""

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a support agent for an online store. Always respond "
                "in English. Help the caller with their order. If they ask "
                "for the detailed, careful, or standard mode, call "
                "switch_to_cascade."
            ),
            agent_id="support",
        )
        self._switch_task: asyncio.Task | None = None
        self._switched = False

    async def on_enter(self) -> None:
        await self.session.say(
            "Hi, you've reached store support. How can I help with your order?"
        )

    async def on_exit(self) -> None:
        pass

    @function_tool
    async def lookup_order(self, order_id: str) -> dict:
        """Look up the status of an order.

        Args:
            order_id: The caller's order number.
        """
        logging.info("Tool invoked: lookup_order(order_id=%s)", order_id)
        # A realtime tool call. The realtime plugin emits a canonical
        # realtime_model_function_executed event, and the SDK records it as a
        # FunctionCall + FunctionCallOutput on agent.chat_context — so this
        # tool call survives the switch to cascade below.
        return {"order_id": order_id, "status": "shipped", "eta": "Tuesday"}

    @function_tool
    async def switch_to_cascade(self) -> dict:
        """Switch the conversation to the standard cascade voice pipeline.

        Call this when the caller asks for the detailed or standard mode.
        """
        logging.info("Tool invoked: switch_to_cascade")

        # The realtime half's transcript turns and tool calls are recorded on
        # the shared chat_context — log how much carries into the switch.
        items = self.chat_context.items
        logging.info(
            "chat_context carries %d item(s) into the switch: [%s]",
            len(items),
            ", ".join(type(i).__name__ for i in items),
        )

        # --- realtime → cascade ------------------------------------------
        # The switch runs in its own task, deferred until AFTER this tool
        # returns. change_pipeline() tears down the realtime model — and this
        # tool is executing *inside* that model's task. Calling change_pipeline
        # synchronously here would make the task cancel itself mid-stack. The
        # deferred task runs the switch from outside the model's task instead.
        #
        # agent.chat_context is NOT touched by the switch — the realtime
        # transcript turns and tool calls already recorded on it carry across,
        # and the cascade LLM reads that history on its next turn.
        async def _do_switch() -> None:
            await self.session.pipeline.change_pipeline(
                stt=DeepgramSTT(),
                llm=GoogleLLM(model="gemini-3.1-flash-lite-preview"),
                tts=CartesiaTTS(),
                vad=SileroVAD(),
                turn_detector=TurnDetector(),
            )
            await self.session.say(
                "Done — I've switched to standard mode. I still have everything "
                "we covered, so let's keep going."
            )

        self._switch_task = asyncio.create_task(_do_switch())
        # Tools return a dict — realtime providers require structured results.
        return {"status": "switching to the cascade pipeline"}


async def entrypoint(ctx: JobContext) -> None:
    # The call starts on a realtime pipeline.
    pipeline = Pipeline(llm=make_realtime_model())
    session = AgentSession(agent=SupportAgent(), pipeline=pipeline)
    await session.start(wait_for_participant=True, run_until_shutdown=True)


def make_context() -> JobContext:
    return JobContext(
        room_options=RoomOptions(
            name="Realtime to Cascade Handoff", playground=True
        )
    )


if __name__ == "__main__":
    WorkerJob(entrypoint=entrypoint, jobctx=make_context).start()
