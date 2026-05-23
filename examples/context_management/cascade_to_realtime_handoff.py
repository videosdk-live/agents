"""
Cascade → Realtime handoff — carrying chat context across a pipeline switch.

One agent, one conversation, one ``AgentSession``. The call starts on a
**cascade** pipeline (STT → LLM → TTS). When the caller asks for faster, more
natural responses, a ``@function_tool`` switches the *pipeline itself* to a
**realtime** speech-to-speech model via ``pipeline.change_pipeline(...)``.

The agent's ``chat_context`` lives on the agent and is untouched by the switch.
On connect, the realtime model **seeds itself** from that ``chat_context`` —
prior user/assistant turns — so the realtime half of the call continues with
the history of the cascade half.

Realtime provider: pick one in ``make_realtime_model()`` below. Exactly one
block must be uncommented — swap which one to try a different provider.

Scope note: switching the pipeline mid-call and seeding the realtime model is
existing/this-feature SDK behavior — this file is a wiring pattern. Running it
end-to-end needs VideoSDK credentials and a room.
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
    EOUConfig,
)
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.google import GoogleLLM  # cascade LLM — always used
from videosdk.plugins.cartesia import CartesiaTTS
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model

# --- Realtime provider imports (the one used in make_realtime_model() picks the provider) ---
from videosdk.plugins.google import GeminiRealtime, GeminiLiveConfig
from videosdk.plugins.openai import OpenAIRealtime, OpenAIRealtimeConfig, OpenAILLM
from videosdk.plugins.xai import XAIRealtime, XAIRealtimeConfig
from videosdk.plugins.ultravox import UltravoxRealtime, UltravoxLiveConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
pre_download_model()


def make_realtime_model():
    """Return the realtime model to switch into. Swap providers here.

    Each provider seeds itself from the agent's ``chat_context`` on connect
    (the realtime context-management feature). Exactly one block must be live.
    """
    # --- Gemini Live ---
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
    """A support agent that begins on cascade and can upgrade to realtime."""

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a support agent for an online store. Always respond "
                "in English. Help the caller with their order. If they ask for "
                "faster, snappier, or more natural-sounding responses, call "
                "switch_to_realtime."
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
        # A normal cascade tool call. The SDK records this as a
        # FunctionCall + FunctionCallOutput on agent.chat_context — so when the
        # call later switches to realtime, the realtime model is seeded with
        # this tool call too, not just the spoken turns.
        return {"order_id": order_id, "status": "shipped", "eta": "Tuesday"}

    @function_tool
    async def switch_to_realtime(self) -> dict:
        """Switch the conversation to a low-latency realtime voice model.

        Call this when the caller asks for faster or more natural responses.
        """
        logging.info("Tool invoked: switch_to_realtime")

        # Idempotent: after the switch the realtime model still has this tool,
        # and — seeded with a switch-heavy context — keeps calling it, looping
        # change_pipeline forever. Ignore repeat calls once switched.
        if self._switched:
            logging.info("Already on the realtime pipeline — ignoring repeat switch.")
            return {"status": "already on the realtime pipeline"}
        self._switched = True

        # The cascade half's transcript turns and tool calls are recorded on
        # the shared chat_context — log how much carries into the switch.
        items = self.chat_context.items
        logging.info(
            "chat_context carries %d item(s) into the switch: [%s]",
            len(items),
            ", ".join(type(i).__name__ for i in items),
        )

        # --- cascade → realtime ------------------------------------------
        # The switch runs in its own task, deferred until AFTER this tool
        # returns. change_pipeline() tears down the pipeline that is currently
        # running this very tool — calling it synchronously here would make the
        # running task cancel itself mid-stack. agent.chat_context is NOT
        # touched; it carries across, and on connect the realtime model seeds
        # itself from it.
        async def _do_switch() -> None:
            await self.session.pipeline.change_pipeline(llm=make_realtime_model())
            await self.session.say(
                "Done — I've switched to realtime mode. I still have our whole "
                "conversation, so let's keep going."
            )

        self._switch_task = asyncio.create_task(_do_switch())
        # Tools return a dict — realtime providers require structured results.
        return {"status": "switching to the realtime pipeline"}


async def entrypoint(ctx: JobContext) -> None:
    # The call starts on a cascade pipeline.
    pipeline = Pipeline(
        stt=DeepgramSTT(),
        llm=GoogleLLM(model="gemini-3.1-flash-lite-preview"),
        tts=CartesiaTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
        eou_config=EOUConfig(
            mode='ADAPTIVE',
            # EOU mode: 'DEFAULT' uses fixed min/max timeouts; 'ADAPTIVE' computes a continuous timeout based on confidence.
            min_max_speech_wait_timeout=[0.1, 0.5],  # Tuple (min_duration, max_duration) for EOU.
        ),
    )
    session = AgentSession(agent=SupportAgent(), pipeline=pipeline)
    await session.start(wait_for_participant=True, run_until_shutdown=True)


def make_context() -> JobContext:
    return JobContext(
        room_options=RoomOptions(
            name="Cascade to Realtime Handoff", playground=True
        )
    )


if __name__ == "__main__":
    WorkerJob(entrypoint=entrypoint, jobctx=make_context).start()
