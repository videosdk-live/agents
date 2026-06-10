import asyncio
import logging

from videosdk.agents import Agent, AgentSession, JobContext, Pipeline, RoomOptions, WorkerJob,  function_tool, TTSAudioCache,InterruptConfig
from videosdk.agents.plugins import CartesiaTTS, DeepgramSTT, GoogleLLM, SileroVAD, TurnDetector, pre_download_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
pre_download_model()


# --- Fixed phrases: synthesized once, replayed from cache forever after. ---
GREETING = "Thanks for calling Northwind support. How can I help you today?"
HOLD = "Sure, let me pull that up for you. One moment."
GOODBYE = "Thanks for calling Northwind. Have a great day!"

CACHED_PHRASES = [GREETING, HOLD, GOODBYE]

# Stand-in for a real order database.
_ORDERS = {
    "1001": {"status": "shipped", "carrier": "UPS", "eta": "Thursday"},
    "1002": {"status": "processing", "carrier": None, "eta": "next week"},
    "1003": {"status": "delivered", "carrier": "FedEx", "eta": "delivered Monday"},
}


class SupportAgent(Agent):
    def __init__(self, tts_cache: TTSAudioCache) -> None:
        super().__init__(
            instructions=(
                "You are Aria, a friendly phone support agent for Northwind, an "
                "online store. Help callers check order status and connect them to "
                "a human specialist when they ask. Keep replies short and natural. "
                "To look up an order, call check_order_status with the order "
                "number, then tell the caller the result in your own words. If the "
                "lookup returns found=false, apologize and ask them to double-check the number. "
            ),
        )
        self._cache = tts_cache

    async def on_enter(self) -> None:
        # Greet from a cached phrase: synthesized once during preload(), then
        # replayed straight from memory here — no TTS call, instant playback.
        await self.session.say(GREETING, audio_data=await self._cache.fetch(GREETING))

        # --- Alternative: greet from a pre-recorded audio file -----------------
        # You can also play a ready-made audio file instead of synthesizing.
        # load_audio_file() decodes WAV/OGG/MP3/etc. to PCM bytes (no TTS call).
        # The `message` arg is still required — it is recorded for transcript
        # and chat context even though the audio is pre-supplied.
        #
        # Commented out because this example does not ship an audio file; drop
        # one next to this script and uncomment to try it:
        #
        #   from pathlib import Path
        #   from videosdk.agents import load_audio_file
        #
        #   GREETING_AUDIO = Path(__file__).parent / "greeting.ogg"
        #   await self.session.say(
        #       GREETING,
        #       audio_data=load_audio_file(str(GREETING_AUDIO)),
        #   )

    async def on_exit(self) -> None:
        await self.session.say(GOODBYE, audio_data=await self._cache.fetch(GOODBYE))

    @function_tool
    async def check_order_status(self, order_id: str) -> dict:
        """Look up the status of a customer's order.

        Args:
            order_id: The order number the caller provides.
        """
        order_id = order_id.strip()

        hold_task = asyncio.create_task(
            self.session.say(
                HOLD,
                audio_data=await self._cache.fetch(HOLD),
                add_to_chat_context=False,
            )
        )
        order = _ORDERS.get(order_id)

        await hold_task

        if order is None:
            return {"found": False, "order_id": order_id}
        return {"found": True, "order_id": order_id, **order}


async def start_session(context: JobContext) -> None:
    # Share one TTS instance between the pipeline and the cache so cached
    # phrases use the exact same voice as the agent's dynamic speech.
    tts = CartesiaTTS()
    tts_cache = TTSAudioCache(tts)

    # Warm every fixed phrase up front. Each preload() entry is one synthesis
    # now; every fetch() at runtime is then an instant cache hit.
    await tts_cache.preload(CACHED_PHRASES)

    pipeline = Pipeline(
        stt=DeepgramSTT(),
        llm=GoogleLLM(),
        tts=tts,
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
        # Smooth barge-in: when the caller interrupts, the agent's voice fades
        # out gradually over `interrupt_fade_duration` seconds instead of being
        # cut abruptly mid-word. Default is 0.4s; set 0 to disable the fade.
        interrupt_config=InterruptConfig(interrupt_fade_duration=0.3),
    )

    session = AgentSession(
        agent=SupportAgent(tts_cache),
        pipeline=pipeline,
    )

    await session.start(wait_for_participant=True, run_until_shutdown=True)


def make_context() -> JobContext:
    return JobContext(
        room_options=RoomOptions(
            room_id="<room_id>",
            name="Northwind Support Agent",
            playground=True,
        )
    )


if __name__ == "__main__":
    job = WorkerJob(entrypoint=start_session, jobctx=make_context)
    job.start()