"""
Use Case: Live sports commentary agent — receives match events as text, reads them aloud.
Pipeline: P6 — GoogleLLM + CartesiaTTS (text in via pubsub, voice out via TTS)
Demonstrates: Text-to-voice pipeline, event-driven pipeline.process_text() for live broadcasting.
Env Vars: VIDEOSDK_AUTH_TOKEN, GOOGLE_API_KEY, CARTESIA_API_KEY
"""

import re
import asyncio
import logging
from videosdk import PubSubSubscribeConfig
from videosdk.agents import Agent, AgentSession, Pipeline, WorkerJob, JobContext, RoomOptions, run_tts
from videosdk.plugins.google import GoogleLLM
from videosdk.plugins.cartesia import CartesiaTTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class SportsCommentatorAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are a live sports commentator broadcasting a football match.
            You receive raw match event data (e.g., goal, foul, substitution, corner, offside)
            and convert it into exciting, energetic spoken commentary.
            Keep each commentary under two sentences.
            Use vivid, action-packed language. Vary your energy level — goals deserve maximum excitement.
            For fouls and cards, be factual but dramatic. For substitutions, be informative and smooth.
            Never repeat event data verbatim — always interpret and dramatize it.
            Never use all-caps words in your commentary — write naturally as spoken language.""",
        )

    async def on_enter(self) -> None:
        await self.session.say(
            "Good evening, and welcome to tonight's live match coverage! "
            "We are live at the stadium — the crowd is buzzing with anticipation. "
            "Standing by for the first event!"
        )

    async def on_exit(self) -> None:
        await self.session.say(
            "And that's a wrap on tonight's coverage. What a match it's been! "
            "This is your commentator, signing off. Until next time!"
        )


async def entrypoint(ctx: JobContext):
    agent = SportsCommentatorAgent()

    pipeline = Pipeline(
        llm=GoogleLLM(),
        tts=CartesiaTTS(),
    )

    @pipeline.on("tts")
    async def normalize_commentary(text_stream):
        """
        Normalize LLM commentary text before TTS renders it.

        Problems this fixes:
          - Repeated characters: "GOOOOOOOOAL!" → "Goal!"
          - All-caps words:      "GOAL"          → "Goal"
          - Mixed e.g.:         "GOOOAL! FOUL!"  → "Goal! Foul!"

        Why here (TTS hook) and not the LLM hook:
          The LLM streams tokens one at a time, so a single word like "GOOOOAL"
          may arrive as multiple chunks. The TTS hook receives the assembled text
          stream, making it the right place for word-level normalization.
        """
        def normalize(text: str) -> str:
            # 1. Collapse 3+ repeated identical characters: "GOOOOAL" → "GOAL"
            text = re.sub(r"(.)\1{2,}", r"\1\1", text)
            # 2. Convert remaining all-caps words (3+ letters) to title case
            #    so TTS speaks them as natural words, not acronyms.
            text = re.sub(r"\b[A-Z]{3,}\b", lambda m: m.group(0).title(), text)
            return text

        async def normalized():
            async for chunk in text_stream:
                yield normalize(chunk)

        async for audio in run_tts(normalized()):
            yield audio

    session = AgentSession(agent=agent, pipeline=pipeline)
    shutdown_event = asyncio.Event()

    async def on_match_event(message):
        """Convert raw match event into live commentary via LLM + TTS."""
        event_data = message.get("message", "") if isinstance(message, dict) else str(message)
        if event_data.strip():
            normalized = event_data.strip().title()
            logging.info("Match event received: %s", normalized)
            await pipeline.process_text(f"Match event: {normalized}")

    def on_match_event_wrapper(message):
        asyncio.create_task(on_match_event(message))

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

        subscribe_config = PubSubSubscribeConfig(topic="CHAT", cb=on_match_event_wrapper)
        await ctx.room.subscribe_to_pubsub(subscribe_config)

        await session.start()
        await shutdown_event.wait()
    except KeyboardInterrupt:
        pass
    finally:
        await session.close()
        await ctx.shutdown()


def make_context() -> JobContext:
    room_options = RoomOptions(room_id="<room_id>", name="Live Sports Commentator", playground=True)
    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()
