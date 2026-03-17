"""
Use Case: Meeting note-taker (Fireflies.ai style) — silent agent joins a meeting with 2 participants,
           listens via STT with speaker diarization, and every 2 minutes generates a structured
           meeting summary and publishes via PubSub.
Pipeline: DeepgramSTT (diarization) + SileroVAD + TurnDetector (STT-only, no LLM in pipeline)
LLM calls are made externally on a 2-minute interval to summarize accumulated transcript.
Env Vars: VIDEOSDK_AUTH_TOKEN, DEEPGRAM_API_KEY, GOOGLE_API_KEY
"""

import asyncio
import logging
import re
from datetime import datetime

from google.genai import types

from videosdk import PubSubPublishConfig
from videosdk.agents import Agent, AgentSession, Pipeline, WorkerJob, JobContext, RoomOptions, run_stt
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.google import GoogleLLM
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
pre_download_model()

NOTE_INTERVAL_SECONDS = 120

NOTES_SYSTEM_PROMPT = """You are a professional meeting note-taker.
Given a chunk of meeting transcript (with timestamps and speaker labels), produce a clear meeting summary.

Format your output EXACTLY like this:

MEETING NOTES
────────────────

TRANSCRIPT SUMMARY
[Write a 2-3 sentence overview of what was discussed in this segment]

KEY POINTS
- [Key point 1]
- [Key point 2]

DECISIONS MADE
- [Decision 1 — who decided, context]

ACTION ITEMS
- [Owner]: [Action item] [Deadline if mentioned]

OPEN QUESTIONS
- [Question that needs follow-up]

Rules:
- If the transcript is just greetings/small talk with nothing substantive, write a brief 1-line summary only.
- Always attribute statements to speakers when speaker labels are available (Speaker 0, Speaker 1, etc.)
- Be specific — include names, numbers, dates, and deadlines mentioned.
- Keep it concise but complete."""


transcript_buffer: list[str] = []
transcript_lock = asyncio.Lock()


class MeetingNotesAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are a silent meeting note-taker. Do NOT speak or respond.
            You are only here to listen and extract structured notes from the conversation.""",
        )

    async def on_enter(self) -> None:
        pass

    async def on_exit(self) -> None:
        pass


async def generate_notes_from_transcript(llm: GoogleLLM, transcript_chunk: str) -> str:
    """Call GoogleLLM directly to generate structured notes from a transcript chunk."""
    response = await llm._client.aio.models.generate_content(
        model=llm.model,
        contents=f"Meeting transcript:\n\n{transcript_chunk}",
        config=types.GenerateContentConfig(
            system_instruction=NOTES_SYSTEM_PROMPT,
            temperature=0.2,
        ),
    )
    return response.text.strip()


async def periodic_notes_task(ctx: JobContext, llm: GoogleLLM, stop_event: asyncio.Event):
    """Every NOTE_INTERVAL_SECONDS, drain the transcript buffer, generate notes, publish via PubSub."""
    logger.info("Periodic notes task started — interval=%ds", NOTE_INTERVAL_SECONDS)

    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=NOTE_INTERVAL_SECONDS)
            break 
        except asyncio.TimeoutError:
            pass

        async with transcript_lock:
            if not transcript_buffer:
                logger.info("[Notes] No new transcript in last %ds, skipping.", NOTE_INTERVAL_SECONDS)
                continue
            chunk = "\n".join(transcript_buffer)
            transcript_buffer.clear()

        logger.info("[Notes] Processing %d chars of transcript...", len(chunk))

        try:
            notes = await generate_notes_from_transcript(llm, chunk)
            if notes and notes != "[No notable items]":
                # Publish the full summary to CHAT topic
                publish_config = PubSubPublishConfig(
                    topic="CHAT",
                    message=notes,
                )
                await ctx.room.publish_to_pubsub(publish_config)
                logger.info("[Notes] Published to CHAT topic")
            else:
                logger.info("[Notes] Nothing notable in this interval.")
        except Exception as e:
            logger.error("[Notes] Failed to generate/publish notes: %s", e)

    logger.info("Periodic notes task stopped.")


async def entrypoint(ctx: JobContext):
    agent = MeetingNotesAgent()
    llm = GoogleLLM()

    pipeline = Pipeline(
        stt=DeepgramSTT(enable_diarization=True),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
    )

    session = AgentSession(agent=agent, pipeline=pipeline)
    shutdown_event = asyncio.Event()
    notes_stop_event = asyncio.Event()

    @pipeline.on("stt")
    async def stt_hook(audio_stream):
        """Clean filler words, capture speaker labels, buffer final transcripts."""
        async def audio_phase():
            async for audio in audio_stream:
                yield audio

        async for event in run_stt(audio_phase()):
            if event.data and event.data.text:
                text = event.data.text
                text = re.sub(r"\b(uh|um|hmm|like|you know)\b", "", text, flags=re.IGNORECASE)
                text = " ".join(text.split())
                if text:
                    event.data.text = text

                    speaker = event.metadata.get("speaker") if event.metadata else None
                    speaker_label = f"Speaker {speaker}" if speaker is not None else "Unknown"

                    logger.info("[STT] [%s] %s (type=%s)", speaker_label, text, event.event_type.value)

                    if event.event_type.value == "final_transcript":
                        async with transcript_lock:
                            transcript_buffer.append(
                                f"[{datetime.now().strftime('%H:%M:%S')}] [{speaker_label}] {text}"
                            )
            yield event

    async def cleanup_session():
        notes_stop_event.set()
        await session.close()
        shutdown_event.set()

    ctx.add_shutdown_callback(cleanup_session)

    def on_session_end(reason: str):
        asyncio.create_task(ctx.shutdown())

    try:
        await ctx.connect()
        ctx.room.setup_session_end_callback(on_session_end)
        await ctx.room.wait_for_participant()
        await session.start()

        notes_task = asyncio.create_task(periodic_notes_task(ctx, llm, notes_stop_event))

        await shutdown_event.wait()
        notes_task.cancel()
    except KeyboardInterrupt:
        pass
    finally:
        notes_stop_event.set()
        await session.close()
        await ctx.shutdown()


def make_context() -> JobContext:
    room_options = RoomOptions(room_id="<room_id>", name="Meeting Notes Agent", playground=True)
    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()
