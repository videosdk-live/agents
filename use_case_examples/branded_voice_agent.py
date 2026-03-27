"""
Use Case: Premium retail shopping assistant (Luxe) for Prestige Boutique — custom brand voice.
Pipeline: P4 — XAIRealtime (STT + LLM) + CartesiaTTS (external brand voice)
Demonstrates: Hybrid TTS mode — realtime model handles understanding, external TTS renders the brand voice.
Env Vars: VIDEOSDK_AUTH_TOKEN, XAI_API_KEY, CARTESIA_API_KEY
"""

import logging
from videosdk.agents import Agent, AgentSession, Pipeline, WorkerJob, JobContext, RoomOptions
from videosdk.plugins.xai import XAIRealtime, XAIRealtimeConfig, XAITurnDetection
from videosdk.plugins.cartesia import CartesiaTTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])


class BoutiqueShoppingAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are Luxe, the virtual shopping assistant for Prestige Boutique, a luxury fashion retailer.
            Help visitors with product recommendations, sizing, styling advice, and availability.
            Use warm, refined, and sophisticated language. Never use casual slang or filler words.
            Keep responses elegant and concise — no more than three sentences per reply.
            If asked about pricing, always frame it positively (e.g., 'starting at' rather than 'costs').
            For out-of-stock items, suggest alternatives gracefully.""",
        )

    async def on_enter(self) -> None:
        await self.session.say(
            "Welcome to Prestige Boutique. I'm Luxe, your personal shopping assistant. "
            "Whether you're looking for the perfect evening gown or a refined everyday piece, "
            "I'm here to guide you. What brings you in today?"
        )

    async def on_exit(self) -> None:
        await self.session.say(
            "It's been a pleasure assisting you. We look forward to your next visit at Prestige Boutique."
        )


async def entrypoint(ctx: JobContext):
    # XAI Realtime handles both STT and LLM; CartesiaTTS renders the brand voice.
    llm = XAIRealtime(
        model="grok-4-1-fast-non-reasoning",
        config=XAIRealtimeConfig(
            voice="Eve",
            turn_detection=XAITurnDetection(
                type="server_vad",
                threshold=0.5,
                prefix_padding_ms=300,
                silence_duration_ms=300,
            ),
        ),
    )

    pipeline = Pipeline(
        llm=llm,
        tts=CartesiaTTS(),
    )

    agent = BoutiqueShoppingAgent()
    session = AgentSession(agent=agent, pipeline=pipeline)
    await session.start(wait_for_participant=True, run_until_shutdown=True)


def make_context() -> JobContext:
    room_options = RoomOptions(room_id="<room_id>", name="Prestige Boutique - Luxe", playground=True)
    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()
