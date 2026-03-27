"""
Use Case: SaaS product support agent backed by a documentation knowledge base (Novu platform).
Pipeline: P1 — SarvamAISTT + GoogleLLM + GoogleTTS + SileroVAD + TurnDetector
Demonstrates: Custom KnowledgeBase subclass with allow_retrieval, pre_process_query, format_context.
Env Vars: VIDEOSDK_AUTH_TOKEN, SARVAMAI_API_KEY, GOOGLE_API_KEY, KNOWLEDGE_BASE_ID
"""

import os
import logging
from typing import List
from videosdk.agents import (
    Agent, AgentSession, Pipeline, WorkerJob, JobContext, RoomOptions,
    KnowledgeBase, KnowledgeBaseConfig,
)
from videosdk.plugins.sarvamai import SarvamAISTT
from videosdk.plugins.google import GoogleLLM, GoogleTTS
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)
pre_download_model()


class NovuKnowledgeBase(KnowledgeBase):
    """Knowledge base for Novu platform documentation retrieval."""

    TRIGGER_PHRASES = [
        "how do i", "how to", "what is", "explain", "tell me about",
        "show me", "help with", "configure", "set up", "integrate",
    ]

    def allow_retrieval(self, transcript: str) -> bool:
        """Only retrieve docs when the user asks a product-specific question."""
        transcript_lower = transcript.lower()
        allowed = any(phrase in transcript_lower for phrase in self.TRIGGER_PHRASES)
        logger.info("KB retrieval %s for: %s", "allowed" if allowed else "skipped", transcript[:60])
        return allowed

    def pre_process_query(self, transcript: str) -> str:
        """Strip trigger phrases to produce a clean search query."""
        query = transcript.lower()
        for phrase in self.TRIGGER_PHRASES:
            query = query.replace(phrase, "").strip()
        return query or transcript

    def format_context(self, documents: List[str]) -> str:
        """Format retrieved docs into context for the LLM."""
        if not documents:
            return ""
        formatted = "\n\n".join(f"• {doc}" for doc in documents)
        return (
            "The following is retrieved from the Novu documentation. "
            "Use it to answer the user's question accurately.\n\n"
            f"{formatted}\n"
        )


class NovuSupportAgent(Agent):
    def __init__(self):
        kb_id = os.getenv("KNOWLEDGE_BASE_ID")
        if not kb_id:
            raise ValueError("KNOWLEDGE_BASE_ID environment variable is not set.")

        super().__init__(
            instructions="""You are the support agent for Novu, an open-source notification infrastructure platform.
            Answer questions about integrations, API usage, workflows, subscribers, and account management.
            Always rely on the retrieved knowledge base context when answering product questions.
            If the knowledge base does not contain the answer, say so clearly and offer to raise a support ticket.
            Do not make up feature names or API parameters. Be precise and developer-friendly.""",
            knowledge_base=NovuKnowledgeBase(KnowledgeBaseConfig(id=kb_id, top_k=3)),
        )

    async def on_enter(self) -> None:
        await self.session.say(
            "Hi, welcome to Novu Support. I'm your product assistant. "
            "I can help with API usage, integrations, workflow setup, and account questions. "
            "What can I help you with today?"
        )

    async def on_exit(self) -> None:
        await self.session.say(
            "Thanks for reaching out to Novu Support. "
            "Check out docs.novu.co for more resources. Goodbye!"
        )


async def entrypoint(ctx: JobContext):
    agent = NovuSupportAgent()

    pipeline = Pipeline(
        stt=SarvamAISTT(),
        llm=GoogleLLM(),
        tts=GoogleTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
    )
    session = AgentSession(agent=agent, pipeline=pipeline)
    await session.start(wait_for_participant=True, run_until_shutdown=True)


def make_context() -> JobContext:
    room_options = RoomOptions(room_id="<room_id>", name="Novu Platform Support", playground=True)
    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()
