import logging
import os
from typing import List
from videosdk.agents import Agent,AgentSession,CascadingPipeline,ConversationFlow,JobContext,RoomOptions,WorkerJob,KnowledgeBase,KnowledgeBaseConfig
from videosdk.plugins.google import GoogleLLM, GoogleTTS
from videosdk.plugins.sarvamai import SarvamAISTT
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
pre_download_model()

class CustomKnowledgeBase(KnowledgeBase):
    """
    Custom knowledge base handler to demonstrate overriding retrieval logic.
    """

    TRIGGER_PHRASES = ["search for", "look up", "what do you know about"]

    def allow_retrieval(self, transcript: str) -> bool:
        """
        Only allow retrieval if the transcript contains a trigger phrase.
        """
        logger.info(f"Checking if transcript '{transcript}' allows retrieval...")
        for phrase in self.TRIGGER_PHRASES:
            if phrase in transcript.lower():
                logger.info("Retrieval allowed.")
                return True
        logger.info("Retrieval not allowed.")
        return False

    def pre_process_query(self, transcript: str) -> str:
        """
        Remove the trigger phrase from the transcript to create a clean query.
        """
        logger.info(f"Pre-processing query: '{transcript}'")
        for phrase in self.TRIGGER_PHRASES:
            if phrase in transcript.lower():
                query = transcript.lower().replace(phrase, "", 1).strip()
                logger.info(f"Processed query: '{query}'")
                return query
        return transcript

    def format_context(self, documents: List[str]) -> str:
        """
        Format retrieved documents into a context string for the LLM.
        """
        logger.info(f"Formatting context for {len(documents)} documents.")

        if not documents:
            return ""

        formatted_docs = "\n\n".join([f"• {doc}" for doc in documents])

        return (
            "The following information was retrieved from the knowledge base. "
            "Use it when answering the user's question.\n\n"
            f"{formatted_docs}\n"
        )

class VoiceAgent(Agent):
    def __init__(self):
        kb_id = os.getenv("KNOWLEDGE_BASE_ID")
        if not kb_id:
            raise ValueError("KNOWLEDGE_BASE_ID environment variable not set.")

        # Initialize Knowledge Base configuration inside the agent
        config = KnowledgeBaseConfig(
            id=kb_id,
            top_k=3,
        )
        super().__init__(
            instructions="You are a helpful voice assistant that can answer questions and help with tasks.",
            knowledge_base=CustomKnowledgeBase(config),
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you today?")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")

async def entrypoint(ctx: JobContext):
    agent = VoiceAgent()
    conversation_flow = ConversationFlow(agent)
    pipeline = CascadingPipeline(
        stt=SarvamAISTT(),
        llm=GoogleLLM(),
        tts=GoogleTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
    )
    session = AgentSession(agent=agent, pipeline=pipeline, conversation_flow=conversation_flow)
    await session.start(wait_for_participant=True, run_until_shutdown=True)

def make_context() -> JobContext:
    room_options = RoomOptions(
        room_id="<room_id>", name="Sandbox Agent", playground=True
    )

    return JobContext(room_options=room_options)

if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()
