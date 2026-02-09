import logging
import os
from typing import List
from videosdk.agents import Agent, AgentSession, Pipeline, JobContext, RoomOptions, WorkerJob, KnowledgeBase, KnowledgeBaseConfig
from videosdk.plugins.google import GeminiRealtime, GeminiLiveConfig
from videosdk.plugins.sarvamai import SarvamAISTT
from videosdk.plugins.silero import SileroVAD

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

class RealtimeCustomKnowledgeBase(KnowledgeBase):
    """
    Custom knowledge base handler to demonstrate KB integration with realtime models.
    """

    TRIGGER_PHRASES = ["tell me about","Context Engineering"]

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


class AddtionalSTTAndRealtime(Agent):
    """Voice agent with knowledge base support in hybrid_stt mode"""
    
    def __init__(self):
        kb_id = os.getenv("KNOWLEDGE_BASE_ID")
        if not kb_id:
            raise ValueError("KNOWLEDGE_BASE_ID environment variable not set.")

        config = KnowledgeBaseConfig(
            id=kb_id,
            top_k=3,
        )
        super().__init__(
            instructions="You are a helpful voice assistant with access to a knowledge base. When users ask questions, use the provided context from the knowledge base to give accurate answers.",
            knowledge_base=RealtimeCustomKnowledgeBase(config),
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello! I'm your voice assistant with knowledge base support. Ask me anything!")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")


async def entrypoint(ctx: JobContext):
    llm = GeminiRealtime(
        model="gemini-2.5-flash-native-audio-preview-12-2025",
        config=GeminiLiveConfig(
            voice="Puck",  
            response_modalities=["AUDIO"],  
        )
    )
    
    pipeline = Pipeline(
        stt=SarvamAISTT(),         
        llm=llm,                  
        vad=SileroVAD(),           
    )
    
    agent = AddtionalSTTAndRealtime()
    session = AgentSession(agent=agent, pipeline=pipeline)
    await session.start(wait_for_participant=True, run_until_shutdown=True)


def make_context() -> JobContext:
    room_options = RoomOptions(room_id="<room_id>",name="STT+Realtime Agent",playground=True)
    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()
