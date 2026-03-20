"""
Use Case: Voice assistant with custom RAG — retrieves from a local ChromaDB vector store.
Pipeline: P1 — DeepgramSTT + OpenAILLM + ElevenLabsTTS + SileroVAD + TurnDetector
Demonstrates: ChromaDB vector retrieval, user_turn_start hook for context injection,
              llm hook for observing generated responses.
Env Vars: VIDEOSDK_AUTH_TOKEN, DEEPGRAM_API_KEY, OPENAI_API_KEY, ELEVENLABS_API_KEY
"""

import os
import asyncio
import logging
import chromadb
from openai import OpenAI, AsyncOpenAI
from videosdk.agents import Agent, AgentSession, Pipeline, WorkerJob, JobContext, RoomOptions
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.openai import OpenAILLM
from videosdk.plugins.elevenlabs import ElevenLabsTTS
from videosdk.plugins.cartesia import CartesiaTTS
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
pre_download_model()


class RAGVoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a helpful voice assistant that answers questions based on "
                "provided context. Use the retrieved documents to ground your answers. "
                "If no relevant context is found, say so. Be concise and conversational."
            )
        )

        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        
        self.documents = [
            "What is VideoSDK? VideoSDK is a comprehensive real-time communication platform that provides APIs and SDKs for video calling, live streaming, and AI-powered voice agents.",
            "How do I authenticate with VideoSDK? Use JWT tokens generated with your API key and secret from the VideoSDK dashboard. Set the token as the VIDEOSDK_AUTH_TOKEN environment variable.",
            "How do I build voice agents with VideoSDK? You can build voice agents by installing the Python library: pip install videosdk-agents. It supports Cascading, Realtime, and Hybrid modes. Visit https://www.videosdk.live/ for more information.",
            "What is a Pipeline in VideoSDK Agents? A Pipeline is a unified component that automatically detects the best mode (Cascading, Realtime, or Hybrid) based on the components you provide.",
            "If a user's question is related to VideoSDK and the answer is unknown, direct them to https://www.videosdk.live/ for more information."
        ]

        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(name="rag_docs")
        self._initialize_knowledge_base()


    def _get_embedding_sync(self, text: str) -> list[float]:
        """Synchronous embedding for initialization."""
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.embeddings.create(input=text, model="text-embedding-ada-002")
        return response.data[0].embedding

    async def get_embedding(self, text: str) -> list[float]:
        """Async embedding for runtime queries."""
        response = await self.openai_client.embeddings.create(
            input=text, model="text-embedding-ada-002"
        )
        return response.data[0].embedding


    def _initialize_knowledge_base(self):
        """Generate embeddings and store documents in ChromaDB."""
        logger.info(f"Initializing knowledge base with {len(self.documents)} documents...")
        embeddings = [self._get_embedding_sync(doc) for doc in self.documents]
        self.collection.add(
            documents=self.documents,
            embeddings=embeddings,
            ids=[f"doc_{i}" for i in range(len(self.documents))],
        )
        logger.info("Knowledge base initialized.")


    async def retrieve(self, query: str, k: int = 2) -> list[str]:
        """Retrieve top-k most relevant documents from the vector store."""
        query_embedding = await self.get_embedding(query)
        results = self.collection.query(query_embeddings=[query_embedding], n_results=k)
        docs = results["documents"][0] if results["documents"] else []
        logger.info(f"Retrieved {len(docs)} documents for query: '{query[:60]}...'")
        return docs

    async def on_enter(self) -> None:
        await self.session.say(
            "Hello! I'm your VideoSDK assistant powered by a local knowledge base. "
            "Ask me anything about VideoSDK."
        )

    async def on_exit(self) -> None:
        await self.session.say("Thank you for using VideoSDK. Goodbye!")

async def entrypoint(ctx: JobContext):
    agent = RAGVoiceAgent()

    pipeline = Pipeline(
        stt=DeepgramSTT(),
        llm=OpenAILLM(),
        tts=CartesiaTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
    )

    # --- RAG hook: retrieve docs and inject context before LLM runs ---
    @pipeline.on("user_turn_start")
    async def on_user_turn_start(transcript: str):
        """
        Fires when the user's final transcript is ready, BEFORE the LLM is called.
        Retrieve relevant documents and inject them into chat_context so the LLM
        can use them to generate a grounded response.
        """
        context_docs = await agent.retrieve(transcript)

        if context_docs:
            context_str = "\n\n".join(
                f"Document {i + 1}: {doc}" for i, doc in enumerate(context_docs)
            )
            agent.chat_context.add_message(
                role="system",
                content=(
                    f"Retrieved Context:\n{context_str}\n\n"
                    "Use this context to answer the user's question."
                ),
            )
            logger.info(f"Injected {len(context_docs)} docs into chat context")
        else:
            logger.info("No relevant documents found for this query")

    @pipeline.on("llm")
    async def on_llm(data: dict):
        text = data.get("text", "")
        logger.info(f"[LLM] Generated ({len(text)} chars): {text[:120]}...")

    session = AgentSession(agent=agent, pipeline=pipeline)
    await session.start(wait_for_participant=True, run_until_shutdown=True)


def make_context() -> JobContext:
    return JobContext(
        room_options=RoomOptions(
            room_id="<room_id>",
            name="RAG Voice Assistant",
            playground=True,
        )
    )


if __name__ == "__main__":
    WorkerJob(entrypoint=entrypoint, jobctx=make_context).start()