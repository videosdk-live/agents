"""
Personal Assistant with Long-Term Memory (Mem0)
Remembers your name, preferences, and past conversations across sessions.
Env: VIDEOSDK_AUTH_TOKEN, DEEPGRAM_API_KEY, GOOGLE_API_KEY, CARTESIA_API_KEY, MEM0_API_KEY
"""

import os
import logging
import httpx
from videosdk.agents import Agent, AgentSession, Pipeline, WorkerJob, JobContext, RoomOptions
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.google import GoogleLLM
from videosdk.plugins.cartesia import CartesiaTTS
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
pre_download_model()

USER_ID = "YOUR_USER_ID"

class Mem0Memory:
    STORE_KEYWORDS = (
        "remember", "my name", "i like", "i dislike", "favorite",
        "i prefer", "i love", "i hate", "i'm", "i am", "i work",
    )

    def __init__(self, api_key: str, user_id: str):
        self.user_id = user_id
        self._client = httpx.AsyncClient(
            base_url="https://api.mem0.ai",
            headers={"Authorization": f"Token {api_key}", "Content-Type": "application/json"},
            timeout=10.0,
        )

    async def get_memories(self, limit: int = 5) -> list[str]:
        try:
            r = await self._client.get("/v1/memories/", params={"user_id": self.user_id})
            r.raise_for_status()
            entries = r.json() if isinstance(r.json(), list) else r.json().get("results", [])
            return [e.get("memory", "") for e in entries if isinstance(e, dict) and e.get("memory", "").strip()][:limit]
        except Exception as e:
            logger.error(f"Fetch memories failed: {e}")
            return []

    def should_store(self, text: str) -> bool:
        return any(kw in text.lower() for kw in self.STORE_KEYWORDS)

    async def search(self, query: str, top_k: int = 5) -> list[str]:
        """Search memories relevant to the user's current query."""
        try:
            r = await self._client.post(
                "/v1/memories/search/",
                json={"query": query, "user_id": self.user_id, "top_k": top_k},
            )
            r.raise_for_status()
            results = r.json() if isinstance(r.json(), list) else r.json().get("results", [])
            memories = [e.get("memory", "") for e in results if isinstance(e, dict) and e.get("memory", "").strip()]
            logger.info(f"Search found {len(memories)} relevant memories for: '{query[:50]}'")
            return memories
        except Exception as e:
            logger.error(f"Search memories failed: {e}")
            return []

    async def store(self, user_msg: str, assistant_msg: str | None = None):
        messages = [{"role": "user", "content": user_msg}]
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
        try:
            r = await self._client.post("/v1/memories/", json={"messages": messages, "user_id": self.user_id})
            r.raise_for_status()
            logger.info(f"Memory stored for {self.user_id}")
        except Exception as e:
            logger.error(f"Store memory failed: {e}")


class PersonalAssistant(Agent):
    def __init__(self, instructions: str, memories: list[str]):
        self._memories = memories
        super().__init__(instructions=instructions)

    async def on_enter(self):
        if self._memories:
            await self.session.say(f"Hey! Welcome back. How can I help you today?")
        else:
            await self.session.say("Hi there! I'm your personal assistant. Tell me about yourself so I can remember you next time!")

    async def on_exit(self):
        await self.session.say("Bye! I'll remember everything for next time.")


async def entrypoint(ctx: JobContext):
    mem0_key = os.getenv("MEM0_API_KEY")
    memory = Mem0Memory(api_key=mem0_key, user_id=USER_ID) if mem0_key else None

    base = (
        "You are a friendly personal assistant. You remember things users tell you "
        "like their name, preferences, and interests. Use what you know to make "
        "conversations feel personal. Keep responses short and conversational."
    )
    memories = await memory.get_memories() if memory else []
    if memories:
        facts = "\n".join(f"- {m}" for m in memories)
        instructions = f"{base}\n\nYou already know this about the user:\n{facts}"
    else:
        instructions = base

    agent = PersonalAssistant(instructions=instructions, memories=memories)

    pipeline = Pipeline(
        stt=DeepgramSTT(),
        llm=GoogleLLM(),
        tts=CartesiaTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
    )

    pending_msg = None

    @pipeline.on("user_turn_start")
    async def on_user(transcript: str):
        nonlocal pending_msg
        pending_msg = transcript
        if not memory:
            return

        relevant = await memory.search(transcript)
        if relevant:
            context = "\n".join(f"- {m}" for m in relevant)
            agent.chat_context.add_message(
                role="system",
                content=f"Relevant memories about this user:\n{context}\n\nUse these to answer personally.",
            )
            logger.info(f"Injected {len(relevant)} memories into context")

    @pipeline.on("llm")
    async def on_llm(data: dict):
        nonlocal pending_msg
        if not memory or not pending_msg:
            pending_msg = None
            return
        await memory.store(pending_msg, data.get("text", ""))
        pending_msg = None

    session = AgentSession(agent=agent, pipeline=pipeline)
    await session.start(wait_for_participant=True, run_until_shutdown=True)


def make_context() -> JobContext:
    return JobContext(
        room_options=RoomOptions(
            room_id="<room_id>",
            name="Personal Assistant",
            playground=True,
        )
    )


if __name__ == "__main__":
    WorkerJob(entrypoint=entrypoint, jobctx=make_context).start()