from videosdk.agents import Agent, AgentSession, Pipeline, JobContext, RoomOptions, WorkerJob
from videosdk.plugins.google import GeminiRealtime, GeminiLiveConfig
import asyncio
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])

FAREWELL_PHRASES = ("goodbye", "bye!", "have a nice day", "have a good day", "talk to you later")

class MyVoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a helpful voice assistant that can answer questions. "
                "When the user asks to hang up, end the call, or stop the conversation, "
            )
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you today?")
    
    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")

async def start_session(context: JobContext):
    agent = MyVoiceAgent()
    model = GeminiRealtime(
        model="gemini-3.1-flash-live-preview",
        # When GOOGLE_API_KEY is set in .env - DON'T pass api_key parameter
        # api_key="AIXXXXXXXXXXXXXXXXXXXX", 
        config=GeminiLiveConfig(
            voice="Leda", # Puck, Charon, Kore, Fenrir, Aoede, Leda, Orus, and Zephyr.
            response_modalities=["AUDIO"]
        )
    )
    
    pipeline = Pipeline(llm=model)
    session = AgentSession(
        agent=agent,
        pipeline=pipeline
    )
    
    @pipeline.on("user_turn_start")
    async def on_user_turn_start(transcript: str):
        logging.info(f"[USER TURN START] {transcript}")
        
    @pipeline.on("agent_turn_start")
    async def on_agent_turn_start():
        logging.info("[AGENT TURN START]")

    @pipeline.on("agent_turn_end")
    async def on_agent_turn_end():
        logging.info("[AGENT TURN END]")

    @pipeline.on("user_turn_end")
    async def on_user_turn_end():
        logging.info("[USER TURN END]")     
        
    @pipeline.on("llm")
    async def on_agent_text(data: dict):
        print(f"agent said: {data['text']}")
        text = (data.get("text") or "").lower()
        logging.info(f"[LLM] Generated {text[:100]}...")
        if not any(phrase in text for phrase in FAREWELL_PHRASES):
            return
        logging.info("Farewell detected — will hang up after TTS playback")
        handle = agent.session.current_utterance

        async def _hangup_after_playback():
            if handle and not handle.done():
                await handle
            await agent.hangup()

        asyncio.create_task(_hangup_after_playback())
        
    await session.start(wait_for_participant=True, run_until_shutdown=True)

def make_context() -> JobContext:
    room_options = RoomOptions(
        room_id="<room_id>", # Replace it with your actual room_id
        name="Gemini Realtime Agent",
        playground=True,
    )

    return JobContext(room_options=room_options)

if __name__ == "__main__":
    job = WorkerJob(entrypoint=start_session, jobctx=make_context)
    job.start()