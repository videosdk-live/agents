import asyncio
from videosdk.agents import Agent, AgentSession, Pipeline, WorkerJob, JobContext, RoomOptions, Options, function_tool
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.google import GoogleLLM
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.cartesia import CartesiaTTS 
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])

from dotenv import load_dotenv
load_dotenv(override=True)

pre_download_model()

class VoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a helpful voice assistant that can answer questions. "
                "When the user asks to hang up, end the call, or stop the conversation, "
                "call the end_call function tool."
            )
        )
    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you today?")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")

    @function_tool
    async def end_call(self) -> dict:
        """End the call when the user asks to hang up, end the conversation, or says goodbye."""
        asyncio.create_task(self._announce_and_hangup())
        return {"status": "ending_call"}

    async def _announce_and_hangup(self) -> None:
        if not self.session:
            return
        self.session.interrupt()
        await asyncio.sleep(1)
        handle = await self.session.say("I am ending the call now.", interruptible=False)
        await handle
        await asyncio.sleep(1)
        await self.hangup()

async def entrypoint(ctx: JobContext):

    agent = VoiceAgent()

    pipeline=Pipeline(
        stt=DeepgramSTT(),
        llm=GoogleLLM(),
        tts=CartesiaTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector()
    )
    
    session = AgentSession(
        agent=agent,
        pipeline=pipeline
    )

    await session.start(wait_for_participant=True, run_until_shutdown=True)

def make_context() -> JobContext:
    room_options = RoomOptions(name="Agent Hangup Example", playground=True)
    return JobContext(room_options=room_options) 
 
if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context, options=Options(agent_id="YOUR_AGENT_ID", max_processes=2, register=True, host="localhost", port=8081))
    job.start()
