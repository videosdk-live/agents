import logging
from videosdk.agents import Agent, AgentSession, Pipeline, function_tool, WorkerJob, JobContext, RoomOptions, Options
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.google import GoogleLLM
from videosdk.plugins.cartesia import CartesiaTTS
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
pre_download_model()


class CallTransferAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are the Call Transfer Agent Which Help and provide to transfer on going call to new number. use transfer_call tool to transfer the call to new number.",
        )
        
    async def on_enter(self) -> None:
        await self.session.say("Hello Buddy, How can I help you today?")
    
    async def on_exit(self) -> None:
        await self.session.say("Goodbye Buddy, Thank you for calling!")

    @function_tool
    async def transfer_call(self) -> None:
        """Transfer the call to Provided number"""
        token = os.getenv("VIDEOSDK_AUTH_TOKEN")
        transfer_to = os.getenv("CALL_TRANSFER_TO") 
        return await self.session.call_transfer(token,transfer_to)

async def entrypoint(ctx: JobContext):
    
    agent = CallTransferAgent()

    pipeline = Pipeline(
        stt=DeepgramSTT(),
        llm=GoogleLLM(),
        tts=CartesiaTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector()
    )
    session = AgentSession(
        agent=agent, 
        pipeline=pipeline,
    )

    await session.start(wait_for_participant=True, run_until_shutdown=True)

def make_context() -> JobContext:
    room_options = RoomOptions(name="Call Transfer Agent", playground=True)
    
    return JobContext(room_options=room_options)

if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context,options=Options(agent_id="YOUR_AGENT_ID",register=True,host="localhost",port=8081))
    job.start()
