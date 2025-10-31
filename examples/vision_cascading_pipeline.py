# This test script is used to test vision functionaly cascading pipeline.
from typing import Any, Dict
from videosdk.agents import Agent, AgentSession, CascadingPipeline, function_tool, WorkerJob, ConversationFlow, JobContext, RoomOptions
from videosdk.plugins.google import GoogleLLM
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
from videosdk.plugins.elevenlabs import ElevenLabsTTS
from videosdk.plugins.rnnoise import RNNoise

pre_download_model()

class VoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a voice assistant with vision. When user asks visual questions "
                "(look, see, describe, have a look, what am I holding, what do you see), "
                "call the vision function_tool to capture fresh frame and describe what you see. "
                "Don't mention frame capture — just answer directly from the image."
                "For non-visual questions, respond normally."
            ) 
        )
        
    async def on_enter(self) -> None:
        await self.session.say("Hello, I am Vision AI. How can I help you today?")
    
    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")

    @function_tool
    async def capture_frame(self) -> Dict[str, Any]:
        """Capture and process a frame from the user's camera."""
        return await self.capture_and_process_frame()

async def entrypoint(ctx: JobContext):

    agent = VoiceAgent()
    conversation_flow = ConversationFlow(agent)

    pipeline = CascadingPipeline(
        stt= DeepgramSTT(),
        llm=GoogleLLM(),
        tts=ElevenLabsTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
        denoise=RNNoise()
    )
    session = AgentSession(
        agent=agent, 
        pipeline=pipeline,
        conversation_flow=conversation_flow
    )

    await ctx.run_until_shutdown(session=session,wait_for_participant=True)

def make_context() -> JobContext:
    room_options = RoomOptions(
        room_id="<room_id>",
        name="Sandbox Agent", 
        playground=True,
        vision=True
    )
    
    return JobContext(
        room_options=room_options
        )

if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()
