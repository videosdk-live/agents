from videosdk.agents import Agent, AgentSession,CascadingPipeline,WorkerJob, ConversationFlow, JobContext, RoomOptions,BackgroundAudioConfig
from videosdk.plugins.openai import OpenAILLM,OpenAITTS
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model

pre_download_model()

class VoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful voice assistant that can answer questions and help with tasks.",
        )
        
    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you today?")
    
    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")

async def entrypoint(ctx: JobContext):
    
    agent = VoiceAgent()
    conversation_flow = ConversationFlow(agent)

    pipeline = CascadingPipeline(
        stt=DeepgramSTT(),
        llm=OpenAILLM(),
        tts=OpenAITTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector()
    )
    session = AgentSession(
        agent=agent, 
        pipeline=pipeline,
        conversation_flow=conversation_flow,
        background_audio=BackgroundAudioConfig(
            file_path="./agent_keyboard.wav"
        ),
    )

    await ctx.run_until_shutdown(session=session,wait_for_participant=True)

def make_context() -> JobContext:
    room_options = RoomOptions(room_id="<room_id>",name="Background Audio Agent", playground=True)
    
    return JobContext(
        room_options=room_options
        )

if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()