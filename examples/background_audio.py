from videosdk.agents import Agent, AgentSession,CascadingPipeline,WorkerJob, ConversationFlow, JobContext, RoomOptions, function_tool, RealTimePipeline
from videosdk.plugins.openai import OpenAILLM,OpenAITTS
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
import logging
logging.basicConfig(level=logging.INFO)

pre_download_model()

class VoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful voice assistant that can answer questions and help with tasks. If the user asks to play music, use the control_background_music tool with action 'play'. To stop, use the action 'stop'.",
        )
        self.set_thinking_audio()
        
    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you today?")
    
    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")

    @function_tool
    async def control_background_music(self, action: str):
        """
        Controls the background music. Call this tool to play or stop music.
        :param action: 'play' to start the music, 'stop' to end it.
        """
        if action.lower() == "play":
            await self.play_background_audio(override_thinking=True, looping=True)
            return "Background music started."
        elif action.lower() == "stop":
            await self.stop_background_audio()
            return "Background music stopped."
        else:
            return "Invalid action. Please use 'play' or 'stop'."

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
    )

    await session.start(wait_for_participant=True, run_until_shutdown=True)

def make_context() -> JobContext:
    room_options = RoomOptions(room_id="<room_id>",name="Background Audio Agent", playground=True, background_audio=True)
    
    return JobContext(
        room_options=room_options
        )

if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()