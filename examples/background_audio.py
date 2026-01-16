from videosdk.agents import Agent, AgentSession,CascadingPipeline,WorkerJob, ConversationFlow, JobContext, RoomOptions, function_tool, RealTimePipeline, InterruptConfig, Options
from videosdk.plugins.openai import OpenAILLM,OpenAITTS
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
from videosdk.plugins.google import GoogleLLM
from videosdk.plugins.sarvamai import SarvamAITTS, SarvamAISTT
from videosdk.plugins.cartesia import CartesiaTTS
import logging
logging.basicConfig(level=logging.INFO)

# pre_download_model()

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
        # stt=SarvamAISTT(
        #     # model="sa:v2.5",
        #     language="unknown"
        # ),
        stt=DeepgramSTT(),
        llm=GoogleLLM(),
        tts=CartesiaTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
        interrupt_config=InterruptConfig(
            mode="HYBRID", # Interruption mode: 'VAD_ONLY' (voice activity), 'STT_ONLY' (speech-to-text), or 'HYBRID' (both).
            interrupt_min_duration=0.2, # Minimum continuous speech duration (VAD-based) to trigger an interruption.
            interrupt_min_words=2, # Minimum number of transcribed words (STT-based) to trigger an interruption.
            false_interrupt_pause_duration=2.0, # Duration to pause TTS for false interruption detection.
            resume_on_false_interrupt=True, # Automatically resume TTS after a false interruption timeout.
        )
    )
    session = AgentSession(
        agent=agent, 
        pipeline=pipeline,
        conversation_flow=conversation_flow
    )

    await session.start(wait_for_participant=True, run_until_shutdown=True)

def make_context() -> JobContext:
    room_options = RoomOptions(
        # room_id="e6bm-1c5g-ty1q", 
        name="VideoSDK's Cascading Agent", 
        playground=True, 
        )
    return JobContext(room_options=room_options)

if __name__ == "__main__":
    worker_options = Options(
        agent_id="new_arch_agent",
        register=True,
        max_processes=3,
        port=8000,
        host="localhost",
        log_level="INFO",
    )
    job = WorkerJob(
        entrypoint=entrypoint, 
        jobctx=make_context, 
        # options=worker_options
        )
    job.start()