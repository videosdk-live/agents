import logging
from videosdk.agents import Agent, AgentSession, CascadingPipeline, WorkerJob,ConversationFlow, JobContext, RoomOptions, EOUConfig, InterruptConfig
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
from videosdk.plugins.google import GoogleLLM
from videosdk.plugins.cartesia import CartesiaTTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
pre_download_model()

class VoiceAgent(Agent):
    def __init__(self):
        super().__init__(instructions="You are a helpful voice assistant that can answer questions and help with tasks.")
        
    async def on_enter(self) -> None:
       await self.session.say("This example script showcases advanced cascading pipeline features, including interruptible speech this message can not be interrupted.",interruptible=False)
        
    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")

async def entrypoint(ctx: JobContext):
    
    agent = VoiceAgent()
    conversation_flow = ConversationFlow()

    pipeline = CascadingPipeline(
        stt= DeepgramSTT(),
        llm=GoogleLLM(),
        tts=CartesiaTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
        eou_config=EOUConfig(
            mode='ADAPTIVE', # EOU mode: 'DEFAULT' uses fixed min/max timeouts; 'ADAPTIVE' computes a continuous timeout based on confidence.
            min_max_speech_wait_timeout=[0.5, 0.8], # Tuple (min_duration, max_duration) for EOU.
        ),
        interrupt_config=InterruptConfig(
            mode="HYBRID", # Interruption mode: 'VAD_ONLY' (voice activity), 'STT_ONLY' (speech-to-text), or 'HYBRID' (both).
            interrupt_min_duration=0.2, # Minimum continuous speech duration (VAD-based) to trigger an interruption.
            interrupt_min_words=2, # Minimum number of transcribed words (STT-based) to trigger an interruption.
            false_interrupt_pause_duration=2.0, # Duration to pause TTS for false interruption detection.
            resume_on_false_interrupt=True, # Automatically resume TTS after a false interruption timeout.
        )
    )
    session = AgentSession(agent=agent,pipeline=pipeline,conversation_flow=conversation_flow)

    await session.start(wait_for_participant=True, run_until_shutdown=True)

def make_context() -> JobContext:
    room_options = RoomOptions(room_id="<room_id>", name="Adavence Config Agent", playground=True)
    
    return JobContext(room_options=room_options)

if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()
