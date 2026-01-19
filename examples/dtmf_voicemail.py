# This test script is used to test DTMF Event and Voicemail Detection and it's handling.
import logging
from videosdk.agents import Agent, AgentSession, Pipeline, DTMFHandler, VoiceMailDetector, WorkerJob, JobContext, RoomOptions, Options
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.openai import OpenAILLM
from videosdk.plugins.elevenlabs import ElevenLabsTTS
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model

pre_download_model()
class VoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful voice assistant that can answer questions."
        )
    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you today?")
    
    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")
        
async def entrypoint(ctx: JobContext):
    
    agent = VoiceAgent()

    pipeline=Pipeline(
        stt=DeepgramSTT(),
        llm=OpenAILLM(),
        tts=ElevenLabsTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector()
    )
    
    async def custom_callback_voicemail():
        print("Voice Mail detected, Shutting down the agent")
        await agent.hangup()

    async def custom_callback_dtmf(message):
        print("DTMF message received:", message)

    dtmf_handler = DTMFHandler(custom_callback_dtmf)
    voice_mail_detector = VoiceMailDetector(llm=OpenAILLM(), duration=5.0, callback = custom_callback_voicemail)

    session = AgentSession(
        agent=agent, 
        pipeline=pipeline,
        dtmf_handler = dtmf_handler,
        voice_mail_detector = voice_mail_detector
    )

    await session.start(wait_for_participant=True, run_until_shutdown=True)

def make_context() -> JobContext:
    room_options = RoomOptions(name="DTMF & Voicemail Agent", playground=True)
    return JobContext(room_options=room_options) 
 
if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context, options=Options(agent_id="YOUR_AGENT_ID", max_processes=2, register=True, host="localhost", port=8081))
    job.start()
