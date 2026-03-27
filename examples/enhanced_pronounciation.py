import os
import re
from typing import Optional
from videosdk.agents import Agent, AgentSession, Pipeline, WorkerJob, JobContext, RoomOptions, run_tts
from videosdk.plugins.openai import OpenAILLM
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
from videosdk.plugins.elevenlabs import ElevenLabsTTS
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])

pre_download_model()

class VoiceAgent(Agent):
    def __init__(self, ctx: Optional[JobContext] = None):
        super().__init__(
            instructions="You are a helpful voice assistant that can answer questions about technology."
        )
        
    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you today?")
    
    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")


async def entrypoint(ctx: JobContext):
    agent = VoiceAgent()
    
    # Create pipeline with new Pipeline class
    pipeline = Pipeline(
        stt=DeepgramSTT(api_key=os.getenv("DEEPGRAM_API_KEY")),
        llm=OpenAILLM(api_key=os.getenv("OPENAI_API_KEY")),
        tts=ElevenLabsTTS(api_key=os.getenv("ELEVENLABS_API_KEY")),
        vad=SileroVAD(),
        turn_detector=TurnDetector(threshold=0.8)
    )
    
    # Pronunciation map for technical terms
    pronunciation_map = {
        "nginx": "engine x",
        "URL": "U R L",
        "API": "A P I",
        "VideoSDK": "Video SDK",
        "HTTP": "H T T P",
        "HTTPS": "H T T P S",
        "JSON": "J SON",
        "SQL": "sequel",
        "AWS": "A W S",
        "CI/CD": "C I C D",
    }
    
    @pipeline.on("tts")
    async def tts_node(text_stream):
        """
        Unified TTS hook

        Covers:
        - agent_response → yields text
        - speech_out     → yields audio frames

        This hook fixes pronunciation of technical terms before TTS synthesis.
        """

        async def text_phase():
            async for response in text_stream:
                processed = response

                for word, pronunciation in pronunciation_map.items():
                    processed = re.sub(
                        rf"\b{word}\b",
                        pronunciation,
                        processed,
                        flags=re.IGNORECASE,
                    )

                if processed != response:
                    print("[Pronunciation] Fixed terms in response")

                yield processed

        async for audio_chunk in run_tts(text_phase()):
            yield audio_chunk
    
    session = AgentSession(
        agent=agent, 
        pipeline=pipeline,
    )

    await session.start(wait_for_participant=True, run_until_shutdown=True)

def make_context() -> JobContext:
    room_options = RoomOptions(room_id="<room_id>", name="Pronunciation Agent", playground=True)
    return JobContext(room_options=room_options)

if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()
