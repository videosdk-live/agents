import logging
import re
from videosdk.agents import Agent,AgentSession,Pipeline,WorkerJob,JobContext,RoomOptions,run_stt,run_tts
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
from videosdk.plugins.google import GoogleLLM
from videosdk.plugins.cartesia import CartesiaTTS

logging.basicConfig(level=logging.INFO)
pre_download_model()

class VoiceAgent(Agent):
    def __init__(self):
        super().__init__(instructions="You are a helpful voice assistant.")

    async def on_enter(self):
        await self.session.say("Hello! How can I help you today?")

    async def on_exit(self):
        await self.session.say("Goodbye!")


async def entrypoint(ctx: JobContext):
    agent = VoiceAgent()

    pipeline = Pipeline(
        stt=DeepgramSTT(),
        llm=GoogleLLM(),
        tts=CartesiaTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
    )

    @pipeline.on("stt")
    async def stt_hook(audio_stream):
        """
        - Preprocess audio before STT
        - Normalize transcript after STT
        """

        async def audio_phase():
            async for audio in audio_stream:
                if len(audio) < 300:
                    continue
                yield audio

        async for event in run_stt(audio_phase()):
            if event.data and event.data.text:
                text = event.data.text.lower()
                text = re.sub(r"\b(uh|um|like)\b", "", text)

                replacements = {
                    "working hours": "office hours",
                    "timing": "office hours",
                }
                for src, dst in replacements.items():
                    text = re.sub(rf"\b{src}\b", dst, text)

                event.data.text = " ".join(text.split())
                logging.info(f"[STT] {event.data.text}")

            yield event

    @pipeline.on("user_turn_start")
    async def on_user_turn_start(transcript: str):
        logging.info(f"[USER TURN START] {transcript}")

    @pipeline.on("user_turn_end")
    async def on_user_turn_end():
        logging.info("[USER TURN END]")

    @pipeline.on("agent_turn_start")
    async def on_agent_turn_start():
        logging.info("[AGENT TURN START]")

    @pipeline.on("agent_turn_end")
    async def on_agent_turn_end():
        logging.info("[AGENT TURN END]")

    @pipeline.on("tts")
    async def tts_hook(text_stream):
        """
        Final speech formatting
        """
        async def preprocess_text():
            async for text in text_stream:
                yield text.replace("AM", "A M").replace("PM", "P M")

        async for audio in run_tts(preprocess_text()):
            yield audio

    @pipeline.on("llm")
    async def llm_gate_decision_making(transcript: str):
        """
        Control LLM invocation.
        - Yield → bypass LLM
        - No yield → normal LLM flow
        """
        transcript_lower = transcript.lower()
        normalized = transcript_lower.replace(" ", "")

        TRIGGERS = ("officehours",)

        if any(trigger in normalized for trigger in TRIGGERS):
            response = (
                "Our online support is available twenty four seven. "
                "Store office working hours are nine A M to five P M, "
                "Monday to Friday."
            )

            for word in response.split():
                yield word + " "

    @pipeline.on("vision_frame")
    async def vision_hook(frame_stream):
        async for frame in frame_stream:
            yield frame  # passthrough

    session = AgentSession(agent=agent, pipeline=pipeline)
    await session.start(wait_for_participant=True, run_until_shutdown=True)


def make_context() -> JobContext:
    return JobContext(
        room_options=RoomOptions(
            room_id="<room_id>",
            name="Pipeline Hooks Example",
            playground=True,
        )
    )


if __name__ == "__main__":
    WorkerJob(entrypoint=entrypoint, jobctx=make_context()).start()