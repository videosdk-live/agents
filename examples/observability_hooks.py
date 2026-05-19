
from videosdk.agents import Agent,AgentSession, Pipeline, JobContext, RoomOptions, WorkerJob, ObservabilityOptions, RecordingOptions, LoggingOptions
from videosdk.plugins.google import GoogleLLM
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.cartesia import CartesiaTTS
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
from videosdk.plugins.openai import OpenAIRealtime, OpenAIRealtimeConfig
from openai.types.beta.realtime.session import InputAudioTranscription, TurnDetection

pre_download_model()
class MyVoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are VideoSDK's Voice Agent. You are a helpful voice assistant that can answer questions.",
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you today?")

    async def on_exit(self) -> None:
        history = self.session.get_context_history(
            include_function_calls=True,
            include_system_messages=False,
        )
        print("\n\n=== SESSION END: CONTEXT HISTORY ===")
        for msg in history:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            if isinstance(content, list):
                text_blocks = [c if isinstance(c, str) else "[Image/Other]" for c in content]
                content = " ".join(text_blocks)
            print(f"{role}: {content}")
        print("===================================\n")
        await self.session.say("Goodbye!")



async def start_session(context: JobContext):

    agent = MyVoiceAgent()
    
    # model = OpenAIRealtime(
    #     model="gpt-realtime-2025-08-28",
    #     # When OPENAI_API_KEY is set in .env - DON'T pass api_key parameter
    #     # api_key="sk-proj-XXXXXXXXXXXXXXXXXXXX",
    #     config=OpenAIRealtimeConfig(
    #         voice="alloy", # alloy, ash, ballad, coral, echo, fable, onyx, nova, sage, shimmer, and verse
    #         modalities=["text", "audio"],
    #         input_audio_transcription=InputAudioTranscription(model="whisper-1"),
    #         turn_detection=TurnDetection(
    #             type="server_vad",
    #             threshold=0.5,
    #             prefix_padding_ms=300,
    #             silence_duration_ms=200,
    #         ),
    #         tool_choice="auto",
    #     )
    # )

    pipeline = Pipeline(
        stt=DeepgramSTT(),
        llm=GoogleLLM(),
        tts=CartesiaTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector()
        # llm=model
    )
        
    @pipeline.on("error")
    def on_pipeline_error(data):
        """
        Catch any errors from STT, LLM, TTS, VAD, TURN-D, 
        or VideoSDK Room Connection.
        """
        source = data.get("source", "unknown")
        error = data.get("error", "No error details")
        print(f"\n[ERROR HOOK] Pipeline Error from {source}: {error}\n")

    @pipeline.on("recording_started")
    def on_recording_started(data):
        """Fired when participant or track recording starts successfully."""
        print(f"\n[RECORDING HOOK] Started: {data}\n")

    @pipeline.on("recording_stopped")
    def on_recording_stopped(data):
        """Fired when recording stops successfully."""
        print(f"\n[RECORDING HOOK] Stopped: {data}\n")

    @pipeline.on("recording_failed")
    def on_recording_failed(data):
        """Fired when recording fails to start or stop."""
        print(f"\n[RECORDING HOOK] Failed: {data}\n")

    @pipeline.metrics.on("stt")
    def on_stt_metrics(metrics: dict):
        """Fired when STT turn completes, includes latency, tokens, etc."""
        print(f"[METRICS] STT Latency: {metrics.get('stt_latency')}ms")

    @pipeline.metrics.on("llm")
    def on_llm_metrics(metrics: dict):
        """Fired when LLM generation completes, includes TTFT, latency, tokens."""
        print(f"[METRICS] LLM TTFT: {metrics.get('llm_ttft')}ms | Total Duration: {metrics.get('llm_duration')}ms")
        print(f"[METRICS] LLM Tokens (P/C/T): {metrics.get('prompt_tokens')}/{metrics.get('completion_tokens')}/{metrics.get('total_tokens')}")

    @pipeline.metrics.on("tts")
    def on_tts_metrics(metrics: dict):
        """Fired when TTS finishes speaking."""
        print(f"[METRICS] TTS TTFB: {metrics.get('ttfb')}ms | Total Latency: {metrics.get('tts_latency')}ms ")

    @pipeline.metrics.on("eou")
    def on_eou_metrics(metrics: dict):
        """Fired when TurnDetector matches end-of-utterance."""
        print(f"[METRICS] EOU Latency: {metrics.get('eou_latency')}ms")

    # @pipeline.metrics.on("realtime")
    # def on_realtime_metrics(metrics: dict):
    #     """Fired for realtime (speech-to-speech) models like Gemini Live / OpenAI Realtime."""
    #     print(
    #         "[METRICS] Realtime "
    #         f"TTFB: {metrics.get('realtime_ttfb')}ms | "
    #         f"Tokens (in/out/total): {metrics.get('realtime_input_tokens')}/{metrics.get('realtime_output_tokens')}/{metrics.get('realtime_total_tokens')} | "
    #         f"TextTokens (in/out): {metrics.get('realtime_input_text_tokens')}/{metrics.get('realtime_output_text_tokens')} | "
    #         f"AudioTokens (in/out): {metrics.get('realtime_input_audio_tokens')}/{metrics.get('realtime_output_audio_tokens')}"
    #     )

    session = AgentSession(
        agent=agent,
        pipeline=pipeline,
    )

    await session.start(
        wait_for_participant=True,
        run_until_shutdown=True,
        observability=ObservabilityOptions(
            recording=RecordingOptions(),                    
            logs=LoggingOptions(level=["INFO", "DEBUG"]),  
        ),
    )


def make_context() -> JobContext:
    room_options = RoomOptions(
        room_id="<room_id>",
        name="Observability Hooks",
        playground=True,
    )

    return JobContext(room_options=room_options)

if __name__ == "__main__":
    job = WorkerJob(entrypoint=start_session, jobctx=make_context)
    job.start()
