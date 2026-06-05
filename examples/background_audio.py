from videosdk.agents import Agent, AgentSession, Pipeline, WorkerJob, JobContext, RoomOptions, function_tool
from videosdk.agents.plugins import OpenAILLM, OpenAITTS, DeepgramSTT, SileroVAD, TurnDetector, pre_download_model
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])

pre_download_model()

class VoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful voice assistant that can answer questions and help with tasks. If the user asks to play music, use the control_background_music tool with action 'play'. To stop, use the action 'stop'.",
        )
        # Thinking audio plays while the agent is generating a response.
        # Any libav-decodable file is supported: WAV, MP3, Ogg/Vorbis, Ogg/Opus, FLAC, M4A/AAC, ...
        # Leave `file` unset to use the SDK's default `agent-keyboard.ogg`.
        self.set_thinking_audio(
            # file="path/to/your_thinking_sound.mp3",
            volume=0.3,
        )

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
            # Background music plays on-demand and can loop.
            # Any libav-decodable file is supported: WAV, MP3, Ogg/Vorbis, Ogg/Opus, FLAC, M4A/AAC, ...
            # Leave `file` unset to use the SDK's default `office-noise.ogg`.
            # override_thinking=True  -> thinking audio layers over the music during LLM generation.
            # override_thinking=False -> music is exclusive; thinking audio is suppressed while it plays.
            await self.play_background_audio(
                # file="path/to/your_background_music.mp3",
                volume=0.8,
                looping=True,
                override_thinking=False,
            )
            return "Background music started."
        elif action.lower() == "stop":
            await self.stop_background_audio()
            return "Background music stopped."
        else:
            return "Invalid action. Please use 'play' or 'stop'."

async def entrypoint(ctx: JobContext):
    
    agent = VoiceAgent()

    pipeline = Pipeline(
        stt=DeepgramSTT(),
        llm=OpenAILLM(),
        tts=OpenAITTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector()
    )

    session = AgentSession(
        agent=agent, 
        pipeline=pipeline
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