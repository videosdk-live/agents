from __future__ import annotations

import asyncio
import os
import logging
import traceback
from typing import Any, Optional, Literal, List
from dataclasses import dataclass, field
import numpy as np
from scipy import signal
from dotenv import load_dotenv
from videosdk.agents import (
    Agent,
    CustomAudioStreamTrack,
    RealtimeBaseModel,
    build_gemini_schema,
    is_function_tool,
    FunctionTool,
    get_tool_info,
    EncodeOptions,
    ResizeOptions,
    encode as encode_image,
)
import av
import time
from videosdk.agents.event_bus import global_event_emitter
from videosdk.agents import realtime_metrics_collector
from google import genai
from google.genai.live import AsyncSession
from google.genai.types import (
    Blob,
    Content,
    LiveConnectConfig,
    Modality,
    Part,
    PrebuiltVoiceConfig,
    SpeechConfig,
    VoiceConfig,
    FunctionResponse,
    Tool,
    GenerationConfig,
    AudioTranscriptionConfig,
)

load_dotenv()

logger = logging.getLogger(__name__)

AUDIO_SAMPLE_RATE = 48000


DEFAULT_IMAGE_ENCODE_OPTIONS = EncodeOptions(
    format="JPEG",
    quality=75,
    resize_options=ResizeOptions(width=1024, height=1024),
)

GeminiEventTypes = Literal["user_speech_started", "text_response", "error"]

Voice = Literal["Puck", "Charon", "Kore", "Fenrir", "Aoede"]


@dataclass
class GeminiLiveConfig:
    """Configuration for the Gemini Live API

    Args:
        voice: Voice ID for audio output. Options: 'Puck', 'Charon', 'Kore', 'Fenrir', 'Aoede'. Defaults to 'Puck'
        language_code: Language code for speech synthesis. Defaults to 'en-US'
        temperature: Controls randomness in response generation. Higher values (e.g. 0.8) make output more random,
                    lower values (e.g. 0.2) make it more focused. Defaults to None
        top_p: Nucleus sampling parameter. Controls diversity via cumulative probability cutoff. Range 0-1. Defaults to None
        top_k: Limits the number of tokens considered for each step of text generation. Defaults to None
        candidate_count: Number of response candidates to generate. Defaults to 1
        max_output_tokens: Maximum number of tokens allowed in model responses. Defaults to None
        presence_penalty: Penalizes tokens based on their presence in the text so far. Range -2.0 to 2.0. Defaults to None
        frequency_penalty: Penalizes tokens based on their frequency in the text so far. Range -2.0 to 2.0. Defaults to None
        response_modalities: List of enabled response types. Options: ["TEXT", "AUDIO"]. Defaults to both
        input_audio_transcription: Configuration for audio transcription features. Defaults to None
        output_audio_transcription: Configuration for audio transcription features. Defaults to None
    """

    voice: Voice | None = "Puck"
    language_code: str | None = "en-US"
    temperature: float | None = None
    top_p: float | None = None
    top_k: float | None = None
    candidate_count: int | None = 1
    max_output_tokens: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    response_modalities: List[Modality] | None = field(
        default_factory=lambda: ["TEXT", "AUDIO"]
    )
    input_audio_transcription: AudioTranscriptionConfig | None = field(
        default_factory=dict
    )
    output_audio_transcription: AudioTranscriptionConfig | None = field(
        default_factory=dict
    )


@dataclass
class GeminiSession:
    """Represents a Gemini Live API session"""

    session: AsyncSession
    session_cm: Any
    tasks: list[asyncio.Task]


class GeminiRealtime(RealtimeBaseModel[GeminiEventTypes]):
    """Gemini's realtime model for audio-only communication"""

    def __init__(
        self,
        *,
        model: str,
        config: GeminiLiveConfig | None = None,
        api_key: str | None = None,
        service_account_path: str | None = None,
    ) -> None:
        """
        Initialize Gemini realtime model.

        Args:
            model: The Gemini model identifier to use (e.g. 'gemini-pro', 'gemini-pro-vision')
            config: Optional configuration object for customizing model behavior. Contains settings for:
                   - voice: Voice ID for audio output ('Puck', 'Charon', 'Kore', 'Fenrir', 'Aoede'). Defaults to 'Puck'
                   - language_code: Language code for speech synthesis. Defaults to 'en-US'
                   - temperature: Controls randomness in responses. Higher values (0.8) more random, lower (0.2) more focused
                   - top_p: Nucleus sampling parameter. Controls diversity via probability cutoff. Range 0-1
                   - top_k: Limits number of tokens considered for each generation step
                   - candidate_count: Number of response candidates to generate. Defaults to 1
                   - max_output_tokens: Maximum tokens allowed in model responses
                   - presence_penalty: Penalizes token presence in text. Range -2.0 to 2.0
                   - frequency_penalty: Penalizes token frequency in text. Range -2.0 to 2.0
                   - response_modalities: List of enabled response types ["TEXT", "AUDIO"]. Defaults to both
                   - output_audio_transcription: Configuration for audio transcription features
            api_key: Gemini API key. If not provided, will attempt to read from GOOGLE_API_KEY env var
            service_account_path: Path to Google service account JSON file.

        Raises:
            ValueError: If neither api_key nor service_account_path is provided and no GOOGLE_API_KEY in env vars
        """
        super().__init__()
        self.model = model
        self._init_client(api_key, service_account_path)
        self._session: Optional[GeminiSession] = None
        self._closing = False
        self._session_should_close = asyncio.Event()
        self._main_task = None
        self.loop = None
        self.audio_track = None
        self._buffered_audio = bytearray()
        self._is_speaking = False
        self._last_audio_time = 0.0
        self._audio_processing_task = None
        self.tools = []
        self._instructions: str = (
            "You are a helpful voice assistant that can answer questions and help with tasks."
        )
        self.config: GeminiLiveConfig = config or GeminiLiveConfig()
        self.target_sample_rate = 24000
        self.input_sample_rate = 48000
        self._user_speaking = False
        self._agent_speaking = False

    def set_agent(self, agent: Agent) -> None:
        self._instructions = agent.instructions
        self.tools = agent.tools
        self.tools_formatted = self._convert_tools_to_gemini_format(self.tools)
        self.formatted_tools = self.tools_formatted

    def _init_client(self, api_key: str | None, service_account_path: str | None):
        if service_account_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_path
            self.client = genai.Client(http_options={"api_version": "v1beta"})
        else:
            self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
            if not self.api_key:
                self.emit("error", "GOOGLE_API_KEY or service account required")
                raise ValueError("GOOGLE_API_KEY or service account required")
            self.client = genai.Client(
                api_key=self.api_key, http_options={"api_version": "v1beta"}
            )

    async def connect(self) -> None:
        """Connect to the Gemini Live API"""
        if self._session:
            await self._cleanup_session(self._session)
            self._session = None

        self._closing = False
        self._session_should_close.clear()

        try:

            if (
                not self.audio_track
                and self.loop
                and "AUDIO" in self.config.response_modalities
            ):
                self.audio_track = CustomAudioStreamTrack(self.loop)
            elif not self.loop and "AUDIO" in self.config.response_modalities:
                self.emit(
                    "error", "Event loop not initialized. Audio playback will not work."
                )
                raise RuntimeError(
                    "Event loop not initialized. Audio playback will not work."
                )

            try:
                initial_session = await self._create_session()
                if initial_session:
                    self._session = initial_session
            except Exception as e:
                self.emit("error", f"Initial session creation failed, will retry: {e}")

            if not self._main_task or self._main_task.done():
                self._main_task = asyncio.create_task(
                    self._session_loop(), name="gemini-main-loop"
                )

        except Exception as e:
            self.emit("error", f"Error connecting to Gemini Live API: {e}")
            traceback.print_exc()
            raise

    async def _create_session(self) -> GeminiSession:
        """Create a new Gemini Live API session"""
        config = LiveConnectConfig(
            response_modalities=self.config.response_modalities,
            generation_config=GenerationConfig(
                candidate_count=(
                    self.config.candidate_count
                    if self.config.candidate_count is not None
                    else None
                ),
                temperature=(
                    self.config.temperature
                    if self.config.temperature is not None
                    else None
                ),
                top_p=self.config.top_p if self.config.top_p is not None else None,
                top_k=self.config.top_k if self.config.top_k is not None else None,
                max_output_tokens=(
                    self.config.max_output_tokens
                    if self.config.max_output_tokens is not None
                    else None
                ),
                presence_penalty=(
                    self.config.presence_penalty
                    if self.config.presence_penalty is not None
                    else None
                ),
                frequency_penalty=(
                    self.config.frequency_penalty
                    if self.config.frequency_penalty is not None
                    else None
                ),
            ),
            system_instruction=self._instructions,
            speech_config=SpeechConfig(
                voice_config=VoiceConfig(
                    prebuilt_voice_config=PrebuiltVoiceConfig(
                        voice_name=self.config.voice
                    )
                ),
                language_code=self.config.language_code,
            ),
            tools=self.formatted_tools or None,
            input_audio_transcription=self.config.input_audio_transcription,
            output_audio_transcription=self.config.output_audio_transcription,
        )
        try:
            session_cm = self.client.aio.live.connect(model=self.model, config=config)
            session = await session_cm.__aenter__()
            return GeminiSession(session=session, session_cm=session_cm, tasks=[])
        except Exception as e:
            self.emit("error", f"Connection error: {e}")
            traceback.print_exc()
            raise

    async def _session_loop(self) -> None:
        """Main processing loop for Gemini sessions"""
        reconnect_attempts = 0
        max_reconnect_attempts = 5
        reconnect_delay = 1

        while not self._closing:
            if not self._session:
                try:
                    self._session = await self._create_session()
                    reconnect_attempts = 0
                    reconnect_delay = 1
                except Exception as e:
                    reconnect_attempts += 1
                    reconnect_delay = min(30, reconnect_delay * 2)
                    self.emit(
                        "error",
                        f"session creation attempt {reconnect_attempts} failed: {e}",
                    )
                    if reconnect_attempts >= max_reconnect_attempts:
                        self.emit("error", "Max reconnection attempts reached")
                        break
                    await asyncio.sleep(reconnect_delay)
                    continue

            session = self._session

            recv_task = asyncio.create_task(
                self._receive_loop(session), name="gemini_receive"
            )
            keep_alive_task = asyncio.create_task(
                self._keep_alive(session), name="gemini_keepalive"
            )
            session.tasks.extend([recv_task, keep_alive_task])

            try:
                await self._session_should_close.wait()
            finally:
                for task in session.tasks:
                    if not task.done():
                        task.cancel()
                try:
                    await asyncio.gather(*session.tasks, return_exceptions=True)
                except Exception as e:
                    self.emit("error", f"Error during task cleanup: {e}")

            if not self._closing:
                await self._cleanup_session(session)
                self._session = None
                await asyncio.sleep(reconnect_delay)
                self._session_should_close.clear()

    async def _handle_tool_calls(
        self, response, active_response_id: str, accumulated_input_text: str
    ) -> str:
        """Handle tool calls from Gemini"""
        if not response.tool_call:
            return accumulated_input_text
        for tool_call in response.tool_call.function_calls:

            if self.tools:
                for tool in self.tools:
                    if not is_function_tool(tool):
                        continue
                    tool_info = get_tool_info(tool)
                    if tool_info.name == tool_call.name:
                        if accumulated_input_text:
                            await realtime_metrics_collector.set_user_transcript(
                                accumulated_input_text
                            )
                            accumulated_input_text = ""
                        try:
                            await realtime_metrics_collector.add_tool_call(
                                tool_info.name
                            )
                            result = await tool(**tool_call.args)
                            await self.send_tool_response(
                                [
                                    FunctionResponse(
                                        id=tool_call.id,
                                        name=tool_call.name,
                                        response=result,
                                    )
                                ]
                            )
                        except Exception as e:
                            self.emit(
                                "error",
                                f"Error executing function {tool_call.name}: {e}",
                            )
                            traceback.print_exc()
                        break
        return accumulated_input_text

    async def _receive_loop(self, session: GeminiSession) -> None:
        """Process incoming messages from Gemini"""
        try:
            active_response_id = None
            chunk_number = 0
            accumulated_text = ""
            final_transcription = ""
            accumulated_input_text = ""

            while not self._closing:
                try:
                    async for response in session.session.receive():
                        if self._closing:
                            break

                        if response.tool_call:
                            accumulated_input_text = await self._handle_tool_calls(
                                response, active_response_id, accumulated_input_text
                            )

                        if server_content := response.server_content:
                            if (
                                input_transcription := server_content.input_transcription
                            ):
                                if input_transcription.text:
                                    if not self._user_speaking:
                                        await realtime_metrics_collector.set_user_speech_start()
                                        self._user_speaking = True
                                    self.emit("user_speech_started", {"type": "done"})
                                    accumulated_input_text += input_transcription.text
                                    global_event_emitter.emit(
                                        "input_transcription",
                                        {
                                            "text": accumulated_input_text,
                                            "is_final": False,
                                        },
                                    )

                            if (
                                output_transcription := server_content.output_transcription
                            ):
                                if output_transcription.text:
                                    final_transcription += output_transcription.text
                                    global_event_emitter.emit(
                                        "output_transcription",
                                        {
                                            "text": final_transcription,
                                            "is_final": False,
                                        },
                                    )

                            if not active_response_id:
                                active_response_id = f"response_{id(response)}"
                                chunk_number = 0
                                accumulated_text = ""
                            if server_content.interrupted:
                                if active_response_id:
                                    active_response_id = None
                                    accumulated_text = ""
                                if (
                                    self.audio_track
                                    and "AUDIO" in self.config.response_modalities
                                ):
                                    self.audio_track.interrupt()
                                continue

                            if model_turn := server_content.model_turn:
                                if self._user_speaking:
                                    await realtime_metrics_collector.set_user_speech_end()
                                    self._user_speaking = False
                                if accumulated_input_text:
                                    await realtime_metrics_collector.set_user_transcript(
                                        accumulated_input_text
                                    )
                                    try:
                                        self.emit(
                                            "realtime_model_transcription",
                                            {
                                                "role": "user",
                                                "text": accumulated_input_text,
                                                "is_final": True,
                                            },
                                        )
                                    except Exception:
                                        pass
                                    accumulated_input_text = ""
                                for part in model_turn.parts:
                                    if (
                                        hasattr(part, "inline_data")
                                        and part.inline_data
                                    ):
                                        raw_audio = part.inline_data.data
                                        if not raw_audio or len(raw_audio) < 2:
                                            continue

                                        if "AUDIO" in self.config.response_modalities:
                                            chunk_number += 1
                                            if not self._agent_speaking:
                                                await realtime_metrics_collector.set_agent_speech_start()
                                                self._agent_speaking = True

                                            if self.audio_track and self.loop:
                                                if len(raw_audio) % 2 != 0:
                                                    raw_audio += b"\x00"

                                                asyncio.create_task(
                                                    self.audio_track.add_new_bytes(
                                                        raw_audio
                                                    ),
                                                    name=f"audio_chunk_{chunk_number}",
                                                )

                                    elif hasattr(part, "text") and part.text:
                                        accumulated_text += part.text

                            if server_content.turn_complete and active_response_id:
                                if accumulated_input_text:
                                    await realtime_metrics_collector.set_user_transcript(
                                        accumulated_input_text
                                    )
                                    accumulated_input_text = ""
                                if final_transcription:
                                    await realtime_metrics_collector.set_agent_response(
                                        final_transcription
                                    )
                                    try:
                                        self.emit(
                                            "realtime_model_transcription",
                                            {
                                                "role": "agent",
                                                "text": final_transcription,
                                                "is_final": True,
                                            },
                                        )
                                    except Exception:
                                        pass
                                if (
                                    "TEXT" in self.config.response_modalities
                                    and accumulated_text
                                ):
                                    global_event_emitter.emit(
                                        "text_response",
                                        {"type": "done", "text": accumulated_text},
                                    )
                                elif (
                                    "TEXT" not in self.config.response_modalities
                                    and final_transcription
                                ):
                                    global_event_emitter.emit(
                                        "text_response",
                                        {"type": "done", "text": final_transcription},
                                    )
                                active_response_id = None
                                accumulated_text = ""
                                final_transcription = ""
                                await realtime_metrics_collector.set_agent_speech_end(
                                    timeout=1.0
                                )
                                self._agent_speaking = False

                except Exception as e:
                    if "1000 (OK)" in str(e):
                        logger.info("Normal WebSocket closure")
                    else:
                        logger.error(f"Error in receive loop: {e}")
                        traceback.print_exc()

                    self._session_should_close.set()
                    break

                await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            self.emit("error", "Receive loop cancelled")
        except Exception as e:
            self.emit("error", e)
            traceback.print_exc()
            self._session_should_close.set()

    async def _keep_alive(self, session: GeminiSession) -> None:
        """Send periodic keep-alive messages"""
        try:
            while not self._closing:
                await asyncio.sleep(10)

                if self._closing:
                    break

                try:
                    await session.session.send_client_content(
                        turns=Content(parts=[Part(text=".")], role="user"),
                        turn_complete=False,
                    )
                except Exception as e:
                    if "closed" in str(e).lower():
                        self._session_should_close.set()
                        break
                    self.emit("error", f"Keep-alive error: {e}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in keep-alive: {e}")
            self._session_should_close.set()

    async def handle_audio_input(self, audio_data: bytes) -> None:
        """Handle incoming audio data from the user"""
        if not self._session or self._closing:
            return

        if "AUDIO" not in self.config.response_modalities:
            return

        audio_data = np.frombuffer(audio_data, dtype=np.int16)
        audio_data = signal.resample(
            audio_data,
            int(len(audio_data) * self.target_sample_rate / self.input_sample_rate),
        )
        audio_data = audio_data.astype(np.int16).tobytes()

        await self._session.session.send_realtime_input(
            audio=Blob(data=audio_data, mime_type=f"audio/pcm;rate={AUDIO_SAMPLE_RATE}")
        )

    async def handle_video_input(self, video_data: av.VideoFrame) -> None:
        """Improved video input handler with error prevention"""
        if not self._session or self._closing:
            return

        try:
            if not video_data or not video_data.planes:
                return

            now = time.monotonic()
            if (
                hasattr(self, "_last_video_frame")
                and (now - self._last_video_frame) < 0.5
            ):
                return
            self._last_video_frame = now

            processed_jpeg = encode_image(video_data, DEFAULT_IMAGE_ENCODE_OPTIONS)

            if not processed_jpeg or len(processed_jpeg) < 100:
                logger.warning("Invalid JPEG data generated")
                return

            await self._session.session.send_realtime_input(
                video=Blob(data=processed_jpeg, mime_type="image/jpeg")
            )
        except Exception as e:
            self.emit("error", f"Video processing error: {str(e)}")

    async def interrupt(self) -> None:
        """Interrupt current response"""
        if not self._session or self._closing:
            return

        try:
            await self._session.session.send_client_content(
                turns=Content(parts=[Part(text="stop")], role="user"),
                turn_complete=True,
            )
            await realtime_metrics_collector.set_interrupted()
            if self.audio_track and "AUDIO" in self.config.response_modalities:
                self.audio_track.interrupt()
        except Exception as e:
            self.emit("error", f"Interrupt error: {e}")

    async def send_message(self, message: str) -> None:
        """Send a text message to get audio response"""
        retry_count = 0
        max_retries = 5
        while not self._session or not self._session.session:
            if retry_count >= max_retries:
                raise RuntimeError("No active Gemini session after maximum retries")
            logger.debug("No active session, waiting for connection...")
            await asyncio.sleep(1)
            retry_count += 1

        try:
            await self._session.session.send_client_content(
                turns=[
                    Content(
                        parts=[
                            Part(
                                text="Repeat the user's exact message back to them [DO NOT ADD ANYTHING ELSE]:"
                                + message
                            )
                        ],
                        role="model",
                    ),
                    Content(parts=[Part(text=".")], role="user"),
                ],
                turn_complete=True,
            )
            await asyncio.sleep(0.1)
        except Exception as e:
            self.emit("error", f"Error sending message: {e}")
            self._session_should_close.set()

    async def send_text_message(self, message: str) -> None:
        """Send a text message for text-only communication"""
        retry_count = 0
        max_retries = 5
        while not self._session or not self._session.session:
            if retry_count >= max_retries:
                raise RuntimeError("No active Gemini session after maximum retries")
            self.emit("error", "No active session, waiting for connection...")
            await asyncio.sleep(1)
            retry_count += 1

        try:
            await self._session.session.send_client_content(
                turns=Content(parts=[Part(text=message)], role="user"),
                turn_complete=True,
            )
        except Exception as e:
            self.emit("error", f"Error sending text message: {e}")
            self._session_should_close.set()

    async def _cleanup_session(self, session: GeminiSession) -> None:
        """Clean up a session's resources"""
        for task in session.tasks:
            if not task.done():
                task.cancel()

        try:
            await session.session_cm.__aexit__(None, None, None)
        except Exception as e:
            self.emit("error", f"Error closing session: {e}")

    async def aclose(self) -> None:
        """Clean up all resources"""
        if self._closing:
            return

        self._closing = True
        self._session_should_close.set()

        if self._audio_processing_task and not self._audio_processing_task.done():
            self._audio_processing_task.cancel()
            try:
                await asyncio.wait_for(self._audio_processing_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        if self._main_task and not self._main_task.done():
            self._main_task.cancel()
            try:
                await asyncio.wait_for(self._main_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        if self._session:
            await self._cleanup_session(self._session)
            self._session = None

        if hasattr(self.audio_track, "cleanup") and self.audio_track:
            try:
                await self.audio_track.cleanup()
            except Exception as e:
                self.emit("error", f"Error cleaning up audio track: {e}")

        self._buffered_audio = bytearray()

    async def _reconnect(self) -> None:
        if self._session:
            await self._cleanup_session(self._session)
            self._session = None
        self._session = await self._create_session()

    async def send_tool_response(
        self, function_responses: List[FunctionResponse]
    ) -> None:
        """Send tool responses back to Gemini"""
        if not self._session or not self._session.session:
            return

        try:
            await self._session.session.send_tool_response(
                function_responses=function_responses
            )
        except Exception as e:
            self.emit("error", f"Error sending tool response: {e}")
            self._session_should_close.set()

    def _convert_tools_to_gemini_format(self, tools: List[FunctionTool]) -> List[Tool]:
        """Convert tool definitions to Gemini's Tool format"""
        function_declarations = []

        for tool in tools:
            if not is_function_tool(tool):
                continue

            try:
                function_declaration = build_gemini_schema(tool)
                function_declarations.append(function_declaration)
            except Exception as e:
                self.emit("error", f"Failed to format tool {tool}: {e}")
                continue
        return (
            [Tool(function_declarations=function_declarations)]
            if function_declarations
            else []
        )
