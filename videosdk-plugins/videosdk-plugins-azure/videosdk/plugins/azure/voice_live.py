from __future__ import annotations

import asyncio
import os
import logging
import traceback
import base64
from typing import Any, Optional, Literal, List, Union, Dict
from dataclasses import dataclass, field
import numpy as np
from scipy import signal
from dotenv import load_dotenv

from videosdk.agents import (
    Agent,
    CustomAudioStreamTrack,
    RealtimeBaseModel,
    realtime_metrics_collector,
)
from videosdk.agents.event_bus import global_event_emitter

from azure.core.credentials import AzureKeyCredential, TokenCredential
from azure.identity import DefaultAzureCredential
from azure.ai.voicelive.aio import connect
from azure.ai.voicelive.models import (
    RequestSession,
    ServerVad,
    AzureStandardVoice,
    Modality,
    AudioFormat,
    ServerEventType,
)

load_dotenv()

logger = logging.getLogger(__name__)

AZURE_VOICE_LIVE_SAMPLE_RATE = 24000
VIDEOSDK_INPUT_SAMPLE_RATE = 48000

AzureVoiceLiveEventTypes = Literal["user_speech_started", "text_response", "error"]


@dataclass
class AzureVoiceLiveConfig:
    """Configuration for Azure Voice Live API (Beta)

    Args:
        voice: Voice ID for audio output. Can be Azure voice (e.g., 'en-US-AvaNeural') or OpenAI voice ('alloy', 'echo', etc.). Default is 'en-US-AvaNeural'
        modalities: List of enabled response types. Options: [Modality.TEXT, Modality.AUDIO]. Default includes both
        input_audio_format: Audio format for input. Default is AudioFormat.PCM16
        output_audio_format: Audio format for output. Default is AudioFormat.PCM16
        turn_detection_threshold: Voice activity detection threshold (0.0-1.0). Default is 0.5
        turn_detection_prefix_padding_ms: Padding before speech start (ms). Default is 300
        turn_detection_silence_duration_ms: Silence duration to mark end (ms). Default is 500
        temperature: Controls randomness in response generation. Higher values make output more random. Default is None
        max_completion_tokens: Maximum number of tokens in response. Default is None
    """

    voice: str = "en-US-AvaNeural"
    modalities: List[Modality] = field(
        default_factory=lambda: [Modality.TEXT, Modality.AUDIO]
    )
    input_audio_format: AudioFormat = AudioFormat.PCM16
    output_audio_format: AudioFormat = AudioFormat.PCM16
    turn_detection_threshold: float = 0.5
    turn_detection_prefix_padding_ms: int = 300
    turn_detection_silence_duration_ms: int = 500
    temperature: Optional[float] = None
    max_completion_tokens: Optional[int] = None


@dataclass
class AzureVoiceLiveSession:
    """Represents an Azure Voice Live session"""

    connection: Any
    session_id: Optional[str] = None
    tasks: list[asyncio.Task] = field(default_factory=list)


class AzureVoiceLive(RealtimeBaseModel[AzureVoiceLiveEventTypes]):
    """Azure Voice Live realtime model implementation"""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        endpoint: str | None = None,
        model: str,
        config: AzureVoiceLiveConfig | None = None,
        credential: Union[AzureKeyCredential, TokenCredential] | None = None,
    ) -> None:
        """
        Initialize Azure Voice Live realtime model.

        Args:
            api_key: Azure Voice Live API key. If not provided, will attempt to read from AZURE_VOICE_LIVE_API_KEY env var
            endpoint: Azure Voice Live endpoint. If not provided, will attempt to read from AZURE_VOICE_LIVE_ENDPOINT env var
            model: The model identifier to use (e.g., 'gpt-4o-realtime-preview')
            config: Optional configuration object for customizing model behavior. Contains settings for:
                   - voice: Voice ID for audio output (Azure or OpenAI voices)
                   - modalities: List of enabled response types [TEXT, AUDIO]
                   - turn_detection: Voice activity detection settings
                   - temperature: Response randomness control
            credential: Azure credential object. If provided, takes precedence over api_key

        Raises:
            ValueError: If no API key or credential is provided and none found in environment variables
        """
        super().__init__()
        self.model = model
        self.endpoint = endpoint or os.getenv(
            "AZURE_VOICE_LIVE_ENDPOINT", "wss://api.voicelive.com/v1"
        )

        if credential:
            self.credential = credential
        elif api_key:
            self.credential = AzureKeyCredential(api_key)
        else:
            env_api_key = os.getenv("AZURE_VOICE_LIVE_API_KEY")
            if env_api_key:
                self.credential = AzureKeyCredential(env_api_key)
            else:
                try:
                    self.credential = DefaultAzureCredential()
                except Exception:
                    self.emit("error", "Azure Voice Live credentials required")
                    raise ValueError(
                        "Azure Voice Live credentials required. Provide api_key, credential, or set AZURE_VOICE_LIVE_API_KEY environment variable"
                    )

        self._session: Optional[AzureVoiceLiveSession] = None
        self._closing = False
        self._instructions: str = (
            "You are a helpful voice assistant that can answer questions and help with tasks."
        )
        self.loop = None
        self.audio_track: Optional[CustomAudioStreamTrack] = None
        self.config: AzureVoiceLiveConfig = config or AzureVoiceLiveConfig()
        self.input_sample_rate = VIDEOSDK_INPUT_SAMPLE_RATE
        self.target_sample_rate = AZURE_VOICE_LIVE_SAMPLE_RATE
        self._agent_speaking = False
        self._user_speaking = False
        self.session_ready = False
        self._session_ready_event = asyncio.Event()

    def set_agent(self, agent: Agent) -> None:
        """Set the agent configuration"""
        self._instructions = agent.instructions

    async def connect(self) -> None:
        """Connect to Azure Voice Live API"""
        if self._session:
            await self._cleanup_session(self._session)
            self._session = None

        self._closing = False

        try:
            if (
                not self.audio_track
                and self.loop
                and Modality.AUDIO in self.config.modalities
            ):
                self.audio_track = CustomAudioStreamTrack(self.loop)
            elif not self.loop and Modality.AUDIO in self.config.modalities:
                self.emit(
                    "error", "Event loop not initialized. Audio playback will not work."
                )
                raise RuntimeError(
                    "Event loop not initialized. Audio playback will not work."
                )

            session = await self._create_session()
            if session:
                self._session = session

            if self._session:
                asyncio.create_task(
                    self._process_events(), name="azure-voice-live-events"
                )
                try:
                    logger.info("Waiting for Azure Voice Live session to be ready...")
                    await asyncio.wait_for(
                        self._session_ready_event.wait(), timeout=10.0
                    )
                    logger.info("Azure Voice Live session is ready.")
                except asyncio.TimeoutError:
                    self.emit("error", "Azure Voice Live session ready timeout")
                    raise RuntimeError(
                        "Azure Voice Live session did not become ready in time"
                    )

        except Exception as e:
            self.emit("error", f"Error connecting to Azure Voice Live API: {e}")
            traceback.print_exc()
            raise

    async def _create_session(self) -> AzureVoiceLiveSession:
        """Create a new Azure Voice Live session"""
        try:
            connection_cm = connect(
                endpoint=self.endpoint,
                credential=self.credential,
                model=self.model,
                connection_options={
                    "max_msg_size": 10 * 1024 * 1024,
                    "heartbeat": 20,
                    "timeout": 20,
                },
            )

            connection = await connection_cm.__aenter__()

            await self._setup_session(connection)

            return AzureVoiceLiveSession(
                connection=connection, session_id=None, tasks=[]
            )

        except Exception as e:
            self.emit("error", f"Failed to create Azure Voice Live session: {e}")
            traceback.print_exc()
            raise

    async def _setup_session(self, connection) -> None:
        """Configure the Azure Voice Live session"""
        logger.info("Setting up Azure Voice Live session...")

        voice_config: Union[AzureStandardVoice, str]
        if (
            self.config.voice.startswith("en-US-")
            or self.config.voice.startswith("en-CA-")
            or "-" in self.config.voice
        ):
            voice_config = AzureStandardVoice(
                name=self.config.voice, type="azure-standard"
            )
        else:
            voice_config = self.config.voice

        turn_detection_config = ServerVad(
            threshold=self.config.turn_detection_threshold,
            prefix_padding_ms=self.config.turn_detection_prefix_padding_ms,
            silence_duration_ms=self.config.turn_detection_silence_duration_ms,
        )

        session_config = RequestSession(
            modalities=self.config.modalities,
            instructions=self._instructions,
            voice=voice_config,
            input_audio_format=self.config.input_audio_format,
            output_audio_format=self.config.output_audio_format,
            turn_detection=turn_detection_config,
        )

        if self.config.temperature is not None:
            session_config.temperature = self.config.temperature
        if self.config.max_completion_tokens is not None:
            session_config.max_completion_tokens = self.config.max_completion_tokens

        await connection.session.update(session=session_config)
        logger.info("Azure Voice Live session configuration sent")

    async def _process_events(self) -> None:
        """Process events from the Azure Voice Live connection"""
        try:
            if not self._session or not self._session.connection:
                return

            async for event in self._session.connection:
                if self._closing:
                    break
                await self._handle_event(event)

        except asyncio.CancelledError:
            logger.info("Event processing cancelled")
        except Exception as e:
            self.emit("error", f"Error processing events: {e}")
            traceback.print_exc()

    async def _handle_event(self, event) -> None:
        """Handle different types of events from Azure Voice Live"""
        try:
            logger.debug(f"Received event: {event.type}")

            if event.type == ServerEventType.SESSION_UPDATED:
                logger.info(f"Session ready: {event.session.id}")
                if self._session:
                    self._session.session_id = event.session.id
                self.session_ready = True
                self._session_ready_event.set()

            elif event.type == ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STARTED:
                logger.info("User started speaking")
                if not self._user_speaking:
                    await realtime_metrics_collector.set_user_speech_start()
                    self._user_speaking = True
                self.emit("user_speech_started", {"type": "done"})

                if self.audio_track and Modality.AUDIO in self.config.modalities:
                    self.audio_track.interrupt()

                if self._session and self._session.connection:
                    try:
                        await self._session.connection.response.cancel()
                    except Exception as e:
                        logger.debug(f"No response to cancel: {e}")

            elif event.type == ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STOPPED:
                logger.info("User stopped speaking")
                if self._user_speaking:
                    await realtime_metrics_collector.set_user_speech_end()
                    self._user_speaking = False

            elif event.type == ServerEventType.RESPONSE_CREATED:
                logger.info("Assistant response created")

            elif event.type == ServerEventType.RESPONSE_AUDIO_DELTA:
                logger.debug("Received audio delta")
                if Modality.AUDIO in self.config.modalities:
                    if not self._agent_speaking:
                        await realtime_metrics_collector.set_agent_speech_start()
                        self._agent_speaking = True

                    if self.audio_track and self.loop:
                        asyncio.create_task(self.audio_track.add_new_bytes(event.delta))

            elif event.type == ServerEventType.RESPONSE_AUDIO_DONE:
                logger.info("Assistant finished speaking")
                if self._agent_speaking:
                    await realtime_metrics_collector.set_agent_speech_end(timeout=1.0)
                    self._agent_speaking = False

            elif event.type == ServerEventType.RESPONSE_TEXT_DELTA:
                if hasattr(self, "_current_text_response"):
                    self._current_text_response += event.delta
                else:
                    self._current_text_response = event.delta

            elif event.type == ServerEventType.RESPONSE_TEXT_DONE:
                if hasattr(self, "_current_text_response"):
                    global_event_emitter.emit(
                        "text_response",
                        {"text": self._current_text_response, "type": "done"},
                    )
                    await realtime_metrics_collector.set_agent_response(
                        self._current_text_response
                    )
                    try:
                        self.emit(
                            "realtime_model_transcription",
                            {
                                "role": "agent",
                                "text": self._current_text_response,
                                "is_final": True,
                            },
                        )
                    except Exception:
                        pass
                    self._current_text_response = ""

            elif event.type == ServerEventType.RESPONSE_DONE:
                logger.info("Response complete")
                if self._agent_speaking:
                    await realtime_metrics_collector.set_agent_speech_end(timeout=1.0)
                    self._agent_speaking = False

            elif event.type == ServerEventType.ERROR:
                logger.error(f"Azure Voice Live error: {event.error.message}")
                self.emit("error", f"Azure Voice Live error: {event.error.message}")

            elif event.type == ServerEventType.CONVERSATION_ITEM_CREATED:
                logger.debug(f"Conversation item created: {event.item.id}")

                if (
                    hasattr(event.item, "content")
                    and event.item.content
                    and hasattr(event.item.content[0], "transcript")
                ):
                    transcript = event.item.content[0].transcript
                    if transcript and event.item.role == "user":
                        await realtime_metrics_collector.set_user_transcript(transcript)
                        try:
                            self.emit(
                                "realtime_model_transcription",
                                {"role": "user", "text": transcript, "is_final": True},
                            )
                        except Exception:
                            pass

            else:
                logger.debug(f"Unhandled event type: {event.type}")

        except Exception as e:
            self.emit("error", f"Error handling event {event.type}: {e}")
            traceback.print_exc()

    async def handle_audio_input(self, audio_data: bytes) -> None:
        """Handle incoming audio data from the user"""
        if not self._session or self._closing or not self.session_ready:
            return

        if Modality.AUDIO not in self.config.modalities:
            return

        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            if len(audio_array) % 2 == 0:
                audio_array = audio_array.reshape(-1, 2)
                audio_array = np.mean(audio_array, axis=1).astype(np.int16)

            target_length = int(
                len(audio_array) * self.target_sample_rate / self.input_sample_rate
            )
            resampled_float = signal.resample(
                audio_array.astype(np.float32), target_length
            )
            resampled_int16 = np.clip(resampled_float, -32768, 32767).astype(np.int16)
            resampled_bytes = resampled_int16.tobytes()

            encoded_audio = base64.b64encode(resampled_bytes).decode("utf-8")

            await self._session.connection.input_audio_buffer.append(
                audio=encoded_audio
            )

        except Exception as e:
            self.emit("error", f"Error processing audio input: {e}")

    async def interrupt(self) -> None:
        """Interrupt current response"""
        if not self._session or self._closing:
            return

        try:
            if self._session.connection:
                await self._session.connection.response.cancel()

            if self.audio_track and Modality.AUDIO in self.config.modalities:
                self.audio_track.interrupt()

            await realtime_metrics_collector.set_interrupted()

            if self._agent_speaking:
                await realtime_metrics_collector.set_agent_speech_end(timeout=1.0)
                self._agent_speaking = False

        except Exception as e:
            self.emit("error", f"Interrupt error: {e}")

    async def send_message(self, message: str) -> None:
        """Send a text message to get audio response"""
        retry_count = 0
        max_retries = 5
        while not self._session or not self.session_ready:
            if retry_count >= max_retries:
                raise RuntimeError(
                    "No active Azure Voice Live session after maximum retries"
                )
            logger.debug("No active session, waiting for connection...")
            await asyncio.sleep(1)
            retry_count += 1

        try:
            await self._session.connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Repeat the user's exact message back to them [DO NOT ADD ANYTHING ELSE]: {message}",
                        }
                    ],
                }
            )

            await self._session.connection.response.create()

        except Exception as e:
            self.emit("error", f"Error sending message: {e}")

    async def send_text_message(self, message: str) -> None:
        """Send a text message for text-only communication"""
        retry_count = 0
        max_retries = 5
        while not self._session or not self.session_ready:
            if retry_count >= max_retries:
                raise RuntimeError(
                    "No active Azure Voice Live session after maximum retries"
                )
            logger.debug("No active session, waiting for connection...")
            await asyncio.sleep(1)
            retry_count += 1

        try:
            await self._session.connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": message}],
                }
            )

            await self._session.connection.response.create()

        except Exception as e:
            self.emit("error", f"Error sending text message: {e}")

    async def _cleanup_session(self, session: AzureVoiceLiveSession) -> None:
        """Clean up session resources"""
        for task in session.tasks:
            if not task.done():
                task.cancel()

        try:
            if session.connection:
                if hasattr(session.connection, "close"):
                    await session.connection.close()
        except Exception as e:
            self.emit("error", f"Error closing session: {e}")

    async def aclose(self) -> None:
        """Clean up all resources"""
        if self._closing:
            return

        self._closing = True

        if self._session:
            await self._cleanup_session(self._session)
            self._session = None

        if hasattr(self.audio_track, "cleanup") and self.audio_track:
            try:
                await self.audio_track.cleanup()
            except Exception as e:
                self.emit("error", f"Error cleaning up audio track: {e}")
