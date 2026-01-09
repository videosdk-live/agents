from __future__ import annotations

import asyncio
import base64
import json
import os
import logging
import time
from typing import Any, Optional, Dict, Literal, List
from dataclasses import dataclass, field

import aiohttp
import numpy as np
from scipy import signal
import av

from videosdk.agents import (
    RealtimeBaseModel,
    CustomAudioStreamTrack,
    Agent,
    FunctionTool,
    build_gemini_schema,
    is_function_tool,
    get_tool_info,
    EncodeOptions,
    ResizeOptions,
    encode as encode_image,
)
from videosdk.agents import realtime_metrics_collector
from videosdk.agents.event_bus import global_event_emitter

logger = logging.getLogger(__name__)

# Default inference gateway URL
DEFAULT_LLM_URL = "wss://inference-gateway.videosdk.live"


# Default image encoding options for vision
DEFAULT_IMAGE_ENCODE_OPTIONS = EncodeOptions(
    format="JPEG",
    quality=75,
    resize_options=ResizeOptions(width=1024, height=1024),
)

Voice = Literal["Puck", "Charon", "Kore", "Fenrir", "Aoede"]
RealtimeEventTypes = Literal[
    "user_speech_started",
    "user_speech_ended",
    "agent_speech_started",
    "agent_speech_ended",
    "realtime_model_transcription",
    "error",
]


@dataclass
class GeminiRealtimeConfig:
    """Configuration for Gemini Realtime via Inference Gateway.

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
        response_modalities: List of enabled response types. Options: ["TEXT", "AUDIO"]. Defaults to ["AUDIO"]
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
    response_modalities: List[str] | None = field(default_factory=lambda: ["AUDIO"])


class Realtime(RealtimeBaseModel[RealtimeEventTypes]):
    """
    VideoSDK Inference Gateway Realtime Plugin.

    A lightweight multimodal realtime client that connects to VideoSDK's Inference Gateway.
    Supports Gemini's realtime model for audio-first communication.

    Example:
        # Using factory method (recommended)
        model = Realtime.gemini(
            model="gemini-2.0-flash-exp",
            voice="Puck",
            language_code="en-US",
        )

        # Use with RealTimePipeline
        pipeline = RealTimePipeline(model=model)
    """

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        config: GeminiRealtimeConfig | None = None,
        base_url: str | None = None,
    ) -> None:
        """
        Initialize the VideoSDK Inference Realtime plugin.

        Args:
            provider: Realtime provider name (currently only "gemini" supported)
            model: Model identifier (e.g., "gemini-2.0-flash-exp")
            config: Provider-specific configuration
            base_url: Custom inference gateway URL (default: production gateway)
        """
        super().__init__()

        self._videosdk_token = os.getenv("VIDEOSDK_AUTH_TOKEN")
        if not self._videosdk_token:
            raise ValueError(
                "VIDEOSDK_AUTH_TOKEN environment variable must be set for authentication"
            )

        self.provider = provider
        self.model = model
        self.model_id = model.split("/")[-1] if "/" in model else model
        self.config = config or GeminiRealtimeConfig()
        self.base_url = base_url or DEFAULT_LLM_URL

        # WebSocket state
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._config_sent: bool = False
        self._closing: bool = False

        # Audio state
        self.audio_track: Optional[CustomAudioStreamTrack] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.target_sample_rate = 24000
        self.input_sample_rate = 48000

        # Speaking state tracking
        self._user_speaking: bool = False
        self._agent_speaking: bool = False

        # Agent configuration
        self._instructions: str = (
            "You are a helpful voice assistant that can answer questions and help with tasks."
        )
        self.tools: List[FunctionTool] = []
        self.tools_formatted: List[Dict] = []

    # ==================== Factory Methods ====================

    @staticmethod
    def gemini(
        *,
        model: str = "gemini-2.0-flash-exp",
        voice: Voice = "Puck",
        language_code: str = "en-US",
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: float | None = None,
        candidate_count: int = 1,
        max_output_tokens: int | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        response_modalities: List[str] | None = None,
        base_url: str | None = None,
    ) -> "Realtime":
        """
        Create a Realtime instance configured for Google Gemini.

        Args:
            model: Gemini model identifier (default: "gemini-2.0-flash-exp")
            voice: Voice ID for audio output. Options: 'Puck', 'Charon', 'Kore', 'Fenrir', 'Aoede'
            language_code: Language code for speech synthesis (default: "en-US")
            temperature: Controls randomness in responses (0.0 to 1.0)
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            top_k: Limits tokens considered for each generation step
            candidate_count: Number of response candidates (default: 1)
            max_output_tokens: Maximum tokens in model responses
            presence_penalty: Penalizes token presence (-2.0 to 2.0)
            frequency_penalty: Penalizes token frequency (-2.0 to 2.0)
            response_modalities: Response types ["TEXT", "AUDIO"] (default: ["AUDIO"])
            base_url: Custom inference gateway URL

        Returns:
            Configured Realtime instance for Gemini
        """
        config = GeminiRealtimeConfig(
            voice=voice,
            language_code=language_code,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            candidate_count=candidate_count,
            max_output_tokens=max_output_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            response_modalities=response_modalities or ["AUDIO"],
        )

        return Realtime(
            provider="gemini",
            model=model,
            config=config,
            base_url=base_url,
        )

    # ==================== Agent Setup ====================

    def set_agent(self, agent: Agent) -> None:
        """Set agent instructions and tools."""
        self._instructions = agent.instructions
        self.tools = agent.tools or []
        self.tools_formatted = self._convert_tools_to_format(self.tools)

    def _convert_tools_to_format(self, tools: List[FunctionTool]) -> List[Dict]:
        """Convert tool definitions to Gemini format."""
        function_declarations = []

        for tool in tools:
            if not is_function_tool(tool):
                continue

            try:
                function_declaration = build_gemini_schema(tool)
                function_declarations.append(function_declaration)
            except Exception as e:
                logger.error(f"[InferenceRealtime] Failed to format tool {tool}: {e}")
                continue

        return function_declarations

    # ==================== Connection Management ====================

    async def connect(self) -> None:
        """Connect to the inference gateway."""
        if self._ws and not self._ws.closed:
            await self._cleanup_connection()

        self._closing = False

        try:
            # Create audio track if needed
            if (
                not self.audio_track
                and self.loop
                and "AUDIO" in self.config.response_modalities
            ):
                self.audio_track = CustomAudioStreamTrack(self.loop)
            elif not self.loop and "AUDIO" in self.config.response_modalities:
                self.emit("error", "Event loop not initialized. Audio playback will not work.")
                raise RuntimeError("Event loop not initialized. Audio playback will not work.")

            # Connect to WebSocket
            await self._connect_ws()

            # Start listening for responses
            if not self._ws_task or self._ws_task.done():
                self._ws_task = asyncio.create_task(
                    self._listen_for_responses(), name="inference-realtime-listener"
                )

            logger.info(f"[InferenceRealtime] Connected to inference gateway (provider={self.provider})")

        except Exception as e:
            self.emit("error", f"Error connecting to inference gateway: {e}")
            logger.error(f"[InferenceRealtime] Connection error: {e}")
            raise

    async def _connect_ws(self) -> None:
        """Establish WebSocket connection to the inference gateway."""
        if not self._session:
            self._session = aiohttp.ClientSession()

        # Build WebSocket URL with query parameters
        ws_url = (
            f"{self.base_url}/v1/llm"
            f"?provider={self.provider}"
            f"&secret={self._videosdk_token}"
            f"&modelId={self.model_id}"
        )

        try:
            logger.info(f"[InferenceRealtime] Connecting to {self.base_url}")
            self._ws = await self._session.ws_connect(ws_url)
            self._config_sent = False
            logger.info("[InferenceRealtime] WebSocket connected successfully")
        except Exception as e:
            logger.error(f"[InferenceRealtime] WebSocket connection failed: {e}")
            raise

    async def _send_config(self) -> None:
        """Send configuration to the inference server."""
        if not self._ws:
            return

        config_data = {
            "model": f"models/{self.model}" if not self.model.startswith("models/") else self.model,
            "instructions": self._instructions,
            "voice": self.config.voice,
            "language_code": self.config.language_code,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "candidate_count": self.config.candidate_count,
            "max_output_tokens": self.config.max_output_tokens,
            "presence_penalty": self.config.presence_penalty,
            "frequency_penalty": self.config.frequency_penalty,
            "response_modalities": self.config.response_modalities,
        }

        # Add tools if available
        if self.tools_formatted:
            config_data["tools"] = self.tools_formatted

        config_message = {
            "type": "config",
            "data": config_data,
        }

        try:
            await self._ws.send_str(json.dumps(config_message))
            self._config_sent = True
            logger.info(f"[InferenceRealtime] Config sent: voice={self.config.voice}, modalities={self.config.response_modalities}")
        except Exception as e:
            logger.error(f"[InferenceRealtime] Failed to send config: {e}")
            raise

    # ==================== Audio Input ====================

    async def handle_audio_input(self, audio_data: bytes) -> None:
        """Handle incoming audio data from the user."""
        if not self._ws or self._closing:
            return

        if self.current_utterance and not self.current_utterance.is_interruptible:
            return

        if "AUDIO" not in self.config.response_modalities:
            return

        try:
            # Ensure connection and config
            if self._ws.closed:
                await self._connect_ws()
                if not self._ws_task or self._ws_task.done():
                    self._ws_task = asyncio.create_task(self._listen_for_responses())

            if not self._config_sent:
                await self._send_config()

            # Resample audio from 48kHz to 24kHz (expected by Gemini)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_array = signal.resample(
                audio_array,
                int(len(audio_array) * self.target_sample_rate / self.input_sample_rate),
            )
            audio_data = audio_array.astype(np.int16).tobytes()

            # Send audio as base64
            audio_message = {
                "type": "audio",
                "data": base64.b64encode(audio_data).decode("utf-8"),
            }

            await self._ws.send_str(json.dumps(audio_message))

        except Exception as e:
            logger.error(f"[InferenceRealtime] Error sending audio: {e}")
            self.emit("error", str(e))

    async def handle_video_input(self, video_data: av.VideoFrame) -> None:
        """Handle incoming video data from the user (vision mode)."""
        if not self._ws or self._closing:
            return

        try:
            if not video_data or not video_data.planes:
                return

            # Rate limit video frames
            now = time.monotonic()
            if hasattr(self, "_last_video_frame") and (now - self._last_video_frame) < 0.5:
                return
            self._last_video_frame = now

            # Encode frame as JPEG
            processed_jpeg = encode_image(video_data, DEFAULT_IMAGE_ENCODE_OPTIONS)
            if not processed_jpeg or len(processed_jpeg) < 100:
                logger.warning("[InferenceRealtime] Invalid JPEG data generated")
                return

            # Send video as base64
            video_message = {
                "type": "video",
                "data": base64.b64encode(processed_jpeg).decode("utf-8"),
            }

            await self._ws.send_str(json.dumps(video_message))

        except Exception as e:
            logger.error(f"[InferenceRealtime] Error sending video: {e}")

    # ==================== Text Messages ====================

    async def send_message(self, message: str) -> None:
        """Send a text message to get audio response."""
        if not self._ws or self._closing:
            logger.warning("[InferenceRealtime] Cannot send message: not connected")
            return

        try:
            if not self._config_sent:
                await self._send_config()

            text_message = {
                "type": "text",
                "data": f"Please start the conversation by saying exactly this, without any additional text: '{message}'",
            }

            await self._ws.send_str(json.dumps(text_message))
            logger.debug(f"[InferenceRealtime] Sent message: {message[:50]}...")

        except Exception as e:
            logger.error(f"[InferenceRealtime] Error sending message: {e}")
            self.emit("error", str(e))

    async def send_text_message(self, message: str) -> None:
        """Send a text message for text-only communication."""
        if not self._ws or self._closing:
            logger.warning("[InferenceRealtime] Cannot send text: not connected")
            return

        try:
            if not self._config_sent:
                await self._send_config()

            text_message = {
                "type": "text",
                "data": message,
            }

            await self._ws.send_str(json.dumps(text_message))

        except Exception as e:
            logger.error(f"[InferenceRealtime] Error sending text message: {e}")
            self.emit("error", str(e))

    async def send_message_with_frames(self, message: str, frames: List[av.VideoFrame]) -> None:
        """Send a text message with video frames for vision-enabled communication."""
        if not self._ws or self._closing:
            logger.warning("[InferenceRealtime] Cannot send message with frames: not connected")
            return

        try:
            if not self._config_sent:
                await self._send_config()

            # Encode frames as base64
            encoded_frames = []
            for frame in frames:
                try:
                    processed_jpeg = encode_image(frame, DEFAULT_IMAGE_ENCODE_OPTIONS)
                    if processed_jpeg and len(processed_jpeg) >= 100:
                        encoded_frames.append(base64.b64encode(processed_jpeg).decode("utf-8"))
                except Exception as e:
                    logger.error(f"[InferenceRealtime] Error encoding frame: {e}")

            # Send message with frames
            message_with_frames = {
                "type": "text_with_frames",
                "data": {
                    "text": message,
                    "frames": encoded_frames,
                },
            }

            await self._ws.send_str(json.dumps(message_with_frames))

        except Exception as e:
            logger.error(f"[InferenceRealtime] Error sending message with frames: {e}")
            self.emit("error", str(e))

    # ==================== Interruption ====================

    async def interrupt(self) -> None:
        """Interrupt current response."""
        if not self._ws or self._closing:
            return

        if self.current_utterance and not self.current_utterance.is_interruptible:
            logger.info("[InferenceRealtime] Utterance not interruptible, skipping interrupt")
            return

        try:
            interrupt_message = {"type": "interrupt"}
            await self._ws.send_str(json.dumps(interrupt_message))

            self.emit("agent_speech_ended", {})
            await realtime_metrics_collector.set_interrupted()

            if self.audio_track and "AUDIO" in self.config.response_modalities:
                self.audio_track.interrupt()

            logger.debug("[InferenceRealtime] Sent interrupt signal")

        except Exception as e:
            logger.error(f"[InferenceRealtime] Interrupt error: {e}")
            self.emit("error", str(e))

    # ==================== Response Handling ====================

    async def _listen_for_responses(self) -> None:
        """Background task to listen for WebSocket responses from the server."""
        if not self._ws:
            return

        accumulated_input_text = ""
        accumulated_output_text = ""

        try:
            async for msg in self._ws:
                if self._closing:
                    break

                if msg.type == aiohttp.WSMsgType.TEXT:
                    accumulated_input_text, accumulated_output_text = await self._handle_message(
                        msg.data, accumulated_input_text, accumulated_output_text
                    )
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"[InferenceRealtime] WebSocket error: {self._ws.exception()}")
                    self.emit("error", f"WebSocket error: {self._ws.exception()}")
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.info("[InferenceRealtime] WebSocket closed by server")
                    break

        except asyncio.CancelledError:
            logger.debug("[InferenceRealtime] WebSocket listener cancelled")
        except Exception as e:
            logger.error(f"[InferenceRealtime] Error in WebSocket listener: {e}")
            self.emit("error", str(e))
        finally:
            await self._cleanup_connection()

    async def _handle_message(
        self,
        raw_message: str,
        accumulated_input_text: str,
        accumulated_output_text: str,
    ) -> tuple[str, str]:
        """Handle incoming messages from the inference server."""
        try:
            data = json.loads(raw_message)
            msg_type = data.get("type")

            if msg_type == "audio":
                await self._handle_audio_data(data.get("data", {}))

            elif msg_type == "event":
                event_data = data.get("data", {})
                accumulated_input_text, accumulated_output_text = await self._handle_event(
                    event_data, accumulated_input_text, accumulated_output_text
                )

            elif msg_type == "error":
                error_msg = data.get("data", {}).get("error") or data.get("message", "Unknown error")
                logger.error(f"[InferenceRealtime] Server error: {error_msg}")
                self.emit("error", error_msg)

        except json.JSONDecodeError as e:
            logger.error(f"[InferenceRealtime] Failed to parse message: {e}")
        except Exception as e:
            logger.error(f"[InferenceRealtime] Error handling message: {e}")

        return accumulated_input_text, accumulated_output_text

    async def _handle_audio_data(self, audio_data: Dict[str, Any]) -> None:
        """Handle audio data received from the inference server."""
        audio_b64 = audio_data.get("audio")
        if not audio_b64:
            return

        if not self.audio_track:
            logger.warning("[InferenceRealtime] Audio track not available")
            return

        try:
            audio_bytes = base64.b64decode(audio_b64)

            # Ensure even number of bytes
            if len(audio_bytes) % 2 != 0:
                audio_bytes += b"\x00"

            if not self._agent_speaking:
                self._agent_speaking = True
                self.emit("agent_speech_started", {})
                await realtime_metrics_collector.set_agent_speech_start()

            await self.audio_track.add_new_bytes(audio_bytes)

        except Exception as e:
            logger.error(f"[InferenceRealtime] Error processing audio data: {e}")

    async def _handle_event(
        self,
        event_data: Dict[str, Any],
        accumulated_input_text: str,
        accumulated_output_text: str,
    ) -> tuple[str, str]:
        """Handle event messages from the inference server."""
        event_type = event_data.get("eventType")

        if event_type == "user_speech_started":
            if not self._user_speaking:
                self._user_speaking = True
                await realtime_metrics_collector.set_user_speech_start()
                self.emit("user_speech_started", {"type": "done"})

        elif event_type == "user_speech_ended":
            if self._user_speaking:
                self._user_speaking = False
                await realtime_metrics_collector.set_user_speech_end()
                self.emit("user_speech_ended", {})

        elif event_type == "agent_speech_started":
            if not self._agent_speaking:
                self._agent_speaking = True
                self.emit("agent_speech_started", {})
                await realtime_metrics_collector.set_agent_speech_start()

        elif event_type == "agent_speech_ended":
            if self._agent_speaking:
                self._agent_speaking = False
                self.emit("agent_speech_ended", {})
                await realtime_metrics_collector.set_agent_speech_end(timeout=1.0)

        elif event_type == "input_transcription":
            text = event_data.get("text", "")
            if text:
                accumulated_input_text = text
                global_event_emitter.emit(
                    "input_transcription",
                    {"text": accumulated_input_text, "is_final": False},
                )

        elif event_type == "output_transcription":
            text = event_data.get("text", "")
            if text:
                accumulated_output_text += text
                global_event_emitter.emit(
                    "output_transcription",
                    {"text": accumulated_output_text, "is_final": False},
                )

        elif event_type == "user_transcript":
            text = event_data.get("text", "")
            if text:
                await realtime_metrics_collector.set_user_transcript(text)
                self.emit(
                    "realtime_model_transcription",
                    {"role": "user", "text": text, "is_final": True},
                )
                accumulated_input_text = ""

        elif event_type == "text_response":
            text = event_data.get("text", "")
            is_final = event_data.get("is_final", False)
            if text and is_final:
                await realtime_metrics_collector.set_agent_response(text)
                self.emit(
                    "realtime_model_transcription",
                    {"role": "agent", "text": text, "is_final": True},
                )
                global_event_emitter.emit("text_response", {"type": "done", "text": text})
                accumulated_output_text = ""

        elif event_type == "response_interrupted":
            if self.audio_track and "AUDIO" in self.config.response_modalities:
                self.audio_track.interrupt()

        return accumulated_input_text, accumulated_output_text

    # ==================== Cleanup ====================

    async def _cleanup_connection(self) -> None:
        """Clean up WebSocket connection resources."""
        if self._ws and not self._ws.closed:
            try:
                # Send stop message before closing
                stop_message = {"type": "stop"}
                await self._ws.send_str(json.dumps(stop_message))
            except Exception:
                pass

            try:
                await self._ws.close()
            except Exception:
                pass

        self._ws = None
        self._config_sent = False

    async def aclose(self) -> None:
        """Clean up all resources."""
        logger.info(f"[InferenceRealtime] Closing (provider={self.provider})")

        self._closing = True

        # Cancel listener task
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
            self._ws_task = None

        # Close WebSocket
        await self._cleanup_connection()

        # Close HTTP session
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

        # Cleanup audio track
        if hasattr(self.audio_track, "cleanup") and self.audio_track:
            try:
                await self.audio_track.cleanup()
            except Exception as e:
                logger.error(f"[InferenceRealtime] Error cleaning up audio track: {e}")

        logger.info("[InferenceRealtime] Closed successfully")

    # ==================== Properties ====================

    @property
    def label(self) -> str:
        """Get a descriptive label for this Realtime instance."""
        return f"videosdk.inference.Realtime.{self.provider}.{self.model_id}"

