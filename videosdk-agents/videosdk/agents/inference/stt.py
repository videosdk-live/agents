from __future__ import annotations

import asyncio
import base64
import json
import os
import time
import logging
from typing import Any, Optional, Dict

import aiohttp

from videosdk.agents import (
    STT as BaseSTT,
    STTResponse,
    SpeechData,
    SpeechEventType,
    global_event_emitter,
)

logger = logging.getLogger(__name__)

# Default inference gateway URLs
VIDEOSDK_INFERENCE_URL = "wss://inference-gateway.videosdk.live"


class STT(BaseSTT):
    """
    VideoSDK Inference Gateway STT Plugin.

    A lightweight Speech-to-Text client that connects to VideoSDK's Inference Gateway.
    Supports multiple providers (Google, Sarvam, Deepgram) through a unified interface.

    Example:
        # Using factory methods (recommended)
        stt = STT.google(language="en-US")
        stt = STT.sarvam(language="en-IN")

        # Using generic constructor
        stt = STT(provider="google", model_id="chirp_3", config={"language": "en-US"})
    """

    def __init__(
        self,
        *,
        provider: str,
        model_id: str,
        language: str = "en-US",
        config: Dict[str, Any] | None = None,
        enable_streaming: bool = True,
        base_url: str | None = None,
    ) -> None:
        """
        Initialize the VideoSDK Inference STT plugin.

        Args:
            provider: STT provider name (e.g., "google", "sarvamai", "deepgram")
            model_id: Model identifier for the provider
            language: Language code (default: "en-US")
            config: Provider-specific configuration dictionary
            enable_streaming: Enable streaming transcription (default: True)
            base_url: Custom inference gateway URL (default: production gateway)
        """
        super().__init__()

        self._videosdk_token = os.getenv("VIDEOSDK_AUTH_TOKEN")
        if not self._videosdk_token:
            raise ValueError(
                "VIDEOSDK_AUTH_TOKEN environment variable must be set for authentication"
            )

        self.provider = provider
        self.model_id = model_id
        self.language = language
        self.config = config or {}
        self.enable_streaming = enable_streaming
        self.base_url = base_url or VIDEOSDK_INFERENCE_URL

        # WebSocket state
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._config_sent: bool = False

        # Speech state tracking
        self._is_speaking: bool = False
        self._last_transcript: str = ""

        # Metrics tracking
        self._stt_start_time: Optional[float] = None

    # ==================== Factory Methods ====================

    @staticmethod
    def google(
        *,
        model_id: str = "chirp_3",
        language: str = "en-US",
        languages: list[str] | None = None,
        interim_results: bool = True,
        punctuate: bool = True,
        location: str = "asia-south1",
        input_sample_rate: int = 48000,
        output_sample_rate: int = 16000,
        enable_streaming: bool = True,
        base_url: str | None = None,
    ) -> "STT":
        """
        Create an STT instance configured for Google Cloud Speech-to-Text.

        Args:
            model_id: Google STT model (default: "chirp_3"). Options: "chirp_3", "latest_long", "latest_short"
            language: Primary language code (default: "en-US")
            languages: List of languages for auto-detection (default: [language])
            interim_results: Return interim transcription results (default: True)
            punctuate: Add punctuation to transcripts (default: True)
            location: Google Cloud region (default: "asia-south1")
            input_sample_rate: Input audio sample rate (default: 48000)
            output_sample_rate: Output sample rate for processing (default: 16000)
            enable_streaming: Enable streaming mode (default: True)
            base_url: Custom inference gateway URL

        Returns:
            Configured STT instance for Google
        """
        config = {
            "model": model_id,
            "language": language,
            "languages": languages or [language],
            "input_sample_rate": input_sample_rate,
            "output_sample_rate": output_sample_rate,
            "interim_results": interim_results,
            "punctuate": punctuate,
            "location": location,
        }

        return STT(
            provider="google",
            model_id=model_id,
            language=language,
            config=config,
            enable_streaming=enable_streaming,
            base_url=base_url,
        )

    @staticmethod
    def sarvam(
        *,
        model_id: str = "saarika:v2.5",
        language: str = "en-IN",
        input_sample_rate: int = 48000,
        output_sample_rate: int = 16000,
        enable_streaming: bool = True,
        base_url: str | None = None,
    ) -> "STT":
        """
        Create an STT instance configured for Sarvam AI.

        Args:
            model_id: Sarvam model (default: "saarika:v2.5")
            language: Language code (default: "en-IN"). Supports Indian languages.
            input_sample_rate: Input audio sample rate (default: 48000)
            output_sample_rate: Output sample rate for processing (default: 16000)
            enable_streaming: Enable streaming mode (default: True)
            base_url: Custom inference gateway URL

        Returns:
            Configured STT instance for Sarvam AI
        """
        config = {
            "model": model_id,
            "language": language,
            "input_sample_rate": input_sample_rate,
            "output_sample_rate": output_sample_rate,
        }

        return STT(
            provider="sarvamai",
            model_id=model_id,
            language=language,
            config=config,
            enable_streaming=enable_streaming,
            base_url=base_url,
        )

    @staticmethod
    def deepgram(
        *,
        model_id: str = "nova-2",
        language: str = "en-US",
        input_sample_rate: int = 48000,
        interim_results: bool = True,
        punctuate: bool = True,
        smart_format: bool = True,
        endpointing: int = 50,
        enable_streaming: bool = True,
        base_url: str | None = None,
    ) -> "STT":
        """
        Create an STT instance configured for Deepgram.

        Args:
            model_id: Deepgram model (default: "nova-2")
            language: Language code (default: "en-US")
            input_sample_rate: Input audio sample rate (default: 48000)
            interim_results: Return interim transcription results (default: True)
            punctuate: Add punctuation to transcripts (default: True)
            smart_format: Enable smart formatting (default: True)
            endpointing: Endpointing threshold in ms (default: 50)
            enable_streaming: Enable streaming mode (default: True)
            base_url: Custom inference gateway URL

        Returns:
            Configured STT instance for Deepgram
        """
        config = {
            "model": model_id,
            "language": language,
            "input_sample_rate": input_sample_rate,
            "interim_results": interim_results,
            "punctuate": punctuate,
            "smart_format": smart_format,
            "endpointing": endpointing,
        }

        return STT(
            provider="deepgram",
            model_id=model_id,
            language=language,
            config=config,
            enable_streaming=enable_streaming,
            base_url=base_url,
        )

    # ==================== Core Methods ====================

    async def process_audio(
        self,
        audio_frames: bytes,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Process audio frames and send to the inference server.

        Args:
            audio_frames: Raw PCM audio bytes (16-bit, typically 48kHz stereo)
            language: Optional language override
            **kwargs: Additional arguments (unused)
        """
        if not self.enable_streaming:
            logger.warning("Non-streaming mode not yet supported for inference STT")
            return

        try:
            # Ensure WebSocket connection
            if not self._ws or self._ws.closed:
                await self._connect_ws()
                if not self._ws_task or self._ws_task.done():
                    self._ws_task = asyncio.create_task(self._listen_for_responses())

            # Send config on first audio (after connection)
            if not self._config_sent:
                await self._send_config()

            # Send audio data
            await self._send_audio(audio_frames)

        except Exception as e:
            logger.error(f"[InferenceSTT] Error in process_audio: {e}")
            self.emit("error", str(e))
            await self._cleanup_connection()

    async def _connect_ws(self) -> None:
        """Establish WebSocket connection to the inference gateway."""
        if not self._session:
            self._session = aiohttp.ClientSession()

        # Build WebSocket URL with query parameters
        ws_url = (
            f"{self.base_url}/v1/stt"
            f"?provider={self.provider}"
            f"&secret={self._videosdk_token}"
            f"&modelId={self.model_id}"
        )

        try:
            logger.info(
                f"[InferenceSTT] Connecting to {self.base_url} (provider={self.provider})"
            )
            self._ws = await self._session.ws_connect(ws_url)
            self._config_sent = False
            logger.info(f"[InferenceSTT] Connected successfully")
        except Exception as e:
            logger.error(f"[InferenceSTT] Connection failed: {e}")
            raise

    async def _send_config(self) -> None:
        """Send configuration message to the inference server."""
        if not self._ws:
            return

        config_message = {
            "type": "config",
            "data": self.config,
        }

        try:
            await self._ws.send_str(json.dumps(config_message))
            self._config_sent = True
            logger.info(f"[InferenceSTT] Config sent: {self.config}")
        except Exception as e:
            logger.error(f"[InferenceSTT] Failed to send config: {e}")
            raise

    async def _send_audio(self, audio_bytes: bytes) -> None:
        """Send audio data to the inference server."""
        if not self._ws:
            return

        # Track STT start time for metrics
        if self._stt_start_time is None:
            self._stt_start_time = time.perf_counter()

        # Encode audio as base64 for JSON transmission
        audio_message = {
            "type": "audio",
            "data": base64.b64encode(audio_bytes).decode("utf-8"),
        }

        try:
            await self._ws.send_str(json.dumps(audio_message))
        except Exception as e:
            logger.error(f"[InferenceSTT] Failed to send audio: {e}")
            raise

    async def _listen_for_responses(self) -> None:
        """Background task to listen for WebSocket responses from the server."""
        if not self._ws:
            return

        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_message(msg.data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(
                        f"[InferenceSTT] WebSocket error: {self._ws.exception()}"
                    )
                    self.emit("error", f"WebSocket error: {self._ws.exception()}")
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.info("[InferenceSTT] WebSocket closed by server")
                    break

        except asyncio.CancelledError:
            logger.debug("[InferenceSTT] WebSocket listener cancelled")
        except Exception as e:
            logger.error(f"[InferenceSTT] Error in WebSocket listener: {e}")
            self.emit("error", str(e))
        finally:
            await self._cleanup_connection()

    async def _handle_message(self, raw_message: str) -> None:
        """
        Handle incoming messages from the inference server.

        Args:
            raw_message: Raw JSON message string from server
        """
        try:
            data = json.loads(raw_message)
            msg_type = data.get("type")

            if msg_type == "event":
                await self._handle_event(data.get("data", {}))
            elif msg_type == "error":
                error_msg = data.get("data", {}).get("error") or data.get(
                    "message", "Unknown error"
                )
                logger.error(f"[InferenceSTT] Server error: {error_msg}")
                self.emit("error", error_msg)

        except json.JSONDecodeError as e:
            logger.error(f"[InferenceSTT] Failed to parse message: {e}")
        except Exception as e:
            logger.error(f"[InferenceSTT] Error handling message: {e}")

    async def _handle_event(self, event_data: Dict[str, Any]) -> None:
        """
        Handle event messages from the inference server.

        Args:
            event_data: Event data dictionary
        """
        event_type = event_data.get("eventType")

        if event_type == "TRANSCRIPT":
            text = event_data.get("text", "")
            language = event_data.get("language", self.language)
            is_final = event_data.get("is_final", True)
            confidence = event_data.get("confidence", 1.0)

            if text.strip():
                self._last_transcript = text.strip()

                # Create STT response
                response = STTResponse(
                    event_type=(
                        SpeechEventType.FINAL if is_final else SpeechEventType.INTERIM
                    ),
                    data=SpeechData(
                        text=text.strip(),
                        language=language,
                        confidence=confidence,
                    ),
                    metadata={
                        "provider": self.provider,
                        "model": self.model_id,
                    },
                )

                # Call transcript callback
                if self._transcript_callback:
                    await self._transcript_callback(response)

                # Log for debugging
                transcript_type = "FINAL" if is_final else "INTERIM"
                logger.debug(
                    f"[InferenceSTT] [{transcript_type}] {text} "
                    f"(lang={language}, conf={confidence:.2f})"
                )

                # Reset STT timing on final transcript
                if is_final:
                    self._stt_start_time = None

        elif event_type == "START_SPEECH":
            if not self._is_speaking:
                self._is_speaking = True
                global_event_emitter.emit("speech_started")
                logger.debug("[InferenceSTT] Speech started")

        elif event_type == "END_SPEECH":
            if self._is_speaking:
                self._is_speaking = False
                global_event_emitter.emit("speech_stopped")
                logger.debug("[InferenceSTT] Speech ended")

    async def _cleanup_connection(self) -> None:
        """Clean up WebSocket connection resources."""
        if self._ws and not self._ws.closed:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = None
        self._config_sent = False

    async def aclose(self) -> None:
        """Clean up all resources."""
        logger.info(f"[InferenceSTT] Closing STT (provider={self.provider})")

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

        # Call parent cleanup
        await super().aclose()

        logger.info(f"[InferenceSTT] Closed successfully")

    # ==================== Properties ====================

    @property
    def label(self) -> str:
        """Get a descriptive label for this STT instance."""
        return f"videosdk.inference.STT.{self.provider}.{self.model_id}"
