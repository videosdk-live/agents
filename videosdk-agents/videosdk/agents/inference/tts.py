from __future__ import annotations

import asyncio
import base64
import json
import os
import logging
from typing import Any, AsyncIterator, Optional, Dict, Union, List

import aiohttp

from videosdk.agents import TTS as BaseTTS

logger = logging.getLogger(__name__)

# Default inference gateway URLs
VIDEOSDK_INFERENCE_URL = "wss://inference-gateway.videosdk.live"

# Default sample rates
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_CHANNELS = 1


class TTS(BaseTTS):
    """
    VideoSDK Inference Gateway TTS Plugin.

    A lightweight Text-to-Speech client that connects to VideoSDK's Inference Gateway.
    Supports multiple providers (Google, Sarvam, Deepgram) through a unified interface.

    Example:
        # Using factory methods (recommended)
        tts = TTS.google(voice_id="Achernar")
        tts = TTS.sarvam(speaker="anushka")
        tts = TTS.deepgram(model_id="aura-asteria-en")

        # Using generic constructor
        tts = TTS(provider="google", model_id="Chirp3-HD", config={"voice_name": "en-US-Chirp3-HD-Achernar"})
    """

    def __init__(
        self,
        *,
        provider: str,
        model_id: str,
        voice_id: str | None = None,
        language: str = "en-US",
        config: Dict[str, Any] | None = None,
        enable_streaming: bool = True,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        base_url: str | None = None,
    ) -> None:
        """
        Initialize the VideoSDK Inference TTS plugin.

        Args:
            provider: TTS provider name (e.g., "google", "sarvamai", "deepgram")
            model_id: Model identifier for the provider
            voice_id: Voice identifier
            language: Language code (default: "en-US")
            config: Provider-specific configuration dictionary
            enable_streaming: Enable streaming synthesis (default: True)
            sample_rate: Audio sample rate (default: 24000)
            base_url: Custom inference gateway URL (default: production gateway)
        """
        super().__init__(sample_rate=sample_rate, num_channels=DEFAULT_CHANNELS)

        self._videosdk_token = os.getenv("VIDEOSDK_AUTH_TOKEN")
        if not self._videosdk_token:
            raise ValueError(
                "VIDEOSDK_AUTH_TOKEN environment variable must be set for authentication"
            )

        self.provider = provider
        self.model_id = model_id
        self.voice_id = voice_id
        self.language = language
        self.config = config or {}
        self.enable_streaming = enable_streaming
        self.base_url = base_url or VIDEOSDK_INFERENCE_URL

        # WebSocket state
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._recv_task: Optional[asyncio.Task] = None
        self._config_sent: bool = False
        self._connection_lock = asyncio.Lock()

        # Synthesis state
        self._interrupted: bool = False
        self._first_chunk_sent: bool = False

    # ==================== Factory Methods ====================

    @staticmethod
    def google(
        *,
        model_id: str = "Chirp3-HD",
        voice_id: str = "Achernar",
        language: str = "en-US",
        speed: float = 1.0,
        pitch: float = 0.0,
        sample_rate: int = 24000,
        enable_streaming: bool = True,
        base_url: str | None = None,
    ) -> "TTS":
        """
        Create a TTS instance configured for Google Cloud Text-to-Speech.

        Args:
            model_id: Google TTS model (default: "Chirp3-HD")
            voice_id: Voice name (default: "Achernar")
            language: Language code (default: "en-US")
            speed: Speech speed (default: 1.0)
            pitch: Voice pitch (default: 0.0)
            sample_rate: Audio sample rate (default: 24000)
            enable_streaming: Enable streaming mode (default: True)
            base_url: Custom inference gateway URL

        Returns:
            Configured TTS instance for Google
        """
        # Build voice_name like: en-US-Chirp3-HD-Achernar
        voice_name = f"{language}-{model_id}-{voice_id}"

        config = {
            "voice_name": voice_name,
            "language_code": language,
            "speed": speed,
            "pitch": pitch,
            "sample_rate": sample_rate,
        }

        return TTS(
            provider="google",
            model_id=model_id,
            voice_id=voice_id,
            language=language,
            config=config,
            enable_streaming=enable_streaming,
            sample_rate=sample_rate,
            base_url=base_url,
        )

    @staticmethod
    def sarvam(
        *,
        model_id: str = "bulbul:v2",
        speaker: str = "anushka",
        language: str = "en-IN",
        sample_rate: int = 24000,
        enable_streaming: bool = True,
        base_url: str | None = None,
    ) -> "TTS":
        """
        Create a TTS instance configured for Sarvam AI.

        Args:
            model_id: Sarvam model (default: "bulbul:v2")
            speaker: Speaker voice (default: "anushka")
            language: Language code (default: "en-IN")
            sample_rate: Audio sample rate (default: 24000)
            enable_streaming: Enable streaming mode (default: True)
            base_url: Custom inference gateway URL

        Returns:
            Configured TTS instance for Sarvam AI
        """
        config = {
            "model": model_id,
            "language": language,
            "speaker": speaker,
            "sample_rate": sample_rate,
        }

        return TTS(
            provider="sarvamai",
            model_id=model_id,
            voice_id=speaker,
            language=language,
            config=config,
            enable_streaming=enable_streaming,
            sample_rate=sample_rate,
            base_url=base_url,
        )

    @staticmethod
    def cartesia(
        *,
        model_id: str = "sonic-2",
        voice_id: Union[str, List[float]] = "faf0731e-dfb9-4cfc-8119-259a79b27e12",
        language: str = "en",
        sample_rate: int = 24000,
        enable_streaming: bool = True,
        base_url: str | None = None,
    ) -> "TTS":
        """
        Create a TTS instance configured for Cartesia.

        Args:
            model_id: Cartesia model (default: "sonic-2")
            voice_id: Voice ID (string) or voice embedding (list of floats)
                     (default: "f786b574-daa5-4673-aa0c-cbe3e8534c02")
            language: Language code (default: "en")
            sample_rate: Audio sample rate (default: 24000)
            enable_streaming: Enable streaming mode (default: True)
            base_url: Custom inference gateway URL

        Returns:
            Configured TTS instance for Cartesia
        """
        config = {
            "model": model_id,
            "language": language,
            "voice": voice_id,
            "sample_rate": sample_rate,
        }

        return TTS(
            provider="cartesia",
            model_id=model_id,
            voice_id=voice_id if isinstance(voice_id, str) else "embedding",
            language=language,
            config=config,
            enable_streaming=enable_streaming,
            sample_rate=sample_rate,
            base_url=base_url,
        )

    @staticmethod
    def deepgram(
        *,
        model_id: str = "aura-2-amalthea-en",
        encoding: str = "linear16",
        sample_rate: int = 24000,
        container: str = "none",
        bit_rate: int | None = None,
        enable_streaming: bool = True,
        base_url: str | None = None,
    ) -> "TTS":
        """
        Create a TTS instance configured for Deepgram Aura.

        Deepgram Aura provides high-quality, low-latency text-to-speech with
        multiple voice options.

        Args:
            model_id: Deepgram Aura model (default: "aura-asteria-en")
                     Available models:
                     - aura-asteria-en (Female, Conversational)
                     - aura-luna-en (Female, Expressive)
                     - aura-stella-en (Female, Warm)
                     - aura-athena-en (Female, Professional)
                     - aura-hera-en (Female, Clear)
                     - aura-orion-en (Male, Deep)
                     - aura-arcas-en (Male, Authoritative)
                     - aura-perseus-en (Male, Dynamic)
                     - aura-angus-en (Male, Conversational)
                     - aura-orpheus-en (Male, Smooth)
                     - aura-helios-en (Male, Energetic)
                     - aura-zeus-en (Male, Commanding)
            encoding: Audio encoding format (default: "linear16")
                     Options: linear16, mulaw, alaw, opus, aac, flac
            sample_rate: Audio sample rate in Hz (default: 24000)
                        Supported: 8000, 16000, 24000, 48000
            container: Container format (default: "none" for raw audio)
                      Options: none, wav, ogg, mp3
            bit_rate: Bitrate in bps for compressed formats (optional)
                     Only used with opus, aac, mp3 encodings
            enable_streaming: Enable streaming mode (default: True)
            base_url: Custom inference gateway URL (optional)

        Returns:
            Configured TTS instance for Deepgram

        Example:
            # Using default voice
            tts = TTS.deepgram()

            # Using specific voice and settings
            tts = TTS.deepgram(
                model_id="aura-orion-en",
                encoding="linear16",
                sample_rate=24000
            )

            # Using compressed audio
            tts = TTS.deepgram(
                model_id="aura-luna-en",
                encoding="opus",
                container="ogg",
                bit_rate=64000
            )
        """
        config = {
            "model": model_id,
            "encoding": encoding,
            "sample_rate": sample_rate,
            "container": container,
        }

        if bit_rate is not None:
            config["bit_rate"] = bit_rate

        return TTS(
            provider="deepgram",
            model_id=model_id,
            voice_id=model_id,  # for Deepgram, model_id includes the voice
            language="en",
            config=config,
            enable_streaming=enable_streaming,
            sample_rate=sample_rate,
            base_url=base_url,
        )

    # ==================== Core Methods ====================

    def reset_first_audio_tracking(self) -> None:
        """Reset tracking for first audio chunk latency."""
        self._first_chunk_sent = False

    async def synthesize(
        self,
        text: AsyncIterator[str] | str,
        voice_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize (string or async iterator of strings)
            voice_id: Optional voice override
            **kwargs: Additional arguments
        """
        if not self.audio_track or not self.loop:
            logger.error("[InferenceTTS] Audio track or event loop not initialized")
            return

        self._interrupted = False
        self.reset_first_audio_tracking()

        try:
            # Ensure connection
            await self._ensure_connection()

            # Send text for synthesis
            if isinstance(text, str):
                await self._send_text(text)
            else:
                await self._send_text_stream(text)

        except Exception as e:
            logger.error(f"[InferenceTTS] Synthesis error: {e}")
            self.emit("error", str(e))

    async def _ensure_connection(self) -> None:
        """Ensure WebSocket connection is established."""
        async with self._connection_lock:
            if self._ws and not self._ws.closed:
                return

            await self._connect_ws()

            # Start receive loop
            if not self._recv_task or self._recv_task.done():
                self._recv_task = asyncio.create_task(self._recv_loop())

            # Send config
            if not self._config_sent:
                await self._send_config()

    async def _connect_ws(self) -> None:
        """Establish WebSocket connection to the inference gateway."""
        if not self._session:
            self._session = aiohttp.ClientSession()

        # Build WebSocket URL with query parameters
        ws_url = (
            f"{self.base_url}/v1/tts"
            f"?provider={self.provider}"
            f"&secret={self._videosdk_token}"
            f"&modelId={self.model_id}"
        )

        if self.voice_id:
            ws_url += f"&voiceId={self.voice_id}"

        try:
            logger.info(
                f"[InferenceTTS] Connecting to {self.base_url} (provider={self.provider})"
            )
            self._ws = await self._session.ws_connect(ws_url, heartbeat=20)
            self._config_sent = False
            logger.info("[InferenceTTS] Connected successfully")
        except Exception as e:
            logger.error(f"[InferenceTTS] Connection failed: {e}")
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
            logger.info(f"[InferenceTTS] Config sent: {self.config}")
        except Exception as e:
            logger.error(f"[InferenceTTS] Failed to send config: {e}")
            raise

    async def _send_text(self, text: str) -> None:
        """Send text for synthesis."""
        if not self._ws or not text.strip():
            return

        text_message = {
            "type": "text",
            "data": text.strip(),
        }

        try:
            await self._ws.send_str(json.dumps(text_message))
            logger.debug(f"[InferenceTTS] Sent text: '{text[:50]}...'")
        except Exception as e:
            logger.error(f"[InferenceTTS] Failed to send text: {e}")
            raise

    async def _send_text_stream(self, text_iterator: AsyncIterator[str]) -> None:
        """Send streaming text for synthesis."""
        if not self._ws:
            return

        buffer = []
        MIN_WORDS = 4
        MAX_DELAY = 1.0
        last_send_time = asyncio.get_event_loop().time()

        try:
            async for chunk in text_iterator:
                if self._interrupted:
                    break

                if not chunk or not chunk.strip():
                    continue

                # Buffer words
                words = chunk.split()
                buffer.extend(words)

                now = asyncio.get_event_loop().time()

                # Send when buffer is full or timeout
                if len(buffer) >= MIN_WORDS or (now - last_send_time > MAX_DELAY):
                    combined_text = " ".join(buffer).strip()
                    if combined_text:
                        await self._send_text(combined_text)
                    buffer.clear()
                    last_send_time = now

            # Send remaining buffer
            if buffer and not self._interrupted:
                combined_text = " ".join(buffer).strip()
                if combined_text:
                    await self._send_text(combined_text)

            # Send flush to signal end
            if not self._interrupted and self._ws:
                await self._ws.send_str(json.dumps({"type": "flush"}))

        except Exception as e:
            logger.error(f"[InferenceTTS] Failed to send text stream: {e}")
            raise

    async def _recv_loop(self) -> None:
        """Background task to receive audio from the server."""
        if not self._ws:
            return

        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_message(msg.data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(
                        f"[InferenceTTS] WebSocket error: {self._ws.exception()}"
                    )
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.info("[InferenceTTS] WebSocket closed by server")
                    break

        except asyncio.CancelledError:
            logger.debug("[InferenceTTS] Receive loop cancelled")
        except Exception as e:
            logger.error(f"[InferenceTTS] Error in receive loop: {e}")
        finally:
            self._config_sent = False

    async def _handle_message(self, raw_message: str) -> None:
        """Handle incoming messages from the inference server."""
        try:
            # Skip empty messages
            if not raw_message or not raw_message.strip():
                logger.debug("[InferenceTTS] Received empty message, skipping")
                return

            # Try to parse as JSON
            try:
                data = json.loads(raw_message)
            except json.JSONDecodeError:
                # Handle non-JSON messages (like plain text acknowledgments)
                logger.debug(
                    f"[InferenceTTS] Received non-JSON message: {raw_message[:100]}"
                )

                # Check if it's a known plain text response
                if "success" in raw_message.lower() or "ok" in raw_message.lower():
                    logger.debug("[InferenceTTS] Received acknowledgment")
                    return

                # Otherwise, log and continue
                logger.warning(
                    f"[InferenceTTS] Unexpected non-JSON message: {raw_message[:200]}"
                )
                return

            msg_type = data.get("type")

            if msg_type == "audio":
                await self._handle_audio(data.get("data", {}))

            elif msg_type == "event":
                event_data = data.get("data", {})
                event_type = event_data.get("eventType")

                if event_type == "TTS_COMPLETE":
                    logger.debug("[InferenceTTS] Synthesis completed")

            elif msg_type == "error":
                error_msg = data.get("data", {}).get("error") or data.get(
                    "message", "Unknown error"
                )
                logger.error(f"[InferenceTTS] Server error: {error_msg}")
                self.emit("error", error_msg)

        except Exception as e:
            logger.error(f"[InferenceTTS] Error handling message: {e}")

    async def _handle_audio(self, audio_data: Dict[str, Any]) -> None:
        """Handle audio data from the server."""
        if self._interrupted or not audio_data:
            return

        audio_b64 = audio_data.get("audio")
        if not audio_b64:
            return

        try:
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_b64)

            # Remove WAV header if present
            audio_bytes = self._remove_wav_header(audio_bytes)

            # Trigger first audio callback for TTFB metrics
            if not self._first_chunk_sent and self._first_audio_callback:
                self._first_chunk_sent = True
                asyncio.create_task(self._first_audio_callback())

            # Send to audio track
            if self.audio_track:
                await self.audio_track.add_new_bytes(audio_bytes)

        except Exception as e:
            logger.error(f"[InferenceTTS] Failed to process audio: {e}")

    def _remove_wav_header(self, audio_bytes: bytes) -> bytes:
        """Remove WAV header if present."""
        if audio_bytes.startswith(b"RIFF"):
            data_pos = audio_bytes.find(b"data")
            if data_pos != -1:
                return audio_bytes[data_pos + 8 :]
        return audio_bytes

    async def interrupt(self) -> None:
        """Interrupt ongoing synthesis."""
        self._interrupted = True
        if self.audio_track:
            self.audio_track.interrupt()

    async def _cleanup_connection(self) -> None:
        """Clean up WebSocket connection resources."""
        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
            self._recv_task = None

        if self._ws and not self._ws.closed:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = None
        self._config_sent = False

    async def aclose(self) -> None:
        """Clean up all resources."""
        logger.info(f"[InferenceTTS] Closing TTS (provider={self.provider})")

        self._interrupted = True

        # Clean up connection
        await self._cleanup_connection()

        # Close HTTP session
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

        # Call parent cleanup
        await super().aclose()

        logger.info("[InferenceTTS] Closed successfully")

    # ==================== Properties ====================

    @property
    def label(self) -> str:
        """Get a descriptive label for this TTS instance."""
        return f"videosdk.inference.TTS.{self.provider}.{self.model_id}"
