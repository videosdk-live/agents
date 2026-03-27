from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import logging
from typing import Any, AsyncIterator, Optional, Dict

import aiohttp
from videosdk.agents import TTS as BaseTTS

logger = logging.getLogger(__name__)

VIDEOSDK_INFERENCE_URL = "wss://inference-gateway.videosdk.live"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_CHANNELS = 1

MIN_CHARS_FOR_TTS = 5

# Providers that require text content validation before sending
TEXT_VALIDATION_PROVIDERS = {"sarvamai","cartesia"}


def _has_enough_content(text: str, provider: str) -> bool:
    """
    Returns True if text has enough real content to send to TTS.
    Prevents 400 errors from Sarvam when text is too short or only punctuation.
    Supports English (a-z) and Devanagari (Hindi/Marathi etc.) characters.
    """
    if provider not in TEXT_VALIDATION_PROVIDERS:
        return True
    real_chars = re.sub(r"[^a-zA-Z0-9\u0900-\u097F]", "", text)
    return len(real_chars) >= MIN_CHARS_FOR_TTS


class TTS(BaseTTS):
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
            raise ValueError("VIDEOSDK_AUTH_TOKEN environment variable must be set")

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
        self._keepalive_task: Optional[asyncio.Task] = None
        self._config_sent: bool = False
        self._connection_lock = asyncio.Lock()

        # Synthesis state
        self._interrupted: bool = False
        self._first_chunk_sent: bool = False
        self._has_error: bool = False

        self._synthesis_id: int = 0
        self._interrupted_at_id: int = -1

    # ==================== Factory Methods ====================

    @staticmethod
    def google(
        *,
        model_id="Chirp3-HD",
        voice_id="Achernar",
        language="en-US",
        speed=1.0,
        pitch=0.0,
        sample_rate=24000,
        enable_streaming=True,
        base_url=None,
        config=None,
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
            "model_id": model_id,
            **(config or {}),
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
        model_id="bulbul:v2",
        speaker="anushka",
        language="en-IN",
        sample_rate=24000,
        enable_streaming=True,
        base_url=None,
        config=None,
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
            **(config or {}),
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
        model_id="sonic-2",
        voice_id="faf0731e-dfb9-4cfc-8119-259a79b27e12",
        language="en",
        sample_rate=24000,
        enable_streaming=True,
        base_url=None,
        config=None,
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
            **(config or {}),
        }
        return TTS(
            provider="cartesia",
            model_id=model_id,
            voice_id=str(voice_id),
            language=language,
            config=config,
            enable_streaming=enable_streaming,
            sample_rate=sample_rate,
            base_url=base_url,
        )

    @staticmethod
    def deepgram(
        *,
        model_id="aura-2",
        voice_id="amalthea",
        language="en",
        encoding="linear16",
        sample_rate=24000,
        container="none",
        bit_rate=None,
        enable_streaming=True,
        base_url=None,
        config=None,
    ) -> "TTS":
        """
        Create a TTS instance configured for Deepgram Aura.

        Args:
            model_id: Deepgram Aura model (default: "aura-2")
            encoding: Audio encoding format (default: "linear16")
            sample_rate: Audio sample rate in Hz (default: 24000)
            container: Container format (default: "none" for raw audio)
            bit_rate: Bitrate in bps for compressed formats (optional)
            enable_streaming: Enable streaming mode (default: True)
            base_url: Custom inference gateway URL (optional)

        Returns:
            Configured TTS instance for Deepgram
        """
        config = {
            "model": model_id,
            "encoding": encoding,
            "sample_rate": sample_rate,
            "container": container,
            "voice_id": voice_id,
            "language": language,
            **(config or {}),
        }
        if bit_rate is not None:
            config["bit_rate"] = bit_rate
        return TTS(
            provider="deepgram",
            model_id=model_id,
            voice_id=voice_id,
            language="en",
            config=config,
            enable_streaming=enable_streaming,
            sample_rate=sample_rate,
            base_url=base_url,
        )

    # ==================== Core ====================

    def reset_first_audio_tracking(self) -> None:
        self._first_chunk_sent = False

    async def warmup(self) -> None:
        """
        Pre-warm the WebSocket connection before the first synthesis request.
        Call this right after session start to eliminate cold-start latency
        (~3-4s) on the first user turn.
        """
        logger.info(f"[InferenceTTS] Warming up connection (provider={self.provider})")
        try:
            await self._ensure_connection()
            logger.info("[InferenceTTS] Warmup complete — connection ready")
        except Exception as e:
            logger.warning(f"[InferenceTTS] Warmup failed (non-fatal): {e}")

    async def synthesize(
        self, text: AsyncIterator[str] | str, voice_id=None, **kwargs
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

        self._synthesis_id += 1
        current_id = self._synthesis_id

        self._interrupted = False
        self.reset_first_audio_tracking()
        logger.debug(f"[InferenceTTS] New synthesis started, id={current_id}")

        if isinstance(text, str):
            if not text.strip():
                logger.debug("[InferenceTTS] Skipping synthesis — empty text")
                return
            if not _has_enough_content(text, self.provider):
                logger.warning(
                    f"[InferenceTTS] Skipping — text too short for {self.provider}: '{text}'"
                )
                return

        text_for_retry = text if isinstance(text, str) else None

        for attempt in range(2):
            # Abort if a newer synthesis has already started
            if self._synthesis_id != current_id:
                logger.debug("[InferenceTTS] Synthesis superseded — aborting")
                return

            try:
                await self._ensure_connection()

                if self._synthesis_id != current_id:
                    logger.debug(
                        "[InferenceTTS] Synthesis superseded after connect — aborting"
                    )
                    return

                if isinstance(text, str):
                    await self._send_text(text, current_id)
                else:
                    await self._send_text_stream(text, current_id)
                return

            except ConnectionError as e:
                if attempt == 0 and text_for_retry is not None:
                    logger.warning(
                        f"[InferenceTTS] Connection lost mid-synthesis, retrying... ({e})"
                    )
                    self._has_error = True
                    await asyncio.sleep(0.05)
                    continue
                logger.error(f"[InferenceTTS] Synthesis failed after retry: {e}")
                self.emit("error", str(e))
                return

            except Exception as e:
                logger.error(f"[InferenceTTS] Synthesis error: {e}")
                self.emit("error", str(e))
                return

    # ==================== Connection ====================

    def _is_connection_alive(self) -> bool:
        return (
            self._ws is not None
            and not self._ws.closed
            and self._recv_task is not None
            and not self._recv_task.done()
            and not self._has_error
        )

    async def _ensure_connection(self) -> None:
        """Ensure WebSocket connection is established."""
        async with self._connection_lock:
            if self._is_connection_alive():
                logger.info("[InferenceTTS] Connection alive, reusing")
                return
            logger.info("[InferenceTTS] Connection not alive — reconnecting...")
            await self._teardown_connection()
            await self._connect_ws()
            self._recv_task = asyncio.create_task(self._recv_loop())
            await self._send_config()
            # FIX 1: Reduced from 100ms to 10ms — just enough for the server
            # to process the config frame before the first text arrives.
            await asyncio.sleep(0.01)
            self._has_error = False

    async def _connect_ws(self) -> None:
        """Establish WebSocket connection to the inference gateway."""
        if not self._session:
            self._session = aiohttp.ClientSession()

        ws_url = (
            f"{self.base_url}/v1/tts"
            f"?provider={self.provider}"
            f"&secret={self._videosdk_token}"
            f"&modelId={self.model_id}"
        )
        if self.voice_id:
            ws_url += f"&voiceId={self.voice_id}"

        logger.info(
            f"[InferenceTTS] Connecting to {self.base_url} (provider={self.provider})"
        )
        self._ws = await self._session.ws_connect(ws_url, heartbeat=20)
        self._config_sent = False
        logger.info("[InferenceTTS] Connected successfully")

        if self._keepalive_task and not self._keepalive_task.done():
            self._keepalive_task.cancel()
        self._keepalive_task = asyncio.create_task(self._keepalive())

    async def _keepalive(self) -> None:
        """Ping every 15s — Sarvam closes idle connections at ~60s."""
        while True:
            await asyncio.sleep(15)
            if self._ws and not self._ws.closed:
                try:
                    await self._ws.ping()
                except Exception:
                    logger.warning("[InferenceTTS] Keepalive ping failed")
                    break
            else:
                break

    async def _teardown_connection(self) -> None:
        if self._keepalive_task and not self._keepalive_task.done():
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass
            self._keepalive_task = None

        recv_task = self._recv_task
        self._recv_task = None
        if recv_task and not recv_task.done():
            recv_task.cancel()
            try:
                await recv_task
            except asyncio.CancelledError:
                pass

        ws = self._ws
        self._ws = None
        self._config_sent = False
        if ws and not ws.closed:
            try:
                await ws.close()
            except Exception:
                pass

    # ==================== Send ====================

    async def _send_config(self) -> None:
        if not self._ws or self._ws.closed:
            raise ConnectionError("WebSocket not connected")
        await self._ws.send_str(json.dumps({"type": "config", "data": self.config}))
        self._config_sent = True
        logger.info(f"[InferenceTTS] Config sent: {self.config}")

    async def _send_text(self, text: str, synthesis_id: int) -> None:
        text = text.strip()
        if not text:
            return
        # Guard: don't send if superseded or interrupted
        if self._synthesis_id != synthesis_id or self._interrupted:
            return
        if not _has_enough_content(text, self.provider):
            logger.warning(
                f"[InferenceTTS] Dropping short chunk for {self.provider}: '{text}'"
            )
            return
        if not self._ws or self._ws.closed:
            raise ConnectionError("WebSocket closed before text could be sent")
        await self._ws.send_str(json.dumps({"type": "text", "data": text}))
        await self._ws.send_str(json.dumps({"type": "flush"}))
        logger.debug(f"[InferenceTTS] Sent text + flush: '{text[:80]}'")

    async def _send_text_stream(
        self, text_iterator: AsyncIterator[str], synthesis_id: int
    ) -> None:
        buffer = []
        # FIX 2: Reduced from 3 words / 300ms to 2 words / 100ms.
        # This gets the first chunk to TTS ~200ms sooner on every streamed turn.
        MIN_WORDS = 2
        MAX_DELAY = 0.1
        last_send_time = asyncio.get_event_loop().time()
        first_chunk_sent = False

        try:
            async for chunk in text_iterator:
                # Stop immediately if interrupted or superseded
                if self._interrupted or self._synthesis_id != synthesis_id:
                    logger.debug(
                        "[InferenceTTS] Stream aborted — interrupted or superseded"
                    )
                    return

                if not chunk or not chunk.strip():
                    continue
                buffer.extend(chunk.split())
                now = asyncio.get_event_loop().time()

                combined = " ".join(buffer).strip()

                if not first_chunk_sent:
                    has_content = _has_enough_content(combined, self.provider)
                    should_send = has_content and (
                        len(buffer) >= MIN_WORDS or (now - last_send_time > MAX_DELAY)
                    )
                else:
                    should_send = len(buffer) >= MIN_WORDS or (
                        now - last_send_time > MAX_DELAY
                    )

                if should_send and combined:
                    await self._send_text(combined, synthesis_id)
                    first_chunk_sent = True
                    buffer.clear()
                    last_send_time = now

            # Flush remaining buffer only if still active
            if buffer and not self._interrupted and self._synthesis_id == synthesis_id:
                combined = " ".join(buffer).strip()
                if combined:
                    await self._send_text(combined, synthesis_id)
            # FIX 3: Removed the redundant trailing flush here.
            # _send_text already sends a flush after every chunk, so this
            # was causing a duplicate flush on the final chunk every time.

        except Exception as e:
            logger.error(f"[InferenceTTS] Stream send error: {e}")
            raise

    # ==================== Receive ====================

    async def _recv_loop(self) -> None:
        logger.debug("[InferenceTTS] Receive loop started")
        try:
            while self._ws and not self._ws.closed:
                msg = await self._ws.receive()
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_message(msg.data)
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    if (
                        not self._interrupted
                        and self._synthesis_id > self._interrupted_at_id
                        and self.audio_track
                    ):
                        await self.audio_track.add_new_bytes(msg.data)
                    else:
                        logger.debug(
                            "[InferenceTTS] Discarding stale binary audio chunk"
                        )
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(
                        f"[InferenceTTS] WebSocket error: {self._ws.exception()}"
                    )
                    break
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
                    logger.info("[InferenceTTS] WebSocket closed by server")
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[InferenceTTS] Receive loop error: {e}")
        finally:
            logger.info("[InferenceTTS] Receive loop terminated")
            self._ws = None
            self._config_sent = False

    async def _handle_message(self, raw_message: str) -> None:
        """Handle incoming messages from the inference server."""
        try:
            if not raw_message or not raw_message.strip():
                return

            try:
                data = json.loads(raw_message)
            except json.JSONDecodeError:
                logger.debug(
                    f"[InferenceTTS] Received non-JSON message: {raw_message[:100]}"
                )
                if "success" in raw_message.lower() or "ok" in raw_message.lower():
                    logger.debug("[InferenceTTS] Received acknowledgment")
                    return
                logger.warning(
                    f"[InferenceTTS] Unexpected non-JSON message: {raw_message[:200]}"
                )
                return

            msg_type = data.get("type")

            if msg_type == "audio":
                await self._handle_audio(data.get("data", {}))

            elif msg_type == "event":
                if data.get("data", {}).get("eventType") == "TTS_COMPLETE":
                    logger.debug("[InferenceTTS] Synthesis completed")

            elif msg_type == "error":
                error_msg = data.get("data", {}).get("error") or data.get(
                    "message", "Unknown error"
                )
                logger.error(f"[InferenceTTS] Server error: {error_msg}")
                self.emit("error", error_msg)
                logger.warning(
                    "[InferenceTTS] Forcing full reconnect due to provider error"
                )
                self._has_error = True
                if self._ws and not self._ws.closed:
                    try:
                        await self._ws.close()
                    except Exception:
                        pass

        except json.JSONDecodeError:
            pass
        except Exception as e:
            logger.error(f"[InferenceTTS] Message handling error: {e}")

    async def _handle_audio(self, audio_data: Dict[str, Any]) -> None:
        if self._interrupted or self._synthesis_id <= self._interrupted_at_id:
            logger.debug("[InferenceTTS] Discarding stale/interrupted audio")
            return
        if not audio_data:
            return
        audio_b64 = audio_data.get("audio")
        if not audio_b64:
            return
        try:
            audio_bytes = base64.b64decode(audio_b64)
            audio_bytes = self._remove_wav_header(audio_bytes)
            if not self._first_chunk_sent and self._first_audio_callback:
                self._first_chunk_sent = True
                await self._first_audio_callback()
            if self.audio_track and not self._interrupted:
                await self.audio_track.add_new_bytes(audio_bytes)
        except Exception as e:
            logger.error(f"[InferenceTTS] Audio processing error: {e}")

    def _remove_wav_header(self, audio_bytes: bytes) -> bytes:
        """Remove WAV header if present."""
        if audio_bytes.startswith(b"RIFF"):
            data_pos = audio_bytes.find(b"data")
            if data_pos != -1:
                return audio_bytes[data_pos + 8 :]
        return audio_bytes

    # ==================== Control ====================

    async def interrupt(self) -> None:
        """Interrupt ongoing synthesis."""
        self._interrupted = True
        self._interrupted_at_id = self._synthesis_id
        logger.debug(
            f"[InferenceTTS] Stamped interrupted_at_id={self._interrupted_at_id}"
        )

        if self.audio_track:
            self.audio_track.interrupt()

    async def aclose(self) -> None:
        logger.info(f"[InferenceTTS] Closing TTS (provider={self.provider})")
        self._interrupted = True
        await self._teardown_connection()
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
        await super().aclose()
        logger.info("[InferenceTTS] Closed successfully")

    @property
    def label(self) -> str:
        return f"videosdk.inference.TTS.{self.provider}.{self.model_id}"
