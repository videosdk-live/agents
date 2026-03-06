from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import time
import logging
from typing import Any, Optional, Dict

import aiohttp
import numpy as np
from scipy.signal import resample_poly
from videosdk.agents.denoise import Denoise as BaseDenoise

logger = logging.getLogger(__name__)

VIDEOSDK_INFERENCE_URL = "wss://dev-inference.videosdk.live"


class Denoise(BaseDenoise):
    """
    VideoSDK Inference Gateway Denoise Plugin.

    A lightweight noise cancellation client that connects to VideoSDK's Inference Gateway.
    Supports SANAS and AI-Coustics noise cancellation through a unified interface.

    Example:
        # Using factory methods (recommended)
        denoise = Denoise.aicoustics(model_id="sparrow-xxs-48khz")

        # Using generic constructor
        denoise = Denoise(
            provider="aicoustics",
            model_id="sparrow-xxs-48khz",
            config={"sample_rate": 48000}
        )

        # Use in pipeline
        pipeline = CascadingPipeline(
            stt=DeepgramSTT(sample_rate=48000),
            llm=GoogleLLM(),
            tts=ElevenLabsTTS(),
            vad=SileroVAD(input_sample_rate=48000),
            turn_detector=TurnDetector(),
            denoise=denoise
        )
    """

    def __init__(
        self,
        *,
        provider: str,
        model_id: str,
        sample_rate: int = 48000,
        channels: int = 1,
        chunk_ms: int = 10,
        config: Dict[str, Any] | None = None,
        base_url: str | None = None,
    ) -> None:
        """
        Initialize the VideoSDK Inference Denoise plugin.

        Args:
            provider: Denoise provider name (e.g., "aicoustics")
            model_id: Model identifier for the provider
            sample_rate: Audio sample rate in Hz (default: 48000)
            channels: Number of audio channels (default: 1 for mono)
            config: Provider-specific configuration dictionary
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
        self.sample_rate = sample_rate
        self.channels = channels
        self.input_sample_rate = 48000
        self.input_channels = 2
        self.config = config or {}
        self.base_url = base_url or VIDEOSDK_INFERENCE_URL
        self.chunk_ms = chunk_ms

        self._send_chunk_bytes: int = (
            self.chunk_ms * self.sample_rate // 1000 * self.channels * 2
        )
        self._send_buffer: bytearray = bytearray()

        # Pre-compute resampling ratios to avoid math.gcd on every call
        self._in_gcd = math.gcd(self.input_sample_rate, self.sample_rate)
        self._in_up = self.sample_rate // self._in_gcd
        self._in_down = self.input_sample_rate // self._in_gcd

        self._out_gcd = math.gcd(self.sample_rate, self.input_sample_rate)
        self._out_up = self.input_sample_rate // self._out_gcd
        self._out_down = self.sample_rate // self._out_gcd

        self._needs_resample_in = (
            self.input_sample_rate != self.sample_rate
            or self.input_channels != self.channels
        )

        # WebSocket state
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self.connected = False
        self.connecting = False
        self._ws_task: Optional[asyncio.Task] = None
        self._connect_lock = asyncio.Lock()
        self._config_sent: bool = False
        self._reconnecting: bool = False

        # Audio buffering for responses
        self._audio_buffer: asyncio.Queue = asyncio.Queue()

        # Latency tracking
        self._send_timestamps: asyncio.Queue = asyncio.Queue()  # timestamps of sent chunks
        self._latency_samples: list[float] = []
        self._total_chunks_latency: float = 0.0
        self._latency_count: int = 0

        self._stats = {
            "chunks_sent": 0,
            "bytes_sent": 0,
            "chunks_received": 0,
            "bytes_received": 0,
            "errors": 0,
            "reconnections": 0,
            "avg_latency_ms": 0.0,
            "min_latency_ms": float("inf"),
            "max_latency_ms": 0.0,
            "last_latency_ms": 0.0,
        }

        logger.info(
            f"[InferenceDenoise] Initialized: provider={provider}, "
            f"model={model_id}, sample_rate={sample_rate}Hz, channels={channels}"
        )

    # ==================== Factory Methods ====================
    @staticmethod
    def aicoustics(
        *,
        model_id: str = "sparrow-xxs-48khz",
        sample_rate: int = 48000,
        channels: int = 1,
        base_url: str | None = None,
    ) -> "Denoise":
        """
        Create a Denoise instance configured for AI-Coustics.

        Args:
            model_id: AI-Coustics model (default: "sparrow-xxs-48khz")
                Sparrow family (human-to-human, 48kHz):
                - "sparrow-xxs-48khz": Ultra-fast, 10ms latency, 1MB
                - "sparrow-s-48khz": Small, 30ms latency, 8.96MB
                - "sparrow-l-48khz": Large, best quality, 30ms latency, 35.1MB

                Quail family (human-to-machine, voice AI, 16kHz):
                - "quail-vf-l-16khz": Voice focus + STT optimization, 35MB
                - "quail-l-16khz": General purpose, 35MB
                - "quail-s-16khz": Faster, 8.88MB

            sample_rate: Audio sample rate in Hz
                - Sparrow models: 48000 Hz (default)
                - Quail models: 16000 Hz
            channels: Number of audio channels (default: 1 for mono)
            base_url: Custom inference gateway URL

        Returns:
            Configured Denoise instance for AI-Coustics

        Example:
            >>> # Ultra-fast for real-time calls
            >>> denoise = Denoise.aicoustics(model_id="sparrow-xxs-48khz")
            >>>
            >>> # Best quality for recordings
            >>> denoise = Denoise.aicoustics(model_id="sparrow-l-48khz")
            >>>
            >>> # Voice AI / STT optimization (16kHz)
            >>> denoise = Denoise.aicoustics(
            ...     model_id="quail-vf-l-16khz",
            ...     sample_rate=16000
            ... )
        """
        config = {
            "model": model_id,
            "sample_rate": sample_rate,
            "channels": channels,
        }

        return Denoise(
            provider="aicoustics",
            model_id=model_id,
            sample_rate=sample_rate,
            channels=channels,
            chunk_ms=10,
            config=config,
            base_url=base_url or VIDEOSDK_INFERENCE_URL,
        )

    @staticmethod
    def krisp(
        *,
        model_id: str = "krisp-viva-tel-v2",
        sample_rate: int = 16000,
        channels: int = 1,
        base_url: str | None = None,
    ) -> "Denoise":

        """
            Create a Denoise instance configured for Krisp.

            Args:
                model_id: Krisp model (default: "krisp-viva-tel-v2")

                sample_rate: Audio sample rate in Hz
                    - krisp-viva-tel-v2 - 16000 for noise cancellation
                channels: Number of audio channels (default: 1 for mono)
                base_url: Custom inference gateway URL

            Returns:
                Configured Denoise instance for Krisp

            Example:
                >>> # Ultra-fast for real-time calls
                >>> denoise = Denoise.krisp(model_id="krisp-viva-tel-v2")
                >>>
                >>> # Best quality for recordings
                >>> denoise = Denoise.krisp(model_id="krisp-viva-tel-v2")
                >>>
                >>> # Voice AI / STT optimization (16kHz)
                >>> denoise = Denoise.krisp(
                ...     model_id="krisp-viva-tel-v2",
                ...     sample_rate=16000
                ... )
        """

        return Denoise(
            provider="krisp",
            model_id=model_id,
            sample_rate=sample_rate,
            channels=channels,
            base_url=base_url or VIDEOSDK_INFERENCE_URL,
        )
    
    @staticmethod
    def sanas(
        *,
        model_id: str = "VI_G_NC3.0",
        sample_rate: int = 16000,
        channels: int = 1,
        base_url: str | None = None,
    ) -> "Denoise":

        """
            Create a Denoise instance configured for Sanas.

            Args:
                model_id: Sanas model (default: "VI_G_NC3.0")

                sample_rate: Audio sample rate in Hz
                    - VI_G_NC3.0 - 16000 for noise cancellation
                channels: Number of audio channels (default: 1 for mono)
                base_url: Custom inference gateway URL

            Returns:
                Configured Denoise instance for Sanas

            Example:
                >>> # Ultra-fast for real-time calls
                >>> denoise = Denoise.aicoustics(model_id="VI_G_NC3.0")
                >>>
                >>> # Best quality for recordings
                >>> denoise = Denoise.aicoustics(model_id="VI_G_NC3.0")
                >>>
                >>> # Voice AI / STT optimization (16kHz)
                >>> denoise = Denoise.sanas(
                ...     model_id="VI_G_NC3.0",
                ...     sample_rate=16000
                ... )
            """

        return Denoise(
            provider="sanas",
            model_id=model_id,
            sample_rate=sample_rate,
            channels=channels,
            chunk_ms=20,
            base_url=base_url or VIDEOSDK_INFERENCE_URL,
        )

    # ==================== Core Denoise Methods ====================

    async def denoise(
        self,
        audio_frames: bytes,
        **kwargs: Any,
    ) -> bytes:
        """
        Process audio frames through noise cancellation.

        Args:
            audio_frames: Raw PCM audio bytes (int16 format)
            **kwargs: Additional arguments (unused)

        Returns:
            Denoised PCM audio bytes (int16 format), or original audio if processing fails
        """
        try:
            if self.connecting:
                return b""

            if not self._ws or self._ws.closed:
                self.connecting = True
                try:
                    await self._connect_ws()
                    self.connected = True
                finally:
                    self.connecting = False

            if not self._ws_task or self._ws_task.done():
                self._ws_task = asyncio.create_task(self._listen_for_responses())

            if not self._config_sent:
                await self._send_config()

            # --- Resample inbound audio (pre-computed ratios) ---
            if self._needs_resample_in:
                resampled = self._resample_fast(
                    audio_frames,
                    self._in_up,
                    self._in_down,
                    self.input_channels,
                    self.channels,
                )
            else:
                resampled = audio_frames

            # --- Chunk and send ---
            self._send_buffer.extend(resampled)
            while len(self._send_buffer) >= self._send_chunk_bytes:
                chunk = bytes(self._send_buffer[: self._send_chunk_bytes])
                del self._send_buffer[: self._send_chunk_bytes]
                await self._send_audio(chunk)
                # Record send timestamp for latency measurement
                await self._send_timestamps.put(time.perf_counter())

            # --- Non-blocking receive (zero timeout = no added latency) ---
            try:
                denoised = self._audio_buffer.get_nowait()

                # Calculate latency
                try:
                    sent_at = self._send_timestamps.get_nowait()
                    latency_ms = (time.perf_counter() - sent_at) * 1000
                    self._update_latency(latency_ms)
                except asyncio.QueueEmpty:
                    pass

                self._stats["chunks_received"] += 1
                self._stats["bytes_received"] += len(denoised)

                # Resample outbound back to original format
                return self._resample_fast(
                    denoised,
                    self._out_up,
                    self._out_down,
                    self.channels,
                    self.input_channels,
                )

            except asyncio.QueueEmpty:
                return b""

        except Exception as e:
            logger.error(f"[InferenceDenoise] Error in denoise: {e}", exc_info=True)
            self._stats["errors"] += 1
            self.emit("error", str(e))

            # Attempt reconnection on persistent errors
            if self._stats["errors"] % 10 == 0:
                logger.warning(
                    f"[InferenceDenoise] {self._stats['errors']} errors detected, "
                    "attempting reconnection..."
                )
                await self._reconnect()

            # Return empty bytes on error (don't pass through original to avoid glitches)
            return b""

    def _update_latency(self, latency_ms: float) -> None:
        """Update running latency statistics."""
        self._latency_count += 1
        self._total_chunks_latency += latency_ms
        self._stats["last_latency_ms"] = round(latency_ms, 2)
        self._stats["avg_latency_ms"] = round(
            self._total_chunks_latency / self._latency_count, 2
        )
        if latency_ms < self._stats["min_latency_ms"]:
            self._stats["min_latency_ms"] = round(latency_ms, 2)
        if latency_ms > self._stats["max_latency_ms"]:
            self._stats["max_latency_ms"] = round(latency_ms, 2)

        # Log every 100 chunks so you can monitor without spam
        if self._latency_count % 100 == 0:
            logger.info(
                f"[Denoise] Latency — "
                f"last={self._stats['last_latency_ms']}ms | "
                f"avg={self._stats['avg_latency_ms']}ms | "
                f"min={self._stats['min_latency_ms']}ms | "
                f"max={self._stats['max_latency_ms']}ms"
            )

    # ==================== WebSocket Communication ====================

    async def _connect_ws(self) -> None:
        """Establish WebSocket connection to the inference gateway."""
        if self._reconnecting:
            return  # Prevent concurrent reconnection attempts

        try:
            if not self._session or self._session.closed:
                self._session = aiohttp.ClientSession()

            ws_url = (
                f"{self.base_url}/v1/denoise"
                f"?provider={self.provider}"
                f"&secret={self._videosdk_token}"
                f"&modelId={self.model_id}"
            )

            logger.info(
                f"[InferenceDenoise] Connecting to {self.base_url} "
                f"(provider={self.provider}, model={self.model_id})"
            )

            self._ws = await self._session.ws_connect(ws_url)
            self._config_sent = False

            logger.info("[InferenceDenoise] Connected successfully")

        except Exception as e:
            logger.error(f"[InferenceDenoise] Connection failed: {e}", exc_info=True)
            raise

    async def _reconnect(self) -> None:
        """Attempt to reconnect to the inference gateway."""
        if self._reconnecting:
            return

        self._reconnecting = True
        try:
            logger.info("[InferenceDenoise] Reconnecting...")

            await self._cleanup_connection()

            # wait a bit before reconnecting
            await asyncio.sleep(1)

            await self._connect_ws()

            # restart listener
            if self._ws_task:
                self._ws_task.cancel()
            self._ws_task = asyncio.create_task(self._listen_for_responses())

            self._stats["reconnections"] += 1
            logger.info(
                f"[InferenceDenoise] Reconnected successfully "
                f"(attempt {self._stats['reconnections']})"
            )

        except Exception as e:
            logger.error(f"[InferenceDenoise] Reconnection failed: {e}")
        finally:
            self._reconnecting = False

    async def _send_config(self) -> None:
        """Send configuration message to the inference server."""
        if not self._ws:
            raise ConnectionError("WebSocket not connected")

        config_message = {
            "type": "config",
            "data": {
                "model": self.model_id,
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                **self.config,
            },
        }

        try:
            await self._ws.send_str(json.dumps(config_message))
            self._config_sent = True
            logger.info(
                f"[InferenceDenoise] Config sent: "
                f"model={self.model_id}, "
                f"sample_rate={self.sample_rate}Hz, "
                f"channels={self.channels}"
            )
        except Exception as e:
            logger.error(f"[InferenceDenoise] Failed to send config: {e}")
            raise

    async def _send_audio(self, audio_bytes: bytes) -> None:
        """Send audio data to the inference server."""
        if not self._ws:
            raise ConnectionError("WebSocket not connected")

        audio_message = {
            "type": "audio",
            "data": base64.b64encode(audio_bytes).decode("utf-8"),
        }

        try:
            await self._ws.send_str(json.dumps(audio_message))
            self._stats["chunks_sent"] += 1
            self._stats["bytes_sent"] += len(audio_bytes)
        except Exception as e:
            logger.error(f"[InferenceDenoise] Failed to send audio: {e}")
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
                        f"[InferenceDenoise] WebSocket error: {self._ws.exception()}"
                    )
                    self.emit("error", f"WebSocket error: {self._ws.exception()}")
                    break

                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.info("[InferenceDenoise] WebSocket closed by server")
                    break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(
                f"[InferenceDenoise] Error in WebSocket listener: {e}", exc_info=True
            )
            self.emit("error", str(e))

        finally:
            self._cleanup_connection()

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
                # Handle event messages (AI-Coustics format)
                event_data = data.get("data", {})
                event_type = event_data.get("eventType")

                if event_type == "DENOISE_AUDIO":
                    audio_data = event_data.get("audio", "")
                    if audio_data:
                        denoised_bytes = base64.b64decode(audio_data)
                        await self._audio_buffer.put(denoised_bytes)

                else:
                    logger.debug(f"[InferenceDenoise] Unknown event type: {event_type}")

            elif msg_type == "audio":
                # handle audio messages (alternative format)
                audio_data = data.get("data", "")
                if audio_data:
                    denoised_bytes = base64.b64decode(audio_data)
                    await self._audio_buffer.put(denoised_bytes)

            elif msg_type == "error":
                error_data = data.get("data", {})
                error_msg = error_data.get("error") or error_data.get(
                    "message", "Unknown error"
                )
                logger.error(f"[InferenceDenoise] Server error: {error_msg}")
                self.emit("error", error_msg)
                self._stats["errors"] += 1

        except json.JSONDecodeError as e:
            logger.error(
                f"[InferenceDenoise] Failed to parse message: {e} | "
                f"raw_message: {raw_message[:100]}"
            )
        except Exception as e:
            logger.error(f"[Denoise] Message handling error: {e}", exc_info=True)

    # ==================== Resampling ====================

    def _resample_fast(
        self,
        audio_bytes: bytes,
        up: int,
        down: int,
        src_channels: int,
        dst_channels: int,
    ) -> bytes:
        """Optimized resample using pre-computed up/down ratios."""
        if not audio_bytes:
            return b""

        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        if audio.size == 0:
            return b""

        remainder = audio.size % src_channels
        if remainder:
            audio = audio[:-remainder]

        audio = audio.reshape(-1, src_channels).astype(np.float32)

        # Channel conversion
        if src_channels != dst_channels:
            if dst_channels == 1:
                audio = audio.mean(axis=1, keepdims=True)
            elif src_channels == 1:
                audio = np.repeat(audio, dst_channels, axis=1)
            else:
                audio = audio[:, : min(src_channels, dst_channels)]
                if dst_channels > audio.shape[1]:
                    pad = np.zeros(
                        (audio.shape[0], dst_channels - audio.shape[1]), dtype=np.float32
                    )
                    audio = np.hstack([audio, pad])

        # Sample rate conversion
        if up != down:
            audio = resample_poly(audio, up, down, axis=0)

        return np.clip(audio, -32768, 32767).astype(np.int16).tobytes()

    # ==================== Lifecycle ====================

    async def _cleanup_connection(self) -> None:
        """Clean up WebSocket connection resources."""
        if self._ws and not self._ws.closed:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = None
        self._config_sent = False

    # ==================== Utility Methods ====================

    def get_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.

        Returns:
            Dictionary containing processing statistics
        """
        return {
            **self._stats,
            "buffer_size": self._audio_buffer.qsize(),
            "provider": self.provider,
            "model": self.model_id,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "connected": self._ws is not None and not self._ws.closed,
        }

    async def flush(self) -> None:
        if self._send_buffer and self._ws and not self._ws.closed:
            padded = bytes(self._send_buffer).ljust(self._send_chunk_bytes, b"\x00")
            await self._send_audio(padded)
            self._send_buffer.clear()
        if self._ws and not self._ws.closed:
            try:
                await self._ws.send_str(json.dumps({"type": "stop"}))
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"[Denoise] Flush error: {e}")

    async def aclose(self) -> None:
        logger.info(f"[Denoise] Closing. Final stats: {self.get_stats()}")
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
        await self._cleanup_connection()
        if self._session and not self._session.closed:
            await self._session.close()
        while not self._audio_buffer.empty():
            try:
                self._audio_buffer.get_nowait()
            except asyncio.QueueEmpty:
                break

        await super().aclose()


    @property
    def label(self) -> str:
        """Get a descriptive label for this Denoise instance."""
        return f"videosdk.inference.Denoise.{self.provider}.{self.model_id}"
