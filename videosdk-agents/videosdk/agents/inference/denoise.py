from __future__ import annotations

import asyncio
import base64
import json
import os
import logging
import time
from collections import deque
from typing import Any, Optional, Dict

import aiohttp

from videosdk.agents.denoise import Denoise as BaseDenoise

logger = logging.getLogger(__name__)

VIDEOSDK_INFERENCE_URL = "wss://inference-gateway.videosdk.live"

# Rolling window size for latency stats
LATENCY_WINDOW = 50


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
            raise ValueError("VIDEOSDK_AUTH_TOKEN environment variable must be set")

        self.provider = provider
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_ms = chunk_ms
        self.config = config or {}
        self.base_url = base_url or VIDEOSDK_INFERENCE_URL

        # WebSocket state
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._config_sent: bool = False
        self.connected: bool = False
        self._shutting_down: bool = False

        # Audio
        self._send_buffer: bytearray = bytearray()
        self._audio_buffer: asyncio.Queue = asyncio.Queue(maxsize=20)
        self._connect_lock: asyncio.Lock | None = None

        # Latency tracking
        # Maps send sequence number → send timestamp (monotonic)
        self._pending_chunks: dict[int, float] = {}
        self._send_seq: int = 0
        self._recv_seq: int = 0

        # Rolling window of round-trip latencies (ms)
        self._latency_window: deque[float] = deque(maxlen=LATENCY_WINDOW)

        # Stats
        self._stats = {
            "chunks_sent": 0,
            "bytes_sent": 0,
            "chunks_received": 0,
            "bytes_received": 0,
            "errors": 0,
            "reconnections": 0,
            "buffer_drops": 0,
            # Latency stats (ms)
            "latency_last_ms": 0.0,
            "latency_avg_ms": 0.0,
            "latency_min_ms": float("inf"),
            "latency_max_ms": 0.0,
            "latency_p95_ms": 0.0,
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

        return Denoise(
            provider="aicoustics",
            model_id=model_id,
            sample_rate=sample_rate,
            channels=channels,
            chunk_ms=10,
            config={},
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
            config={},
            base_url=base_url or VIDEOSDK_INFERENCE_URL,
        )

    # ==================== Latency Helpers ====================

    def _record_latency(self, latency_ms: float) -> None:
        """Update all latency stats with a new measurement."""
        self._latency_window.append(latency_ms)

        self._stats["latency_last_ms"] = round(latency_ms, 2)
        self._stats["latency_min_ms"] = round(
            min(self._stats["latency_min_ms"], latency_ms), 2
        )
        self._stats["latency_max_ms"] = round(
            max(self._stats["latency_max_ms"], latency_ms), 2
        )
        self._stats["latency_avg_ms"] = round(
            sum(self._latency_window) / len(self._latency_window), 2
        )

        # p95 over rolling window
        if len(self._latency_window) >= 2:
            sorted_w = sorted(self._latency_window)
            p95_idx = int(len(sorted_w) * 0.95)
            self._stats["latency_p95_ms"] = round(sorted_w[p95_idx], 2)

        logger.debug(
            f"[InferenceDenoise] Latency: last={latency_ms:.1f}ms  "
            f"avg={self._stats['latency_avg_ms']}ms  "
            f"min={self._stats['latency_min_ms']}ms  "
            f"max={self._stats['latency_max_ms']}ms  "
            f"p95={self._stats['latency_p95_ms']}ms"
        )

    def _reset_latency_state(self) -> None:
        """Clear pending chunk map on reconnect so stale timestamps don't pollute stats."""
        self._pending_chunks.clear()
        self._send_seq = 0
        self._recv_seq = 0

    # ==================== Core Denoise ====================

    async def denoise(self, audio_frames: bytes, **kwargs: Any) -> bytes:
        # logger.info(f"Using Sanas secret: {self._secret}")
        # print("enter in denoise")
        try:
            if self._connect_lock is None:
                self._connect_lock = asyncio.Lock()

            frame_size = len(audio_frames)

            if self._shutting_down:
                return audio_frames

            if not self._ws or self._ws.closed:
                if self._connect_lock.locked():
                    return audio_frames

                async with self._connect_lock:
                    if not self._ws or self._ws.closed:
                        try:
                            await self._connect_ws()
                            self.connected = True
                            self._stats["errors"] = 0
                            await self._send_config()

                            chunk_size = (
                                (self.chunk_ms * self.sample_rate // 1000)
                                * self.channels
                                * 2
                            )
                            self._send_buffer.extend(audio_frames)
                            if len(self._send_buffer) >= chunk_size:
                                first_chunk = bytes(self._send_buffer[:chunk_size])
                                del self._send_buffer[:chunk_size]
                                await self._send_audio(first_chunk)

                            if not self._ws_task or self._ws_task.done():
                                self._ws_task = asyncio.create_task(
                                    self._listen_for_responses()
                                )
                            logger.info(
                                f"[InferenceDenoise] Ready (provider={self.provider})"
                            )
                        except Exception as e:
                            logger.error(f"[InferenceDenoise] Setup failed: {e}")
                            self._ws = None
                            self._config_sent = False
                            self._send_buffer.clear()
                            return audio_frames

            if not self._config_sent:
                return audio_frames

            chunk_size = (self.chunk_ms * self.sample_rate // 1000) * self.channels * 2
            self._send_buffer.extend(audio_frames)

            while len(self._send_buffer) >= chunk_size:
                chunk = bytes(self._send_buffer[:chunk_size])
                del self._send_buffer[:chunk_size]
                try:
                    await self._send_audio(chunk)
                except Exception as e:
                    logger.error(f"[InferenceDenoise] Send failed: {e} — resetting")
                    await asyncio.sleep(0.5)
                    self._ws = None
                    self._config_sent = False
                    self._send_buffer.clear()
                    self._reset_latency_state()
                    return audio_frames

            denoised_chunks = []
            while not self._audio_buffer.empty():
                try:
                    denoised_chunks.append(self._audio_buffer.get_nowait())
                except asyncio.QueueEmpty:
                    break

            if denoised_chunks:
                all_denoised = b"".join(denoised_chunks)
                total = len(all_denoised)
                self._stats["chunks_received"] += len(denoised_chunks)
                self._stats["bytes_received"] += total

                if total > frame_size:
                    excess = all_denoised[frame_size:]
                    for i in range(0, len(excess), frame_size):
                        piece = excess[i : i + frame_size]
                        if self._audio_buffer.full():
                            try:
                                self._audio_buffer.get_nowait()
                                self._stats["buffer_drops"] += 1
                            except asyncio.QueueEmpty:
                                pass
                        try:
                            self._audio_buffer.put_nowait(piece)
                        except asyncio.QueueFull:
                            pass
                    return all_denoised[:frame_size]

                return all_denoised

            return audio_frames

        except Exception as e:
            logger.error(f"[InferenceDenoise] Error in denoise: {e}", exc_info=True)
            self._stats["errors"] += 1
            return audio_frames

    # ==================== WebSocket ====================

    async def _connect_ws(self) -> None:
        try:
            if self._shutting_down:
                return

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

            self._ws = await self._session.ws_connect(
                ws_url, timeout=aiohttp.ClientTimeout(total=10)
            )
            if self._shutting_down:
                await self._ws.close()
                return
            self._config_sent = False
            self._send_buffer.clear()
            self._reset_latency_state()
            logger.info("[InferenceDenoise] Connected successfully")

        except Exception as e:
            logger.error(f"[InferenceDenoise] Connection failed: {e}", exc_info=True)
            raise

    async def _send_config(self) -> None:
        """Send configuration message to the inference server."""
        if not self._ws or self._ws.closed:
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
        await self._ws.send_str(json.dumps(config_message))
        self._config_sent = True
        logger.info(
            f"[InferenceDenoise] Config sent: "
            f"model={self.model_id}, sample_rate={self.sample_rate}Hz, channels={self.channels}"
        )

    async def _send_audio(self, audio_bytes: bytes) -> None:
        """Send one audio chunk, stamping it with a sequence number for latency tracking."""
        if not self._ws or self._ws.closed:
            raise ConnectionError("WebSocket not connected")

        seq = self._send_seq
        self._send_seq += 1

        # Record send timestamp BEFORE the await so network time is included
        self._pending_chunks[seq] = time.monotonic()

        await self._ws.send_str(
            json.dumps(
                {
                    "type": "audio",
                    "data": base64.b64encode(audio_bytes).decode("utf-8"),
                    "seq": seq,  # server will echo this back if it supports it
                }
            )
        )
        self._stats["chunks_sent"] += 1
        self._stats["bytes_sent"] += len(audio_bytes)

    def _resolve_latency(self, recv_seq: int | None = None) -> None:
        """
        Match a received chunk to a sent chunk and record the round-trip latency.

        If the server echoes 'seq', we match exactly.
        Otherwise we consume the oldest pending timestamp (FIFO approximation).
        """
        now = time.monotonic()

        if recv_seq is not None and recv_seq in self._pending_chunks:
            sent_at = self._pending_chunks.pop(recv_seq)
        elif self._pending_chunks:
            # FIFO: oldest sent chunk corresponds to oldest received chunk
            oldest_seq = min(self._pending_chunks)
            sent_at = self._pending_chunks.pop(oldest_seq)
        else:
            return  # no pending chunk to match

        latency_ms = (now - sent_at) * 1000
        self._record_latency(latency_ms)

    async def _listen_for_responses(self) -> None:
        """Background task to listen for WebSocket responses from the server."""
        if not self._ws:
            return

        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_message(msg.data)
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    # Binary frames carry raw denoised PCM — measure latency (FIFO)
                    self._resolve_latency()
                    if self._audio_buffer.full():
                        try:
                            self._audio_buffer.get_nowait()
                            self._stats["buffer_drops"] += 1
                        except asyncio.QueueEmpty:
                            pass
                    try:
                        self._audio_buffer.put_nowait(msg.data)
                    except asyncio.QueueFull:
                        pass
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(
                        f"[InferenceDenoise] WebSocket error: {self._ws.exception()}"
                    )
                    break
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
                    logger.info("[InferenceDenoise] WebSocket closed by server")
                    break

        except asyncio.CancelledError:
            logger.debug("[InferenceDenoise] Listener cancelled")
        except Exception as e:
            logger.error(f"[InferenceDenoise] Listener error: {e}", exc_info=True)
        finally:
            self._ws = None
            self._config_sent = False
            logger.info("[InferenceDenoise] Listener exited — connection marked dead")

    async def _handle_message(self, raw_message: str) -> None:
        """
        Handle incoming messages from the inference server.

        Args:
            raw_message: Raw JSON message string from server
        """
        # logger.info(f"[STT DEBUG] raw server msg: {raw_message}")
        try:
            data = json.loads(raw_message)
            msg_type = data.get("type")

            if msg_type == "event":
                event_data = data.get("data", {})
                event_type = event_data.get("eventType")

                if event_type == "DENOISE_AUDIO":
                    audio_data = event_data.get("audio", "")
                    # Echo'd seq from server (optional — works without it too)
                    recv_seq = event_data.get("seq", None)
                    if audio_data:
                        self._resolve_latency(recv_seq)
                        denoised = base64.b64decode(audio_data)
                        if self._audio_buffer.full():
                            try:
                                self._audio_buffer.get_nowait()
                                self._stats["buffer_drops"] += 1
                            except asyncio.QueueEmpty:
                                pass
                        try:
                            self._audio_buffer.put_nowait(denoised)
                        except asyncio.QueueFull:
                            pass
                # START_SPEECH, END_SPEECH, TRANSCRIPT silently ignored

            elif msg_type == "audio":
                audio_data = data.get("data", "")
                recv_seq = data.get("seq", None)
                if audio_data:
                    self._resolve_latency(recv_seq)
                    denoised = base64.b64decode(audio_data)
                    if self._audio_buffer.full():
                        try:
                            self._audio_buffer.get_nowait()
                            self._stats["buffer_drops"] += 1
                        except asyncio.QueueEmpty:
                            pass
                    try:
                        self._audio_buffer.put_nowait(denoised)
                    except asyncio.QueueFull:
                        pass

            elif msg_type == "error":
                # logger.error(f"[InferenceDenoise] FULL ERROR MESSAGE: {raw_message}")
                error_data = data.get("data", {})

                # Safely extract error message
                error_msg = (
                    error_data.get("error")
                    or error_data.get("message")
                    or json.dumps(error_data)
                    or "Unknown error"
                )

                self._stats["errors"] += 1

                logger.error(
                    f"[InferenceDenoise] Server error: {error_msg} "
                    f"(total: {self._stats['errors']})"
                )

                # Force reset connection on first error
                if self._stats["errors"] == 1:
                    self._send_buffer.clear()
                    self._config_sent = False

                    if self._ws and not self._ws.closed:
                        try:
                            await self._ws.close()
                        except Exception:
                            pass

                    self._ws = None

        except json.JSONDecodeError as e:
            logger.error(f"[InferenceDenoise] Failed to parse message: {e}")
        except Exception as e:
            logger.error(
                f"[InferenceDenoise] Message handling error: {e}", exc_info=True
            )

    async def _cleanup_connection(self) -> None:
        if self._ws and not self._ws.closed:
            try:
                await asyncio.wait_for(
                    self._ws.send_str(json.dumps({"type": "stop"})), timeout=1.0
                )
                await asyncio.sleep(0.1)
            except Exception:
                pass
            try:
                await self._ws.close()
            except Exception:
                pass

        self._ws = None
        self._config_sent = False
        self._send_buffer.clear()

    # ==================== Utilities ====================

    def get_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.

        Returns:
            Dictionary containing processing statistics
        """
        return {
            **self._stats,
            "buffer_size": self._audio_buffer.qsize(),
            "pending_chunks": len(self._pending_chunks),
            "provider": self.provider,
            "model": self.model_id,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "connected": self._ws is not None and not self._ws.closed,
        }

    def get_latency_stats(self) -> Dict[str, Any]:
        """Return only the latency-related stats — handy for logging/monitoring."""
        return {
            "last_ms": self._stats["latency_last_ms"],
            "avg_ms": self._stats["latency_avg_ms"],
            "min_ms": self._stats["latency_min_ms"],
            "max_ms": self._stats["latency_max_ms"],
            "p95_ms": self._stats["latency_p95_ms"],
            "samples": len(self._latency_window),
        }

    async def aclose(self) -> None:
        logger.info(
            f"[InferenceDenoise] Closing (provider={self.provider}). "
            f"Final stats: {self.get_stats()}"
        )
        self._shutting_down = True
        # Log final latency summary on close
        lat = self.get_latency_stats()
        if lat["samples"] > 0:
            logger.info(
                f"[InferenceDenoise] Latency summary — "
                f"avg={lat['avg_ms']}ms  p95={lat['p95_ms']}ms  "
                f"min={lat['min_ms']}ms  max={lat['max_ms']}ms  "
                f"over {lat['samples']} samples"
            )

        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            try:
                await asyncio.wait_for(self._ws_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._ws_task = None

        await self._cleanup_connection()

        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

        while not self._audio_buffer.empty():
            try:
                self._audio_buffer.get_nowait()
            except asyncio.QueueEmpty:
                break

        await super().aclose()
        logger.info("[InferenceDenoise] Closed successfully")

    @property
    def label(self) -> str:
        return f"videosdk.inference.Denoise.{self.provider}.{self.model_id}"
