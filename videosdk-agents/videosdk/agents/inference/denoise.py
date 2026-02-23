from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import time
import logging
from collections import deque
from typing import Any, Optional, Dict

import aiohttp
import numpy as np
from scipy.signal import resample_poly
from videosdk.agents.denoise import Denoise as BaseDenoise

logger = logging.getLogger(__name__)

VIDEOSDK_INFERENCE_URL = "wss://dev-inference.videosdk.live"

_SPARROW_MODELS = frozenset(
    {
        "sparrow-xxs-48khz",
        "sparrow-s-48khz",
        "sparrow-l-48khz",
    }
)
_ALL_AICOUSTICS_MODELS = _SPARROW_MODELS
_EXPECTED_SAMPLE_RATE: dict[str, int] = {**{m: 48000 for m in _SPARROW_MODELS}}


class DenoiseLatencyTracker:

    _WINDOW = 500

    def __init__(self, log_every_n_chunks: int = 100) -> None:
        # FIFO of send timestamps — one entry per sent chunk
        self._send_times: deque[float] = deque()

        # Rolling latency samples
        self._samples: deque[float] = deque(maxlen=self._WINDOW)

        self._log_every = log_every_n_chunks
        self._recv_count = 0
        self._min_ms = float("inf")
        self._max_ms = 0.0
        self._dropped = 0

    def record_send(self) -> None:
        """
        Push current timestamp immediately after a chunk is sent to the server.
        Must be called exactly once per _send_audio() call.
        """
        self._send_times.append(time.perf_counter())

    def record_receive(self) -> float | None:
        """
        Pop the oldest send timestamp and compute round-trip latency.

        Returns:
            Latency in milliseconds for this specific chunk, or None if
            there was no matching send (indicates a bug — log a warning).
        """
        if not self._send_times:
            self._dropped += 1
            logger.warning(
                f"[DenoiseLatency] Received chunk with no matching send timestamp "
                f"(total dropped={self._dropped}). "
                f"This may happen on reconnect — should resolve automatically."
            )
            return None

        sent_at = self._send_times.popleft()  # FIFO: oldest send = this chunk
        latency_ms = (time.perf_counter() - sent_at) * 1000

        self._samples.append(latency_ms)
        self._recv_count += 1

        if latency_ms < self._min_ms:
            self._min_ms = latency_ms
        if latency_ms > self._max_ms:
            self._max_ms = latency_ms

        # Periodic rolling log — every N received chunks
        if self._recv_count % self._log_every == 0:
            logger.info(
                f"[Denoise][Latency] "
                f"chunks_received={self._recv_count} | "
                f"last={latency_ms:.1f}ms | "
                f"avg(last {self._WINDOW})={self.avg_ms:.1f}ms | "
                f"min={self._min_ms:.1f}ms | "
                f"max={self._max_ms:.1f}ms | "
                f"queue_depth={self.queue_depth}"
            )

        return latency_ms

    @property
    def avg_ms(self) -> float:
        """Rolling average over the last _WINDOW chunks."""
        return sum(self._samples) / len(self._samples) if self._samples else 0.0

    @property
    def last_ms(self) -> float:
        return self._samples[-1] if self._samples else 0.0

    @property
    def queue_depth(self) -> int:
        """
        Number of sent chunks still waiting for a matching receive.
        Under normal operation: 0–2.
        Growing number = server is falling behind or losing chunks.
        """
        return len(self._send_times)

    def summary(self) -> dict:
        return {
            "recv_count": self._recv_count,
            "avg_ms": round(self.avg_ms, 2),
            "min_ms": round(self._min_ms, 2) if self._recv_count else 0,
            "max_ms": round(self._max_ms, 2),
            "last_ms": round(self.last_ms, 2),
            "queue_depth": self.queue_depth,
            "dropped": self._dropped,
        }

    def reset(self) -> None:
        """Clear all state — call after reconnect to avoid stale timestamp mismatches."""
        self._send_times.clear()
        self._samples.clear()
        self._recv_count = 0
        self._min_ms = float("inf")
        self._max_ms = 0.0
        self._dropped = 0


class Denoise(BaseDenoise):
    """
    VideoSDK Inference Gateway Denoise Plugin — with accurate latency tracking.

    Latency is measured as true per-chunk round-trip time:
        send_time[chunk_N]  →  receive_time[chunk_N]

    Use get_latency_stats() or get_stats() to read current numbers.
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

        # Pre-computed resampling ratios
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
        self._needs_resample_out = (
            self.sample_rate != self.input_sample_rate
            or self.channels != self.input_channels
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

        self._audio_buffer: asyncio.Queue = asyncio.Queue(maxsize=50)

        self._latency = DenoiseLatencyTracker(log_every_n_chunks=100)

        self._stats = {
            "chunks_sent": 0,
            "bytes_sent": 0,
            "chunks_received": 0,
            "bytes_received": 0,
            "errors": 0,
            "reconnections": 0,
        }

        logger.info(
            f"[InferenceDenoise] Initialized: provider={provider}, "
            f"model={model_id}, sample_rate={sample_rate}Hz, channels={channels}"
        )

    @staticmethod
    def aicoustics(
        *,
        model_id: str = "sparrow-xxs-48khz",
        sample_rate: int | None = None,
        channels: int = 1,
        base_url: str | None = None,
    ) -> "Denoise":
        if model_id not in _ALL_AICOUSTICS_MODELS:
            raise ValueError(
                f"Unknown AI-Coustics model_id: '{model_id}'. "
                f"Valid options: {sorted(_ALL_AICOUSTICS_MODELS)}"
            )
        expected_rate = _EXPECTED_SAMPLE_RATE[model_id]
        if sample_rate is None:
            sample_rate = expected_rate
        elif sample_rate != expected_rate:
            raise ValueError(
                f"Model '{model_id}' requires sample_rate={expected_rate} Hz, "
                f"got {sample_rate} Hz. Omit sample_rate to auto-detect."
            )
        if channels != 1:
            raise ValueError(
                f"AI-Coustics only supports mono (channels=1), got channels={channels}."
            )
        config = {"model": model_id, "sample_rate": sample_rate, "channels": channels}
        return Denoise(
            provider="aicoustics",
            model_id=model_id,
            sample_rate=sample_rate,
            channels=channels,
            chunk_ms=10,
            config=config,
            base_url=base_url if base_url is not None else VIDEOSDK_INFERENCE_URL,
        )

    @staticmethod
    def krisp(
        *,
        model_id: str = "krisp-viva-tel-v2",
        sample_rate: int = 16000,
        channels: int = 1,
        base_url: str | None = None,
    ) -> "Denoise":
        return Denoise(
            provider="krisp",
            model_id=model_id,
            sample_rate=sample_rate,
            channels=channels,
            base_url=base_url if base_url is not None else VIDEOSDK_INFERENCE_URL,
        )

    @staticmethod
    def sanas(
        *,
        model_id: str = "VI_G_NC3.0",
        sample_rate: int = 16000,
        channels: int = 1,
        base_url: str | None = None,
    ) -> "Denoise":
        return Denoise(
            provider="sanas",
            model_id=model_id,
            sample_rate=sample_rate,
            channels=channels,
            chunk_ms=20,
            base_url=base_url if base_url is not None else VIDEOSDK_INFERENCE_URL,
        )

    async def connect(self) -> None:
        """Eagerly establish the WebSocket. Call once at startup."""
        async with self._connect_lock:
            if self.connected and self._ws and not self._ws.closed:
                return
            await self._connect_ws()
            self.connected = True
            self._ws_task = asyncio.create_task(self._listen_for_responses())
            logger.info("[InferenceDenoise] Ready — eager connection established.")

    async def denoise(self, audio_frames: bytes, **kwargs: Any) -> bytes:
        try:
            if not self._ws or self._ws.closed:
                if self.connecting:
                    return b""
                self.connecting = True
                try:
                    await self.connect()
                finally:
                    self.connecting = False

            if not self._ws_task or self._ws_task.done():
                self._ws_task = asyncio.create_task(self._listen_for_responses())

            if not self._config_sent:
                await self._send_config()

            # Inbound resample
            resampled = (
                self._resample_fast(
                    audio_frames,
                    self._in_up,
                    self._in_down,
                    self.input_channels,
                    self.channels,
                )
                if self._needs_resample_in
                else audio_frames
            )

            # Chunk, send, record timestamp
            self._send_buffer.extend(resampled)
            while len(self._send_buffer) >= self._send_chunk_bytes:
                chunk = bytes(self._send_buffer[: self._send_chunk_bytes])
                del self._send_buffer[: self._send_chunk_bytes]
                await self._send_audio(chunk)

                self._latency.record_send()

            try:
                denoised = self._audio_buffer.get_nowait()
                self._stats["chunks_received"] += 1
                self._stats["bytes_received"] += len(denoised)

                return (
                    self._resample_fast(
                        denoised,
                        self._out_up,
                        self._out_down,
                        self.channels,
                        self.input_channels,
                    )
                    if self._needs_resample_out
                    else denoised
                )

            except asyncio.QueueEmpty:
                return b""

        except Exception as e:
            logger.error(f"[InferenceDenoise] Error in denoise: {e}", exc_info=True)
            self._stats["errors"] += 1
            self.emit("error", str(e))
            if self._stats["errors"] % 10 == 0:
                asyncio.create_task(self._reconnect())
            return b""

    def get_latency_stats(self) -> dict:
        """
        Returns accurate per-chunk round-trip latency stats.

        Keys:
            recv_count  — number of denoised chunks received so far
            avg_ms      — rolling average over last 500 received chunks
            min_ms      — all-time minimum round-trip
            max_ms      — all-time maximum round-trip
            last_ms     — latency of the most recently received chunk
            queue_depth — sent chunks awaiting a receive (0-2 = healthy)
            dropped     — receives with no matching send (should be 0)
        """
        return self._latency.summary()

    async def _connect_ws(self) -> None:
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
            logger.info(
                f"[InferenceDenoise] WS headers: {dict(self._ws._response.headers)}"
            )
            self._config_sent = False
            logger.info("[InferenceDenoise] Connected successfully")
        except Exception as e:
            logger.error(f"[InferenceDenoise] Connection failed: {e}", exc_info=True)
            raise

    async def _reconnect(self) -> None:
        if self._reconnecting:
            return
        self._reconnecting = True
        backoff = 0.5
        try:
            for attempt in range(1, 6):
                try:
                    logger.info(
                        f"[InferenceDenoise] Reconnect attempt {attempt} "
                        f"(backoff={backoff}s)..."
                    )
                    await self._cleanup_connection()
                    await asyncio.sleep(backoff)
                    await self._connect_ws()
                    if self._ws_task:
                        self._ws_task.cancel()
                    self._ws_task = asyncio.create_task(self._listen_for_responses())

                    self._latency.reset()
                    self._stats["reconnections"] += 1
                    self.connected = True
                    logger.info(
                        f"[InferenceDenoise] Reconnected "
                        f"(attempt {self._stats['reconnections']})"
                    )
                    return
                except Exception as e:
                    logger.error(f"[InferenceDenoise] Attempt {attempt} failed: {e}")
                    backoff = min(backoff * 2, 8.0)
            logger.error("[InferenceDenoise] All reconnection attempts failed.")
        finally:
            self._reconnecting = False

    async def _send_config(self) -> None:
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
                f"[InferenceDenoise] Config sent: model={self.model_id}, "
                f"sample_rate={self.sample_rate}Hz, channels={self.channels}"
            )
        except Exception as e:
            logger.error(f"[InferenceDenoise] Failed to send config: {e}")
            raise

    async def _send_audio(self, audio_bytes: bytes) -> None:
        if not self._ws:
            raise ConnectionError("WebSocket not connected")
        try:
            await self._ws.send_str(
                json.dumps(
                    {
                        "type": "audio",
                        "data": base64.b64encode(audio_bytes).decode("utf-8"),
                    }
                )
            )
            self._stats["chunks_sent"] += 1
            self._stats["bytes_sent"] += len(audio_bytes)
        except Exception as e:
            logger.error(f"[InferenceDenoise] Failed to send audio: {e}")
            raise

    async def _listen_for_responses(self) -> None:
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
            logger.error(f"[InferenceDenoise] Listener error: {e}", exc_info=True)
            self.emit("error", str(e))
        finally:
            await self._cleanup_connection()

    async def _handle_message(self, raw_message: str) -> None:
        try:
            data = json.loads(raw_message)
            msg_type = data.get("type")

            if msg_type == "event":
                event_data = data.get("data", {})
                if event_data.get("eventType") == "DENOISE_AUDIO":
                    audio_data = event_data.get("audio", "")
                    if audio_data:
                        denoised_bytes = base64.b64decode(audio_data)

                        latency_ms = self._latency.record_receive()
                        if latency_ms is not None and latency_ms > 200:
                            logger.warning(
                                f"[InferenceDenoise] High chunk latency detected: "
                                f"{latency_ms:.1f}ms | "
                                f"queue_depth={self._latency.queue_depth}"
                            )

                        # Bounded buffer — drop oldest frame if full
                        try:
                            self._audio_buffer.put_nowait(denoised_bytes)
                        except asyncio.QueueFull:
                            try:
                                self._audio_buffer.get_nowait()
                            except asyncio.QueueEmpty:
                                pass
                            self._audio_buffer.put_nowait(denoised_bytes)

            elif msg_type == "audio":
                audio_data = data.get("data", "")
                if audio_data:
                    denoised_bytes = base64.b64decode(audio_data)
                    self._latency.record_receive()
                    try:
                        self._audio_buffer.put_nowait(denoised_bytes)
                    except asyncio.QueueFull:
                        try:
                            self._audio_buffer.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                        self._audio_buffer.put_nowait(denoised_bytes)

            elif msg_type == "error":
                error_data = data.get("data", {})
                logger.error(
                    f"[InferenceDenoise] Server error — full response: {json.dumps(data)}"
                )
                error_msg = error_data.get("error") or error_data.get(
                    "message", "Unknown error"
                )
                self.emit("error", error_msg)
                self._stats["errors"] += 1

        except json.JSONDecodeError as e:
            logger.error(
                f"[InferenceDenoise] Failed to parse message: {e} | "
                f"raw: {raw_message[:100]}"
            )
        except Exception as e:
            logger.error(f"[Denoise] Message handling error: {e}", exc_info=True)

    def _resample_fast(
        self,
        audio_bytes: bytes,
        up: int,
        down: int,
        src_channels: int,
        dst_channels: int,
    ) -> bytes:
        if not audio_bytes:
            return b""
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        if audio.size == 0:
            return b""
        remainder = audio.size % src_channels
        if remainder:
            audio = audio[:-remainder]
        audio = audio.reshape(-1, src_channels).astype(np.float32)
        if src_channels != dst_channels:
            if dst_channels == 1:
                audio = audio.mean(axis=1, keepdims=True)
            elif src_channels == 1:
                audio = np.repeat(audio, dst_channels, axis=1)
            else:
                audio = audio[:, : min(src_channels, dst_channels)]
                if dst_channels > audio.shape[1]:
                    pad = np.zeros(
                        (audio.shape[0], dst_channels - audio.shape[1]),
                        dtype=np.float32,
                    )
                    audio = np.hstack([audio, pad])
        if up != down:
            audio = resample_poly(audio, up, down, axis=0)
        return np.clip(audio, -32768, 32767).astype(np.int16).tobytes()

    async def _cleanup_connection(self) -> None:
        if self._ws and not self._ws.closed:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = None
        self._config_sent = False
        self.connected = False

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "buffer_size": self._audio_buffer.qsize(),
            "provider": self.provider,
            "model": self.model_id,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "connected": self._ws is not None and not self._ws.closed,
            "latency": self._latency.summary(),
        }

    async def flush(self) -> None:
        if self._send_buffer and self._ws and not self._ws.closed:
            padded = bytes(self._send_buffer).ljust(self._send_chunk_bytes, b"\x00")
            await self._send_audio(padded)
            self._latency.record_send()
            self._send_buffer.clear()
        if self._ws and not self._ws.closed:
            try:
                await self._ws.send_str(json.dumps({"type": "stop"}))
                deadline = time.perf_counter() + 0.05
                while time.perf_counter() < deadline:
                    if not self._audio_buffer.empty():
                        break
                    await asyncio.sleep(0.005)
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
        return f"videosdk.inference.Denoise.{self.provider}.{self.model_id}"
