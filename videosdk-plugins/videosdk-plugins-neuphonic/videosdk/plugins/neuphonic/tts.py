from __future__ import annotations

from typing import Any, AsyncIterator, Literal, Optional, Union
import os
import json
import logging
import aiohttp
import asyncio
import base64
from urllib.parse import urlencode

from videosdk.agents import TTS, FlushMarker

logger = logging.getLogger(__name__)

NEUPHONIC_DEFAULT_SAMPLE_RATE = 22050
NEUPHONIC_CHANNELS = 1
NEUPHONIC_BASE_URL = "wss://eu-west-1.api.neuphonic.com"
NEUPHONIC_SSE_BASE_URL = "https://eu-west-1.api.neuphonic.com"
DEFAULT_CONNECTION_MAX_AGE_SEC = 300.0


class NeuphonicTTS(TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        lang_code: str = "en",
        voice_id: Optional[str] = None,
        speed: float = 0.8,
        sampling_rate: int = NEUPHONIC_DEFAULT_SAMPLE_RATE,
        encoding: Literal["pcm_linear", "pcm_mulaw"] = "pcm_linear",
        base_url: str = NEUPHONIC_BASE_URL,
        max_connection_age_sec: float = DEFAULT_CONNECTION_MAX_AGE_SEC,
    ) -> None:
        """Initialize the Neuphonic TTS plugin.

        Args:
            api_key (Optional[str], optional): Neuphonic API key. Defaults to None.
            lang_code (str): The language code to use for the TTS plugin. Defaults to "en".
            voice_id (Optional[str], optional): The voice ID to use for the TTS plugin. Defaults to None.
            speed (float): The speed to use for the TTS plugin. Must be between 0.7 and 2.0. Defaults to 0.8.
            sampling_rate (int): The sampling rate to use for the TTS plugin. Must be one of: 8000, 16000, 22050. Defaults to 22050.
            encoding (Literal["pcm_linear", "pcm_mulaw"]): The encoding to use for the TTS plugin. Defaults to "pcm_linear".
            base_url (str): The base URL to use for the TTS plugin. Defaults to "wss://eu-west-1.api.neuphonic.com".
            max_connection_age_sec (float): Refresh the WebSocket after this many seconds
                to avoid hitting Neuphonic's idle/session limits.
        """
        super().__init__(sample_rate=sampling_rate, num_channels=NEUPHONIC_CHANNELS)

        self.lang_code = lang_code
        self.voice_id = voice_id
        self.speed = speed
        self.encoding = encoding
        self.base_url = base_url
        self.audio_track = None
        self.loop = None
        self._first_chunk_sent = False
        self._interrupted = False
        self._max_connection_age_sec = max_connection_age_sec

        self._ws_session: Optional[aiohttp.ClientSession] = None
        self._ws_connection: Optional[aiohttp.ClientWebSocketResponse] = None
        self._ws_connect_time: float = 0.0
        self._connection_lock = asyncio.Lock()
        self._receive_task: Optional[asyncio.Task] = None
        self._active_future: Optional[asyncio.Future[None]] = None

        self.api_key = api_key or os.getenv("NEUPHONIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Neuphonic API key must be provided either through api_key parameter "
                "or NEUPHONIC_API_KEY environment variable"
            )

        if not 0.7 <= self.speed <= 2.0:
            raise ValueError(
                f"Speed must be between 0.7 and 2.0, got {self.speed}")

        if sampling_rate not in [8000, 16000, 22050]:
            raise ValueError(
                f"Sampling rate must be one of 8000, 16000, 22050, got {sampling_rate}")

    def reset_first_audio_tracking(self) -> None:
        """Reset the first audio tracking state for next TTS task"""
        self._first_chunk_sent = False
        self._interrupted = False

    async def prewarm(self) -> None:
        """Pre-establish the Neuphonic WebSocket so the first ``synthesize()``
        call does not pay the TLS + auth + upgrade cost. Safe to call repeatedly."""
        try:
            await self._ensure_ws_connection()
        except Exception as e:
            logger.warning(f"Neuphonic TTS prewarm failed (non-fatal): {e}")

    def _build_ws_url(self) -> str:
        params = {
            "api_key": self.api_key,
            "speed": self.speed,
            "sampling_rate": self._sample_rate,
            "encoding": self.encoding,
        }
        if self.voice_id:
            params["voice_id"] = self.voice_id
        return f"{self.base_url}/speak/{self.lang_code}?{urlencode(params)}"

    async def _ensure_ws_connection(self) -> None:
        async with self._connection_lock:
            now = asyncio.get_event_loop().time()

            if self._ws_connection and not self._ws_connection.closed:
                age = now - self._ws_connect_time
                if age < self._max_connection_age_sec:
                    return
                logger.info(f"Refreshing Neuphonic WebSocket (age={age:.1f}s)")
                await self._close_connection_locked()
            elif self._ws_connection or self._ws_session:
                await self._close_connection_locked()

            try:
                self._ws_session = aiohttp.ClientSession()
                self._ws_connection = await asyncio.wait_for(
                    self._ws_session.ws_connect(
                        self._build_ws_url(), heartbeat=30.0
                    ),
                    timeout=10.0,
                )
                self._ws_connect_time = now
                self._receive_task = asyncio.create_task(self._receive_loop())
            except Exception as e:
                logger.error(f"Failed to establish Neuphonic WebSocket: {e}")
                self.emit("error", f"Failed to establish WebSocket connection: {e}")
                if self._ws_session and not self._ws_session.closed:
                    try:
                        await self._ws_session.close()
                    except Exception:
                        pass
                self._ws_session = None
                self._ws_connection = None
                raise

    async def _close_connection_locked(self) -> None:
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except (asyncio.CancelledError, Exception):
                pass
        self._receive_task = None
        if self._ws_connection and not self._ws_connection.closed:
            try:
                await self._ws_connection.close()
            except Exception:
                pass
        self._ws_connection = None
        if self._ws_session and not self._ws_session.closed:
            try:
                await self._ws_session.close()
            except Exception:
                pass
        self._ws_session = None

    async def synthesize(
        self,
        text: AsyncIterator[Union[str, FlushMarker]] | str,
        **kwargs: Any,
    ) -> None:
        """Synthesize text via Neuphonic's WebSocket. Each ``FlushMarker`` (and
        the end-of-stream) maps to a ``<STOP>`` boundary token, the provider's
        per-sentence flush primitive."""
        try:
            if not self.audio_track or not self.loop:
                self.emit("error", "Audio track or event loop not set")
                return

            self._interrupted = False
            await self._ensure_ws_connection()
            if not self._ws_connection or self._ws_connection.closed:
                raise RuntimeError("WebSocket connection is not available.")

            done_future: asyncio.Future[None] = asyncio.get_event_loop().create_future()
            self._active_future = done_future

            try:
                if isinstance(text, str):
                    if text.strip() and not self._interrupted:
                        await self._ws_connection.send_str(f"{text} <STOP>")
                else:
                    pending: list[str] = []
                    async for chunk in text:
                        if self._interrupted:
                            break
                        if isinstance(chunk, FlushMarker):
                            # Drain accumulated buffer to a STOP boundary —
                            # Neuphonic's per-sentence flush primitive.
                            if pending:
                                joined = "".join(pending).strip()
                                pending = []
                                if joined:
                                    await self._ws_connection.send_str(f"{joined} <STOP>")
                            continue
                        if not chunk:
                            continue
                        pending.append(chunk)
                    if pending and not self._interrupted:
                        joined = "".join(pending).strip()
                        if joined:
                            await self._ws_connection.send_str(f"{joined} <STOP>")

                # Give the server up to 30s to drain remaining frames.
                try:
                    await asyncio.wait_for(asyncio.shield(done_future), timeout=30.0)
                except asyncio.TimeoutError:
                    if not done_future.done():
                        done_future.set_result(None)
                except asyncio.CancelledError:
                    pass
            finally:
                if self._active_future is done_future:
                    self._active_future = None

        except Exception as e:
            self.emit("error", f"TTS synthesis failed: {str(e)}")
            raise

    async def _receive_loop(self) -> None:
        """Continuously read audio frames from the persistent WebSocket."""
        try:
            while self._ws_connection and not self._ws_connection.closed:
                msg = await self._ws_connection.receive()
                if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING):
                    break
                if msg.type == aiohttp.WSMsgType.ERROR:
                    raise ConnectionError(f"WebSocket error: {self._ws_connection.exception()}")
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    if not self._interrupted:
                        self.emit("error", f"Invalid JSON response: {msg.data}")
                    continue

                # Late frames for cancelled contexts are silently dropped.
                fut = self._active_future
                if self._interrupted or fut is None or fut.done():
                    continue

                inner = data.get("data") if isinstance(data, dict) else None
                if isinstance(inner, dict):
                    audio_b64 = inner.get("audio")
                    if audio_b64:
                        try:
                            audio_data = base64.b64decode(audio_b64)
                            await self._stream_audio_chunks(audio_data)
                        except Exception as e:
                            logger.error(f"Failed to decode/stream audio: {e}")
                    if inner.get("stop"):
                        # Server signaled end of the current segment — resolve
                        # the active future so synthesize() returns.
                        if fut is not None and not fut.done():
                            fut.set_result(None)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if not self._interrupted:
                logger.error(f"Neuphonic receive loop error: {e}")
            fut = self._active_future
            if fut is not None and not fut.done():
                fut.set_exception(e)

    async def _stream_audio_chunks(self, audio_bytes: bytes) -> None:
        """Stream audio data in chunks for smooth playback"""
        if self._interrupted or not audio_bytes:
            return

        chunk_duration_ms = 20
        bytes_per_sample = 2
        chunk_size = int(self._sample_rate * NEUPHONIC_CHANNELS *
                         bytes_per_sample * chunk_duration_ms / 1000)

        if chunk_size % 2 != 0:
            chunk_size += 1

        for i in range(0, len(audio_bytes), chunk_size):
            if self._interrupted:
                return
            chunk = audio_bytes[i:i + chunk_size]

            if len(chunk) < chunk_size and len(chunk) > 0:
                padding_needed = chunk_size - len(chunk)
                chunk += b'\x00' * padding_needed

            if len(chunk) == chunk_size:
                if not self._first_chunk_sent and self._first_audio_callback:
                    self._first_chunk_sent = True
                    await self._first_audio_callback()

                asyncio.create_task(self.audio_track.add_new_bytes(chunk))
                await asyncio.sleep(0.001)

    async def aclose(self) -> None:
        """Cleanup resources"""
        self._interrupted = True
        async with self._connection_lock:
            await self._close_connection_locked()
        await super().aclose()

    async def interrupt(self) -> None:
        """Stop emitting audio for the current synthesis. Keeps the WebSocket
        open so the next turn does not pay reconnect cost; in-flight audio
        frames are dropped via the active-future filter in the receive loop."""
        self._interrupted = True

        if self._active_future is not None and not self._active_future.done():
            self._active_future.cancel()

        if self.audio_track:
            self.audio_track.interrupt()
