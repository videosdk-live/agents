from __future__ import annotations

import asyncio
import aiohttp
import json
import logging
from typing import Any, AsyncIterator, Union, Optional
import os
from videosdk.agents import TTS, FlushMarker

logger = logging.getLogger(__name__)

DEEPGRAM_SAMPLE_RATE = 24000
DEEPGRAM_CHANNELS = 1
DEFAULT_MODEL = "aura-2-andromeda-en"
DEFAULT_ENCODING = "linear16"
API_BASE_URL = "wss://api.deepgram.com/v1/speak"
DEFAULT_CONNECTION_MAX_AGE_SEC = 300.0


class DeepgramTTS(TTS):
    def __init__(
            self,
            *,
            api_key: str | None = None,
            model: str = DEFAULT_MODEL,
            encoding: str = DEFAULT_ENCODING,
            sample_rate: int = DEEPGRAM_SAMPLE_RATE,
            base_url: str = API_BASE_URL,
            max_connection_age_sec: float = DEFAULT_CONNECTION_MAX_AGE_SEC,
            **kwargs: Any,
    ) -> None:
        """Initialize the Deepgram TTS plugin.

        Args:
            api_key: Deepgram API key. Falls back to ``DEEPGRAM_API_KEY`` env var.
            model: Deepgram TTS voice/model id. Defaults to ``aura-2-andromeda-en``.
            encoding: Output audio encoding. Defaults to ``linear16``.
            sample_rate: Output sample rate.
            base_url: WebSocket base URL.
            max_connection_age_sec: Refresh the WebSocket after this many seconds
                to avoid hitting Deepgram's idle/session limits.
        """
        super().__init__(sample_rate=sample_rate, num_channels=DEEPGRAM_CHANNELS)

        self.model = model
        self.encoding = encoding
        self.base_url = base_url
        self.audio_track = None
        self.loop = None
        self._max_connection_age_sec = max_connection_age_sec

        self._ws_session: aiohttp.ClientSession | None = None
        self._ws_connection: aiohttp.ClientWebSocketResponse | None = None
        self._ws_connect_time: float = 0.0
        self._connection_lock = asyncio.Lock()
        self._receive_task: asyncio.Task | None = None

        self._active_future: asyncio.Future[None] | None = None
        self._active_send_task: asyncio.Task | None = None

        self._interrupted = False
        self._first_chunk_sent = False

        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Deepgram API key must be provided either through the 'api_key' parameter or the DEEPGRAM_API_KEY environment variable."
            )

    def reset_first_audio_tracking(self) -> None:
        self._first_chunk_sent = False

    async def prewarm(self) -> None:
        """Pre-establish the Deepgram WebSocket so the first ``synthesize()`` call
        does not pay the TLS + auth + upgrade cost. Safe to call repeatedly."""
        try:
            await self._ensure_connection()
        except Exception as e:
            logger.warning(f"Deepgram TTS prewarm failed (non-fatal): {e}")

    async def _ensure_connection(self) -> None:
        async with self._connection_lock:
            now = asyncio.get_event_loop().time()

            if self._ws_connection and not self._ws_connection.closed:
                age = now - self._ws_connect_time
                if age < self._max_connection_age_sec:
                    return
                logger.info(f"Refreshing Deepgram WebSocket (age={age:.1f}s)")
                await self._close_connection_locked()
            elif self._ws_connection or self._ws_session:
                await self._close_connection_locked()

            params = {
                "model": self.model,
                "encoding": self.encoding,
                "sample_rate": self.sample_rate,
            }
            param_string = "&".join([f"{k}={v}" for k, v in params.items()])
            full_ws_url = f"{self.base_url}?{param_string}"
            headers = {"Authorization": f"Token {self.api_key}"}

            self._ws_session = aiohttp.ClientSession()
            self._ws_connection = await asyncio.wait_for(
                self._ws_session.ws_connect(
                    full_ws_url, headers=headers, heartbeat=30.0
                ),
                timeout=10.0,
            )
            self._ws_connect_time = now
            self._receive_task = asyncio.create_task(self._receive_audio_task())

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
            voice_id: Optional[str] = None,
            **kwargs: Any,
    ) -> None:
        """Synthesize text via Deepgram's streaming WebSocket API.

        Each ``FlushMarker`` in the input stream is forwarded to Deepgram as a
        ``{"type": "Flush"}`` message, letting the server emit audio for the
        completed sentence without waiting for end-of-stream.
        """
        try:
            if not self.audio_track or not self.loop:
                self.emit("error", "Audio track or event loop not set")
                return

            self._interrupted = False
            await self._ensure_connection()
            if not self._ws_connection:
                raise RuntimeError("WebSocket connection is not available.")

            done_future: asyncio.Future[None] = asyncio.get_event_loop().create_future()
            self._active_future = done_future

            send_task = asyncio.create_task(self._send_text_task(text, done_future))
            self._active_send_task = send_task

            try:
                await done_future
            except asyncio.CancelledError:
                pass

            await send_task

        except Exception as e:
            self.emit("error", f"TTS synthesis failed: {str(e)}")
            raise
        finally:
            if self._active_future is done_future:
                self._active_future = None
            if self._active_send_task is send_task:
                self._active_send_task = None

    async def _send_text_task(
        self,
        text: Union[AsyncIterator[Union[str, FlushMarker]], str],
        done_future: asyncio.Future[None],
    ) -> None:
        if not self._ws_connection or self._ws_connection.closed:
            if not done_future.done():
                done_future.set_exception(RuntimeError("WebSocket closed"))
            return

        has_sent = False
        try:
            if isinstance(text, str):
                if text and not self._interrupted:
                    await self._ws_connection.send_json({"type": "Speak", "text": text})
                    has_sent = True
            else:
                async for chunk in text:
                    if self._interrupted or self._ws_connection.closed:
                        break
                    if isinstance(chunk, FlushMarker):
                        if has_sent:
                            await self._ws_connection.send_json({"type": "Flush"})
                        continue
                    if not chunk:
                        continue
                    await self._ws_connection.send_json({"type": "Speak", "text": chunk})
                    has_sent = True
                    
            if has_sent and not self._interrupted and not self._ws_connection.closed:
                await self._ws_connection.send_json({"type": "Flush"})

            if not done_future.done():
                try:
                    await asyncio.wait_for(asyncio.shield(done_future), timeout=30.0)
                except asyncio.TimeoutError:
                    if not done_future.done():
                        done_future.set_result(None)
                except asyncio.CancelledError:
                    pass

        except asyncio.CancelledError:
            if not done_future.done():
                done_future.cancel()
            raise
        except Exception as e:
            if not self._interrupted:
                self.emit("error", f"Send task error: {str(e)}")
            if not done_future.done():
                done_future.set_exception(e)

    async def _receive_audio_task(self) -> None:
        if not self._ws_connection:
            return

        try:
            while self._ws_connection and not self._ws_connection.closed:
                msg = await self._ws_connection.receive()

                if msg.type == aiohttp.WSMsgType.BINARY:
                    fut = self._active_future
                    if not self._interrupted and fut is not None and not fut.done():
                        await self._stream_audio_chunks(msg.data)
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    msg_type = data.get("type")
                    if msg_type == "Error":
                        err = data.get("description", "Unknown error")
                        if not self._interrupted:
                            self.emit("error", f"Deepgram error: {err}")
                        fut = self._active_future
                        if fut is not None and not fut.done():
                            fut.set_exception(RuntimeError(err))
                    elif msg_type == "Flushed":
                        fut = self._active_future
                        if fut is not None and not fut.done():
                            fut.set_result(None)
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING):
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    raise ConnectionError(f"WebSocket error: {self._ws_connection.exception()}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if not self._interrupted:
                self.emit("error", f"Receive task error: {str(e)}")
            fut = self._active_future
            if fut is not None and not fut.done():
                fut.set_exception(e)

    async def _stream_audio_chunks(self, audio_bytes: bytes) -> None:
        if not audio_bytes or self._interrupted:
            return

        if not self._first_chunk_sent and self._first_audio_callback:
            self._first_chunk_sent = True
            await self._first_audio_callback()

        if self.audio_track and self.loop:
            await self.audio_track.add_new_bytes(audio_bytes)

    async def interrupt(self) -> None:
        """Stop emitting audio for the current synthesis. Keeps the WebSocket
        open so the next turn does not pay reconnect cost; in-flight audio
        frames are dropped via the ``_active_future`` filter in the receive
        loop."""
        self._interrupted = True

        if self.audio_track:
            self.audio_track.interrupt()

        if self._active_send_task and not self._active_send_task.done():
            self._active_send_task.cancel()

        if self._active_future is not None and not self._active_future.done():
            self._active_future.cancel()

    async def aclose(self) -> None:
        self._interrupted = True

        if self._active_send_task and not self._active_send_task.done():
            self._active_send_task.cancel()

        async with self._connection_lock:
            await self._close_connection_locked()

        await super().aclose()
