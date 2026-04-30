from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from typing import Any, AsyncIterator, Literal, Optional
from urllib.parse import urlencode

import aiohttp

from videosdk.agents import TTS

logger = logging.getLogger(__name__)

XAI_TTS_BASE_URL = "wss://api.x.ai/v1/tts"
XAI_TTS_NUM_CHANNELS = 1
SUPPORTED_VOICES = {"eve", "ara", "rex", "sal", "leo"}
SUPPORTED_SAMPLE_RATES = {8000, 16000, 22050, 24000, 44100, 48000}

SUPPORTED_CODECS = {"pcm", "mulaw"}


class XAITTS(TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        voice: str = "eve",
        language: str = "en",
        codec: Literal["pcm", "mulaw"] = "pcm",
        sample_rate: int = 24000,
        optimize_streaming_latency: int = 0,
        text_normalization: bool = False,
        base_url: str = XAI_TTS_BASE_URL,
    ) -> None:
        """Initialize the xAI TTS plugin.

        Args:
            api_key: xAI API key. Falls back to XAI_API_KEY env var.
            voice: Voice ID — one of "eve", "ara", "rex", "sal", "leo". Case-insensitive.
            language: BCP-47 language code (e.g. "en", "fr", "pt-BR") or "auto" for
                automatic language detection. Required by xAI.
            codec: Output codec. Restricted to "pcm" (signed 16-bit LE, default) or
                "mulaw" — both are raw, byte-streamable formats compatible with the
                framework's audio_track. mp3/wav/alaw are not exposed because they
                require a decoder before bytes can be played.
            sample_rate: Output sample rate in Hz. One of 8000/16000/22050/24000/44100/48000.
                Defaults to 24000 (xAI's recommended rate).
            optimize_streaming_latency: 0 (default, best quality) or 1 (lower
                time-to-first-audio with minor quality tradeoff).
            text_normalization: When true, xAI normalizes written-form text
                (numbers, abbreviations, symbols) into spoken-form before synthesis.
            base_url: WebSocket endpoint URL.

        Speech tags:
            xAI supports inline expression tags ([pause], [long-pause], [laugh],
            [sigh], [breath], etc.) and wrapping style tags (<whisper>...</whisper>,
            <soft>, <loud>, <slow>, <fast>, <higher-pitch>, <lower-pitch>,
            <emphasis>, <singing>, <sing-song>, <laugh-speak>, <build-intensity>,
            <decrease-intensity>) directly inside the `text` you pass to
            `synthesize()`. No separate parameter is needed — the tags are sent
            verbatim as part of each text.delta message and parsed server-side.

            Example::

                await tts.synthesize(
                    "So I walked in and [pause] there it was. [laugh] Incredible!"
                )
                await tts.synthesize(
                    "I need to tell you something. "
                    "<whisper>It is a secret.</whisper> Pretty cool, right?"
                )

            Caveat for streaming input: when synthesize() receives an
            AsyncIterator[str] (e.g. LLM tokens), a single tag can be split across
            two chunks ("[pa", "use]") which xAI will not recognize. Tags only work
            reliably when an entire tag arrives within one text chunk.
        """
        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"sample_rate must be one of {sorted(SUPPORTED_SAMPLE_RATES)}, got {sample_rate}"
            )
        if codec not in SUPPORTED_CODECS:
            raise ValueError(
                f"codec must be one of {sorted(SUPPORTED_CODECS)} for raw PCM-compatible "
                f"output (got {codec}). mp3/wav/alaw are not supported because they "
                f"produce framed audio that the audio_track cannot consume directly."
            )
        if optimize_streaming_latency not in (0, 1):
            raise ValueError("optimize_streaming_latency must be 0 or 1")

        super().__init__(
            sample_rate=sample_rate,
            num_channels=XAI_TTS_NUM_CHANNELS,
            word_timestamps=False,
        )

        self._api_key = api_key or os.getenv("XAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "xAI API key must be provided either through the api_key parameter "
                "or the XAI_API_KEY environment variable"
            )

        voice_lower = voice.lower()
        if voice_lower not in SUPPORTED_VOICES:
            raise ValueError(
                f"voice must be one of {sorted(SUPPORTED_VOICES)}, got {voice}"
            )

        self._voice = voice_lower
        self.language = language
        self.codec = codec
        self.optimize_streaming_latency = optimize_streaming_latency
        self.text_normalization = text_normalization
        self.base_url = base_url

        self._ws_session: Optional[aiohttp.ClientSession] = None
        self._ws_connection: Optional[aiohttp.ClientWebSocketResponse] = None
        self._connection_lock = asyncio.Lock()
        self._synthesis_lock = asyncio.Lock()
        self._receive_task: Optional[asyncio.Task] = None
        self._current_done_future: Optional[asyncio.Future[None]] = None
        self._first_chunk_sent = False
        self._interrupted = False
        self._closed = False

    def reset_first_audio_tracking(self) -> None:
        """Reset the first-audio-byte tracking state for the next synthesis turn."""
        self._first_chunk_sent = False

    async def synthesize(
        self,
        text: AsyncIterator[str] | str,
        voice_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Synthesize text to speech via xAI's bidirectional WebSocket API."""
        try:
            if not self.audio_track or not self.loop:
                self.emit("error", "Audio track or event loop not set")
                return

            if voice_id:
                voice_lower = voice_id.lower()
                if voice_lower not in SUPPORTED_VOICES:
                    self.emit(
                        "error",
                        f"voice_id must be one of {sorted(SUPPORTED_VOICES)}, got {voice_id}",
                    )
                    return
                self._voice = voice_lower

            async with self._synthesis_lock:
                self._interrupted = False
                self._first_chunk_sent = False

                await self._ensure_ws_connection()
                if not self._ws_connection:
                    raise RuntimeError("WebSocket connection is not available.")

                done_future: asyncio.Future[None] = (
                    asyncio.get_event_loop().create_future()
                )
                self._current_done_future = done_future

                async def _string_iterator(s: str) -> AsyncIterator[str]:
                    yield s

                text_iterator = (
                    _string_iterator(text) if isinstance(text, str) else text
                )

                send_task = asyncio.create_task(
                    self._send_task(text_iterator, done_future)
                )

                try:
                    await done_future
                finally:
                    if not send_task.done():
                        try:
                            await send_task
                        except Exception:
                            pass

        except Exception as e:
            self.emit("error", f"TTS synthesis failed: {e}")
            raise
        finally:
            self._current_done_future = None

    async def _send_task(
        self,
        text_iterator: AsyncIterator[str],
        done_future: asyncio.Future[None],
    ) -> None:
        """Send text.delta messages, then text.done at end of utterance."""
        has_sent = False
        try:
            async for chunk in text_iterator:
                if self._interrupted:
                    break
                if not chunk or not chunk.strip():
                    continue
                if not self._ws_connection or self._ws_connection.closed:
                    break
                payload = {"type": "text.delta", "delta": chunk}
                await self._ws_connection.send_str(json.dumps(payload))
                has_sent = True
        except Exception as e:
            if not done_future.done():
                done_future.set_exception(e)
            return
        finally:
            if (
                has_sent
                and not self._interrupted
                and self._ws_connection
                and not self._ws_connection.closed
            ):
                try:
                    await self._ws_connection.send_str(
                        json.dumps({"type": "text.done"})
                    )
                except Exception as e:
                    if not done_future.done():
                        done_future.set_exception(e)

        if not has_sent and not done_future.done():
            done_future.set_result(None)

    async def _receive_loop(self) -> None:
        """Long-running task: read audio.delta / audio.done / error frames."""
        try:
            while self._ws_connection and not self._ws_connection.closed:
                msg = await self._ws_connection.receive()
                if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
                    break
                if msg.type == aiohttp.WSMsgType.ERROR:
                    err = self._ws_connection.exception()
                    self._fail_pending(RuntimeError(f"xAI TTS WebSocket error: {err}"))
                    break
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                try:
                    data = json.loads(msg.data)
                except Exception as e:
                    logger.error(f"Failed to parse xAI TTS message: {e}")
                    continue

                event_type = data.get("type")
                if event_type == "audio.delta":
                    delta = data.get("delta")
                    if delta:
                        try:
                            await self._stream_audio(base64.b64decode(delta))
                        except Exception as e:
                            logger.error(f"Failed to decode/stream audio: {e}")
                elif event_type == "audio.done":
                    future = self._current_done_future
                    if future and not future.done():
                        future.set_result(None)
                elif event_type == "error":
                    message = data.get("message", "unknown error")
                    self._fail_pending(RuntimeError(f"xAI TTS error: {message}"))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._fail_pending(e)

    def _fail_pending(self, exc: BaseException) -> None:
        future = self._current_done_future
        if future and not future.done():
            future.set_exception(exc)

    async def _stream_audio(self, audio_chunk: bytes) -> None:
        """Push a chunk of raw audio bytes into the framework's audio_track."""
        if self._interrupted or not audio_chunk:
            return

        if not self._first_chunk_sent:
            self._first_chunk_sent = True
            if self._first_audio_callback:
                await self._first_audio_callback()

        if self.audio_track:
            await self.audio_track.add_new_bytes(audio_chunk)

    async def _ensure_ws_connection(self) -> None:
        """Open or re-open the WebSocket connection if needed."""
        async with self._connection_lock:
            if self._ws_connection and not self._ws_connection.closed:
                return

            if self._receive_task and not self._receive_task.done():
                self._receive_task.cancel()
                try:
                    await self._receive_task
                except (asyncio.CancelledError, Exception):
                    pass
            self._receive_task = None

            if self._ws_connection:
                try:
                    await self._ws_connection.close()
                except Exception:
                    pass
                self._ws_connection = None

            if self._ws_session:
                try:
                    await self._ws_session.close()
                except Exception:
                    pass
                self._ws_session = None

            try:
                self._ws_session = aiohttp.ClientSession()

                params = [
                    ("voice", self._voice),
                    ("language", self.language),
                    ("codec", self.codec),
                    ("sample_rate", str(self.sample_rate)),
                    ("optimize_streaming_latency", str(self.optimize_streaming_latency)),
                    ("text_normalization", str(self.text_normalization).lower()),
                ]
                ws_url = f"{self.base_url}?{urlencode(params)}"
                headers = {"Authorization": f"Bearer {self._api_key}"}

                self._ws_connection = await asyncio.wait_for(
                    self._ws_session.ws_connect(
                        ws_url, headers=headers, heartbeat=30.0
                    ),
                    timeout=5.0,
                )
                self._receive_task = asyncio.create_task(self._receive_loop())
            except aiohttp.WSServerHandshakeError as e:
                self.emit(
                    "error",
                    f"xAI TTS WebSocket handshake failed (status {e.status}): {e.message}",
                )
                raise
            except Exception as e:
                self.emit("error", f"Failed to establish xAI TTS WebSocket: {e}")
                raise

    async def interrupt(self) -> None:
        """Interrupt any in-flight synthesis and clear the audio_track buffer."""
        self._interrupted = True

        if self.audio_track:
            self.audio_track.interrupt()

        future = self._current_done_future
        if future and not future.done():
            future.set_result(None)

        if self._ws_connection and not self._ws_connection.closed:
            try:
                await self._ws_connection.close()
            except Exception:
                pass

    async def aclose(self) -> None:
        """Gracefully clean up all resources."""
        await super().aclose()
        self._interrupted = True
        self._closed = True

        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except (asyncio.CancelledError, Exception):
                pass
            self._receive_task = None

        if self._ws_connection and not self._ws_connection.closed:
            await self._ws_connection.close()
        self._ws_connection = None

        if self._ws_session and not self._ws_session.closed:
            await self._ws_session.close()
        self._ws_session = None
