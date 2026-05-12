from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, AsyncIterator, Literal, Optional, Union

import aiohttp

from videosdk.agents import TTS, FlushMarker

logger = logging.getLogger(__name__)

LMNT_WSS_URL = "wss://api.lmnt.com/v1/ai/speech/stream"
LMNT_SAMPLE_RATE = 24000
LMNT_CHANNELS = 1
LMNT_VERSION = "1.0"

DEFAULT_MODEL = "blizzard"
DEFAULT_VOICE = "ava"
DEFAULT_LANGUAGE = "auto"
DEFAULT_FORMAT = "pcm_s16le"

_LanguageCode = Union[
    Literal["auto", "ar", "as", "bn", "cs", "da", "de", "en", "es", "fi", "fr",
            "hi", "id", "it", "ja", "ko", "ml", "mr", "nl", "pl", "pt", "ru",
            "sk", "sv", "ta", "te", "th", "tr", "uk", "ur", "vi", "zh"],
    str
]
_FormatType = Union[Literal["mp3", "pcm_s16le", "pcm_f32le", "ulaw", "webm"], str]
_SampleRate = Union[Literal[8000, 16000, 24000], int]


class LMNTTTS(TTS):
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        voice: str = DEFAULT_VOICE,
        model: str = DEFAULT_MODEL,
        language: _LanguageCode = DEFAULT_LANGUAGE,
        format: _FormatType = DEFAULT_FORMAT,
        sample_rate: _SampleRate = LMNT_SAMPLE_RATE,
        seed: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 0.8,
        ws_url: str = LMNT_WSS_URL,
    ) -> None:
        """Initialize the LMNT TTS plugin (WebSocket streaming).

        Args:
            api_key: LMNT API key. Falls back to ``LMNT_API_KEY`` env var.
            voice: Voice id. Defaults to ``ava``.
            model: Model id. Defaults to ``blizzard``.
            language: ISO 639-1 language code or ``auto``. Defaults to ``auto``.
            format: Audio output format. Defaults to ``pcm_s16le`` (raw 16-bit
                little-endian PCM) — feeds the audio track directly with no
                container/decoding step.
            sample_rate: Output sample rate. One of 8000, 16000, 24000.
            seed: Optional generation seed for reproducibility.
            temperature: Sampling temperature, 0.0-1.0.
            top_p: Nucleus sampling parameter, 0.0-1.0.
            ws_url: Override for the WSS endpoint.
        """
        super().__init__(sample_rate=sample_rate, num_channels=LMNT_CHANNELS)

        self.voice = voice
        self.model = model
        self.language = language
        self.format = format
        self.output_sample_rate = sample_rate
        self.seed = seed
        self.temperature = temperature
        self.top_p = top_p
        self.ws_url = ws_url
        self.audio_track = None
        self.loop = None
        self._first_chunk_sent = False
        self._interrupted = False
        self._session: Optional[aiohttp.ClientSession] = None
        self._active_ws: Optional[aiohttp.ClientWebSocketResponse] = None

        self.api_key = api_key or os.getenv("LMNT_API_KEY")
        if not self.api_key:
            raise ValueError(
                "LMNT API key must be provided either through api_key parameter "
                "or LMNT_API_KEY environment variable"
            )

    def reset_first_audio_tracking(self) -> None:
        self._first_chunk_sent = False

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def synthesize(
        self,
        text: AsyncIterator[Union[str, FlushMarker]] | str,
        voice_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Synthesize via LMNT's WebSocket streaming API.

        Each ``FlushMarker`` in the input stream is forwarded as a
        ``{"flush": true}`` command, prompting LMNT to emit audio for the
        current text buffer immediately. End-of-stream is signalled with
        ``{"eof": true}``; the server then drains its buffer and closes the
        connection.
        """
        if not self.audio_track or not self.loop:
            self.emit("error", "Audio track or event loop not set")
            return

        self._interrupted = False

        try:
            ws = await asyncio.wait_for(
                self._ensure_session().ws_connect(self.ws_url),
                timeout=10.0,
            )
        except Exception as e:
            self.emit("error", f"LMNT WSS connect failed: {e}")
            return

        self._active_ws = ws
        try:
            init = {
                "X-API-Key": self.api_key,
                "lmnt-version": LMNT_VERSION,
                "voice": voice_id or self.voice,
                "model": kwargs.get("model", self.model),
                "format": kwargs.get("format", self.format),
                "language": kwargs.get("language", self.language),
                "sample_rate": kwargs.get("sample_rate", self.output_sample_rate),
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
            }
            seed = kwargs.get("seed", self.seed)
            if seed is not None:
                init["seed"] = seed

            await ws.send_json(init)

            send_task = asyncio.create_task(self._send_text(ws, text))
            try:
                await self._receive_audio(ws)
            finally:
                if not send_task.done():
                    send_task.cancel()
                try:
                    await send_task
                except (asyncio.CancelledError, Exception):
                    pass
        except Exception as e:
            if not self._interrupted:
                self.emit("error", f"LMNT WSS synthesis failed: {e}")
        finally:
            self._active_ws = None
            try:
                if not ws.closed:
                    await ws.close()
            except Exception:
                pass

    async def _send_text(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        text: Union[AsyncIterator[Union[str, FlushMarker]], str],
    ) -> None:
        try:
            if isinstance(text, str):
                if text and not self._interrupted and not ws.closed:
                    await ws.send_json({"text": text})
            else:
                async for chunk in text:
                    if self._interrupted or ws.closed:
                        break
                    if isinstance(chunk, FlushMarker):
                        await ws.send_json({"flush": True})
                        continue
                    if not chunk:
                        continue
                    await ws.send_json({"text": chunk})

            if not self._interrupted and not ws.closed:
                await ws.send_json({"eof": True})
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if not self._interrupted:
                self.emit("error", f"LMNT send error: {e}")

    async def _receive_audio(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        try:
            async for msg in ws:
                if self._interrupted:
                    break
                if msg.type == aiohttp.WSMsgType.BINARY:
                    if not self._first_chunk_sent and self._first_audio_callback:
                        self._first_chunk_sent = True
                        await self._first_audio_callback()
                    if self.audio_track:
                        await self.audio_track.add_new_bytes(msg.data)
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(data, dict) and data.get("error"):
                        if not self._interrupted:
                            self.emit("error", f"LMNT server error: {data['error']}")
                        break
                elif msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    if not self._interrupted:
                        self.emit("error", f"LMNT WSS error: {ws.exception()}")
                    break
        except asyncio.CancelledError:
            raise

    async def aclose(self) -> None:
        self._interrupted = True
        if self._active_ws and not self._active_ws.closed:
            try:
                await self._active_ws.close()
            except Exception:
                pass
        if self._session and not self._session.closed:
            try:
                await self._session.close()
            except Exception:
                pass
        await super().aclose()

    async def interrupt(self) -> None:
        """Stop synthesis. Closes the active WSS so the server stops emitting
        audio for the current session; the next ``synthesize()`` opens a
        fresh connection."""
        self._interrupted = True
        if self.audio_track:
            self.audio_track.interrupt()
        if self._active_ws and not self._active_ws.closed:
            try:
                await self._active_ws.close()
            except Exception:
                pass
