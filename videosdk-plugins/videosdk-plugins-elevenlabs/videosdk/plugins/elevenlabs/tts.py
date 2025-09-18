from __future__ import annotations

from typing import Any, AsyncIterator, Optional, Union
import os
import httpx
import asyncio
import json
import aiohttp
import weakref
from dataclasses import dataclass
from videosdk.agents import TTS, segment_text
import base64
import uuid

ELEVENLABS_SAMPLE_RATE = 24000
ELEVENLABS_CHANNELS = 1

DEFAULT_MODEL = "eleven_flash_v2_5"
DEFAULT_VOICE_ID = "EXAVITQu4vr4xnSDxMaL"
API_BASE_URL = "https://api.elevenlabs.io/v1"
WS_INACTIVITY_TIMEOUT = 300


@dataclass
class VoiceSettings:
    stability: float = 0.71
    similarity_boost: float = 0.5
    style: float = 0.0
    use_speaker_boost: bool = True


class ElevenLabsTTS(TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        voice: str = DEFAULT_VOICE_ID,
        speed: float = 1.0,
        response_format: str = "pcm_24000",
        voice_settings: VoiceSettings | None = None,
        base_url: str = API_BASE_URL,
        enable_streaming: bool = True,
        inactivity_timeout: int = WS_INACTIVITY_TIMEOUT,
    ) -> None:
        """Initialize the ElevenLabs TTS plugin.

        Args:
            api_key (Optional[str], optional): ElevenLabs API key. Uses ELEVENLABS_API_KEY environment variable if not provided. Defaults to None.
            model (str): The model to use for the TTS plugin. Defaults to "eleven_flash_v2_5".
            voice (str): The voice to use for the TTS plugin. Defaults to "EXAVITQu4vr4xnSDxMaL".
            speed (float): The speed to use for the TTS plugin. Defaults to 1.0.
            response_format (str): The response format to use for the TTS plugin. Defaults to "pcm_24000".
            voice_settings (Optional[VoiceSettings], optional): The voice settings to use for the TTS plugin. Defaults to None.
            base_url (str): The base URL to use for the TTS plugin. Defaults to "https://api.elevenlabs.io/v1".
            enable_streaming (bool): Whether to enable streaming for the TTS plugin. Defaults to True.
            inactivity_timeout (int): The inactivity timeout to use for the TTS plugin. Defaults to 300.
        """
        super().__init__(
            sample_rate=ELEVENLABS_SAMPLE_RATE, num_channels=ELEVENLABS_CHANNELS
        )

        self.model = model
        self.voice = voice
        self.speed = speed
        self.audio_track = None
        self.loop = None
        self.response_format = response_format
        self.base_url = base_url
        self.enable_streaming = enable_streaming
        self.voice_settings = voice_settings or VoiceSettings()
        self.inactivity_timeout = inactivity_timeout
        self._first_chunk_sent = False
        self._ws_session = None
        self._ws_connection = None
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ElevenLabs API key must be provided either through api_key parameter or ELEVENLABS_API_KEY environment variable")

        self._session = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=15.0, read=30.0,
                                  write=5.0, pool=5.0),
            follow_redirects=True,
        )

        self._streams = weakref.WeakSet()
        self._send_task: asyncio.Task | None = None
        self._recv_task: asyncio.Task | None = None
        self._should_stop = False

        self._connection_lock = asyncio.Lock()
        self._ws_voice_id: str | None = None
        self._active_contexts: set[str] = set()
        self._context_futures: dict[str, asyncio.Future[None]] = {}

    def reset_first_audio_tracking(self) -> None:
        """Reset the first audio tracking state for next TTS task"""
        self._first_chunk_sent = False

    async def synthesize(
        self,
        text: AsyncIterator[str] | str,
        voice_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        try:
            if not self.audio_track or not self.loop:
                self.emit("error", "Audio track or event loop not set")
                return

            target_voice = voice_id or self.voice
            self._should_stop = False

            if self.enable_streaming:
                await self._stream_synthesis(text, target_voice)
            else:
                if isinstance(text, AsyncIterator):
                    async for segment in segment_text(text):
                        if self._should_stop:
                            break
                        await self._chunked_synthesis(segment, target_voice)
                else:
                    await self._chunked_synthesis(text, target_voice)

        except Exception as e:
            self.emit("error", f"TTS synthesis failed: {str(e)}")

    async def _chunked_synthesis(self, text: str, voice_id: str) -> None:
        """Non-streaming synthesis using the standard API"""
        url = f"{self.base_url}/text-to-speech/{voice_id}/stream"

        params = {
            "model_id": self.model,
            "output_format": self.response_format,
        }

        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "text": text,
            "voice_settings": {
                "stability": self.voice_settings.stability,
                "similarity_boost": self.voice_settings.similarity_boost,
                "style": self.voice_settings.style,
                "use_speaker_boost": self.voice_settings.use_speaker_boost,
            },
        }

        try:
            async with self._session.stream(
                "POST",
                url,
                headers=headers,
                json=payload,
                params=params
            ) as response:
                response.raise_for_status()

                async for chunk in response.aiter_bytes():
                    if self._should_stop:
                        break
                    if chunk:
                        await self._stream_audio_chunks(chunk)

        except httpx.HTTPStatusError as e:
            self.emit(
                "error", f"HTTP error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            self.emit("error", f"Chunked synthesis failed: {str(e)}")

    async def _stream_synthesis(self, text: Union[AsyncIterator[str], str], voice_id: str) -> None:
        """WebSocket-based streaming synthesis using multi-context connection"""
        try:
            await self._ensure_connection(voice_id)

            context_id = uuid.uuid4().hex[:12]
            done_future: asyncio.Future[None] = asyncio.get_event_loop().create_future()
            self.register_context(context_id, done_future)

            async def _single_chunk_gen(s: str) -> AsyncIterator[str]:
                yield s

            async def _send_chunks() -> None:
                try:
                    first_message_sent = False
                    if isinstance(text, str):
                        async for segment in segment_text(_single_chunk_gen(text)):
                            if self._should_stop:
                                break
                            await self.send_text(context_id, f"{segment} ",
                                                 voice_settings=None if first_message_sent else self._voice_settings_dict(),
                                                 flush=True)
                            first_message_sent = True
                    else:
                        async for chunk in text:
                            if self._should_stop:
                                break
                            await self.send_text(context_id, f"{chunk} ",
                                                 voice_settings=None if first_message_sent else self._voice_settings_dict())
                            first_message_sent = True

                    if not self._should_stop:
                        await self.flush_context(context_id)
                        await self.close_context(context_id)
                except Exception as e:
                    if not done_future.done():
                        done_future.set_exception(e)

            sender = asyncio.create_task(_send_chunks())

            await done_future
            await sender

        except Exception as e:
            self.emit("error", f"Streaming synthesis failed: {str(e)}")

            if isinstance(text, str):
                await self._chunked_synthesis(text, voice_id)
            else:
                async for segment in segment_text(text):
                    if self._should_stop:
                        break
                    await self._chunked_synthesis(segment, voice_id)

    def _voice_settings_dict(self) -> dict[str, Any]:
        return {
            "stability": self.voice_settings.stability,
            "similarity_boost": self.voice_settings.similarity_boost,
            "style": self.voice_settings.style,
            "use_speaker_boost": self.voice_settings.use_speaker_boost,
        }

    async def _stream_audio_chunks(self, audio_bytes: bytes) -> None:
        if not audio_bytes or self._should_stop:
            return

        if not self._first_chunk_sent and hasattr(self, '_first_audio_callback') and self._first_audio_callback:
            self._first_chunk_sent = True
            asyncio.create_task(self._first_audio_callback())

        if self.audio_track and self.loop:
            await self.audio_track.add_new_bytes(audio_bytes)

    async def interrupt(self) -> None:
        """Simple but effective interruption"""
        self._should_stop = True

        if self.audio_track:
            self.audio_track.interrupt()

        await self.close_all_contexts()

    async def aclose(self) -> None:
        """Cleanup resources"""
        self._should_stop = True

        for task in [self._send_task, self._recv_task]:
            if task and not task.done():
                task.cancel()

        for stream in list(self._streams):
            try:
                await stream.aclose()
            except Exception:
                pass

        self._streams.clear()

        if self._ws_connection and not self._ws_connection.closed:
            try:
                await self._ws_connection.send_str(json.dumps({"close_socket": True}))
            except Exception:
                pass
            await self._ws_connection.close()
        if self._ws_session and not self._ws_session.closed:
            await self._ws_session.close()
        self._ws_connection = None
        self._ws_session = None
        if self._session:
            await self._session.aclose()
        await super().aclose()

    async def _ensure_connection(self, voice_id: str) -> None:
        async with self._connection_lock:
            if self._ws_connection and not self._ws_connection.closed and self._ws_voice_id == voice_id:
                return

            if self._ws_connection and not self._ws_connection.closed:
                try:
                    await self._ws_connection.send_str(json.dumps({"close_socket": True}))
                except Exception:
                    pass
                await self._ws_connection.close()
            if self._ws_session and not self._ws_session.closed:
                await self._ws_session.close()

            self._ws_session = aiohttp.ClientSession()
            self._ws_voice_id = voice_id

            ws_url = f"{self.base_url}/text-to-speech/{voice_id}/multi-stream-input".replace("https://", "wss://").replace("http://", "ws://")
            params = {
                "model_id": self.model,
                "output_format": self.response_format,
                "inactivity_timeout": self.inactivity_timeout,
            }
            param_string = "&".join([f"{k}={v}" for k, v in params.items()])
            full_ws_url = f"{ws_url}?{param_string}"
            headers = {"xi-api-key": self.api_key}
            self._ws_connection = await asyncio.wait_for(self._ws_session.ws_connect(full_ws_url, headers=headers), timeout=10.0)

            if self._recv_task and not self._recv_task.done():
                self._recv_task.cancel()
            self._recv_task = asyncio.create_task(self._recv_loop())

    def register_context(self, context_id: str, done_future: asyncio.Future[None]) -> None:
        self._context_futures[context_id] = done_future

    async def send_text(
        self,
        context_id: str,
        text: str,
        *,
        voice_settings: Optional[dict[str, Any]] = None,
        flush: bool = False,
    ) -> None:
        if not self._ws_connection or self._ws_connection.closed:
            raise RuntimeError("WebSocket connection is closed")

        if context_id not in self._active_contexts:
            init_msg = {
                "context_id": context_id,
                "text": " ",
            }
            if voice_settings:
                init_msg["voice_settings"] = voice_settings
            await self._ws_connection.send_str(json.dumps(init_msg))
            self._active_contexts.add(context_id)

        pkt: dict[str, Any] = {"context_id": context_id, "text": text}
        if flush:
            pkt["flush"] = True
        await self._ws_connection.send_str(json.dumps(pkt))

    async def flush_context(self, context_id: str) -> None:
        if not self._ws_connection or self._ws_connection.closed:
            return
        await self._ws_connection.send_str(json.dumps({"context_id": context_id, "flush": True}))

    async def close_context(self, context_id: str) -> None:
        if not self._ws_connection or self._ws_connection.closed:
            return
        await self._ws_connection.send_str(json.dumps({"context_id": context_id, "close_context": True}))

    async def close_all_contexts(self) -> None:
        try:
            for context_id in list(self._active_contexts):
                await self.close_context(context_id)
        except Exception:
            pass

    async def _recv_loop(self) -> None:
        try:
            while self._ws_connection and not self._ws_connection.closed:
                msg = await self._ws_connection.receive()
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)

                    if data.get("error"):
                        ctx_id = data.get("contextId")
                        fut = self._context_futures.get(ctx_id)
                        if fut and not fut.done():
                            fut.set_exception(RuntimeError(data["error"]))
                        continue

                    if data.get("audio"):
                        audio_chunk = base64.b64decode(data["audio"]) if isinstance(data["audio"], str) else None
                        if audio_chunk:
                            if not self._first_chunk_sent and hasattr(self, '_first_audio_callback') and self._first_audio_callback:
                                self._first_chunk_sent = True
                                asyncio.create_task(self._first_audio_callback())
                            if self.audio_track:
                                await self.audio_track.add_new_bytes(audio_chunk)

                    if data.get("is_final") or data.get("isFinal"):
                        ctx_id = data.get("contextId")
                        if ctx_id:
                            fut = self._context_futures.pop(ctx_id, None)
                            self._active_contexts.discard(ctx_id)
                            if fut and not fut.done():
                                fut.set_result(None)

                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING):
                    break
        except Exception:
            for fut in self._context_futures.values():
                if not fut.done():
                    fut.set_exception(RuntimeError("WebSocket receive loop error"))
            self._context_futures.clear()