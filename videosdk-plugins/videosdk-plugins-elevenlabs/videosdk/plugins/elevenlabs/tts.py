from __future__ import annotations

from typing import Any, AsyncIterator, Literal, Optional, Union
import os
import httpx
import asyncio
import json
import aiohttp
import weakref
from dataclasses import dataclass
from videosdk.agents import TTS, segment_text

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
        model: str = DEFAULT_MODEL,
        voice: str = DEFAULT_VOICE_ID,
        speed: float = 1.0,
        api_key: str | None = None,
        response_format: str = "pcm_24000",
        voice_settings: VoiceSettings | None = None,
        base_url: str = API_BASE_URL,
        enable_streaming: bool = True,
        inactivity_timeout: int = WS_INACTIVITY_TIMEOUT,
    ) -> None:
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
        """WebSocket-based streaming synthesis"""

        ws_session = None
        ws_connection = None

        try:
            ws_url = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"
            params = {
                "model_id": self.model,
                "output_format": self.response_format,
                "inactivity_timeout": self.inactivity_timeout,
            }
            param_string = "&".join([f"{k}={v}" for k, v in params.items()])
            full_ws_url = f"{ws_url}?{param_string}"

            headers = {"xi-api-key": self.api_key}

            ws_session = aiohttp.ClientSession()
            ws_connection = await asyncio.wait_for(
                ws_session.ws_connect(full_ws_url, headers=headers),
                timeout=10.0
            )

            init_message = {
                "text": " ",
                "voice_settings": {
                    "stability": self.voice_settings.stability,
                    "similarity_boost": self.voice_settings.similarity_boost,
                    "style": self.voice_settings.style,
                    "use_speaker_boost": self.voice_settings.use_speaker_boost,
                },
            }
            await ws_connection.send_str(json.dumps(init_message))

            self._send_task = asyncio.create_task(
                self._send_text_task(ws_connection, text))
            self._recv_task = asyncio.create_task(
                self._receive_audio_task(ws_connection))

            await asyncio.gather(self._send_task, self._recv_task)

        except Exception as e:
            self.emit("error", f"Streaming synthesis failed: {str(e)}")

            if isinstance(text, str):
                await self._chunked_synthesis(text, voice_id)
            else:
                async for segment in segment_text(text):
                    if self._should_stop:
                        break
                    await self._chunked_synthesis(segment, voice_id)

        finally:
            for task in [self._send_task, self._recv_task]:
                if task and not task.done():
                    task.cancel()

            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        *(t for t in [self._send_task, self._recv_task] if t),
                        return_exceptions=True
                    ),
                    timeout=0.3
                )
            except asyncio.TimeoutError:
                pass

            self._send_task = None
            self._recv_task = None

            if ws_connection and not ws_connection.closed:
                await ws_connection.close()
            if ws_session and not ws_session.closed:
                await ws_session.close()

    async def _send_text_task(self, ws_connection: aiohttp.ClientWebSocketResponse, text: Union[AsyncIterator[str], str]) -> None:
        """Task for sending text to WebSocket"""
        try:
            if isinstance(text, str):
                if not self._should_stop:
                    text_message = {"text": f"{text} "}
                    await ws_connection.send_str(json.dumps(text_message))
            else:
                async for chunk in text:
                    if ws_connection.closed or self._should_stop:
                        break

                    chunk_message = {"text": f"{chunk} "}
                    await ws_connection.send_str(json.dumps(chunk_message))

            if not ws_connection.closed and not self._should_stop:
                eos_message = {"text": ""}
                await ws_connection.send_str(json.dumps(eos_message))

        except Exception as e:
            if not self._should_stop:
                self.emit("error", f"Send task error: {str(e)}")
            raise

    async def _receive_audio_task(self, ws_connection: aiohttp.ClientWebSocketResponse) -> None:
        """Task for receiving audio from WebSocket"""
        try:
            while not ws_connection.closed and not self._should_stop:
                try:
                    msg = await ws_connection.receive()

                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)

                        if data.get("audio"):
                            import base64
                            audio_chunk = base64.b64decode(data["audio"])
                            if not self._should_stop:
                                await self._stream_audio_chunks(audio_chunk)

                        elif data.get("isFinal"):
                            break

                        elif data.get("error"):
                            self.emit(
                                "error", f"ElevenLabs error: {data['error']}")
                            raise ValueError(
                                f"ElevenLabs error: {data['error']}")

                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        raise ConnectionError(
                            f"WebSocket error: {ws_connection.exception()}")

                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING):
                        break

                except asyncio.TimeoutError:
                    if not self._should_stop:
                        self.emit("error", "WebSocket receive timeout")
                    break

        except Exception as e:
            if not self._should_stop:
                self.emit("error", f"Receive task error: {str(e)}")
            raise

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

        for task in [self._send_task, self._recv_task]:
            if task and not task.done():
                task.cancel()

        if self._ws_connection and not self._ws_connection.closed:
            await self._ws_connection.close()

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
            await self._ws_connection.close()
        if self._ws_session:
            await self._ws_session.close()
        if self._session:
            await self._session.aclose()
        await super().aclose()
