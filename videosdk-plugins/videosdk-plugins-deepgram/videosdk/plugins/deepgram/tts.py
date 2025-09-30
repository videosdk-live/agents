from __future__ import annotations

import asyncio
import aiohttp
import json
from typing import Any, AsyncIterator, Union, Optional
import os
from videosdk.agents import TTS
DEEPGRAM_SAMPLE_RATE = 24000
DEEPGRAM_CHANNELS = 1
DEFAULT_MODEL = "aura-2-thalia-en"
DEFAULT_ENCODING = "linear16"
API_BASE_URL = "wss://api.deepgram.com/v1/speak"


class DeepgramTTS(TTS):
    def __init__(
            self,
            *,
            api_key: str | None = None,
            model: str = DEFAULT_MODEL,
            encoding: str = DEFAULT_ENCODING,
            sample_rate: int = DEEPGRAM_SAMPLE_RATE,
            base_url: str = API_BASE_URL,
            **kwargs: Any,
    ) -> None:
        super().__init__(sample_rate=sample_rate, num_channels=DEEPGRAM_CHANNELS)

        self.model = model
        self.encoding = encoding
        self.base_url = base_url
        self.audio_track = None
        self.loop = None
        self._ws_session: aiohttp.ClientSession | None = None
        self._ws_connection: aiohttp.ClientWebSocketResponse | None = None
        self._send_task: asyncio.Task | None = None
        self._recv_task: asyncio.Task | None = None
        self._should_stop = False
        self._first_chunk_sent = False
        self._connection_lock = asyncio.Lock()

        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Deepgram API key must be provided either through the 'api_key' parameter or the DEEPGRAM_API_KEY environment variable."
            )

    def reset_first_audio_tracking(self) -> None:
        self._first_chunk_sent = False

    async def _ensure_connection(self) -> None:
        async with self._connection_lock:
            if self._ws_connection and not self._ws_connection.closed:
                return

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
                self._ws_session.ws_connect(full_ws_url, headers=headers),
                timeout=50.0
            )
            if self._recv_task and not self._recv_task.done():
                self._recv_task.cancel()
            self._recv_task = asyncio.create_task(self._receive_audio_task())

    async def synthesize(
            self,
            text: AsyncIterator[str] | str,
            **kwargs: Any,
    ) -> None:
        try:
            if not self.audio_track or not self.loop:
                self.emit("error", "Audio track or event loop not set")
                return

            await self.interrupt()
            self._should_stop = False
            await self._stream_synthesis(text)

        except Exception as e:
            self.emit("error", f"TTS synthesis failed: {str(e)}")

    async def _stream_synthesis(self, text: Union[AsyncIterator[str], str]) -> None:
        try:
            await self._ensure_connection()
            self._send_task = asyncio.create_task(self._send_text_task(text))
            await self._send_task
        except Exception as e:
            self.emit("error", f"Streaming synthesis failed: {str(e)}")
            await self.aclose()
        finally:
            if self._send_task and not self._send_task.done():
                self._send_task.cancel()
            self._send_task = None

    async def _send_text_task(self, text: Union[AsyncIterator[str], str]) -> None:
        if not self._ws_connection or self._ws_connection.closed:
            return

        try:
            if isinstance(text, str):
                if not self._should_stop:
                    payload = {"type": "Speak", "text": text}
                    await self._ws_connection.send_json(payload)
            else:
                async for chunk in text:
                    if self._ws_connection.closed or self._should_stop:
                        break
                    payload = {"type": "Speak", "text": chunk}
                    await self._ws_connection.send_json(payload)

            if not self._ws_connection.closed and not self._should_stop:
                await self._ws_connection.send_json({"type": "Flush"})
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if not self._should_stop:
                self.emit("error", f"Send task error: {str(e)}")

    async def _receive_audio_task(self) -> None:
        if not self._ws_connection:
            return

        try:
            while not self._ws_connection.closed:
                msg = await self._ws_connection.receive()

                if msg.type == aiohttp.WSMsgType.BINARY:
                    if not self._should_stop:
                        await self._stream_audio_chunks(msg.data)
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if data.get('type') == 'Error' and not self._should_stop:
                        self.emit("error", f"Deepgram error: {data.get('description', 'Unknown error')}")
                        break
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING):
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    raise ConnectionError(f"WebSocket error: {self._ws_connection.exception()}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if not self._should_stop:
                self.emit("error", f"Receive task error: {str(e)}")

    async def _stream_audio_chunks(self, audio_bytes: bytes) -> None:
        if not audio_bytes or self._should_stop:
            return

        if self.audio_track and self.loop:
            await self.audio_track.add_new_bytes(audio_bytes)

    async def interrupt(self) -> None:
        self._should_stop = True

        if self.audio_track:
            self.audio_track.interrupt()

        if self._send_task and not self._send_task.done():
            self._send_task.cancel()


    async def aclose(self) -> None:
        self._should_stop = True

        for task in [self._send_task, self._recv_task]:
            if task and not task.done():
                task.cancel()

        if self._ws_connection and not self._ws_connection.closed:
            await self._ws_connection.close()
        if self._ws_session and not self._ws_session.closed:
            await self._ws_session.close()

        self._ws_connection = None
        self._ws_session = None

        await super().aclose()