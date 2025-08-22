from __future__ import annotations
import asyncio
import base64
import json
import os
from typing import Any, AsyncIterator, List, Optional, Union

import aiohttp

from videosdk.agents import TTS

CARTESIA_SAMPLE_RATE = 24000
CARTESIA_CHANNELS = 1
DEFAULT_MODEL = "sonic-2"
DEFAULT_VOICE_ID = "794f9389-aac1-45b6-b726-9d9369183238"
PLAYBACK_CHUNK_SIZE = 960
API_VERSION = "2024-06-10"

# Streaming text pacing thresholds
MIN_CHARS_FLUSH = 100
MIN_WORDS_FLUSH = 12
INACTIVITY_TIMEOUT_SEC = 0.18


class CartesiaTTS(TTS):
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        voice_id: Union[str, List[float]] = DEFAULT_VOICE_ID,
        api_key: str | None = None,
        language: str = "en",
        base_url: str = "https://api.cartesia.ai",
    ) -> None:
        super().__init__(sample_rate=CARTESIA_SAMPLE_RATE, num_channels=CARTESIA_CHANNELS)

        self.model = model
        self.language = language
        self.base_url = base_url
        self._voice = voice_id
        self._first_chunk_sent = False
        self._audio_buffer = bytearray()
        self._interrupted = False
        self._current_tasks: list[asyncio.Task] = []

        api_key = api_key or os.getenv("CARTESIA_API_KEY")
        if not api_key:
            raise ValueError("Cartesia API key must be provided")
        self._api_key = api_key
        
        self._ws_session: aiohttp.ClientSession | None = None
        self._ws_connection: aiohttp.ClientWebSocketResponse | None = None
        self._connection_lock = asyncio.Lock()

    def reset_first_audio_tracking(self) -> None:
        self._first_chunk_sent = False
        self._audio_buffer.clear()
        self._interrupted = False

    async def _ensure_ws_connection(self) -> aiohttp.ClientWebSocketResponse:
        async with self._connection_lock:
            if self._ws_connection and not self._ws_connection.closed:
                return self._ws_connection

            if self._ws_session is None or self._ws_session.closed:
                self._ws_session = aiohttp.ClientSession()

            ws_url = self.base_url.replace('http', 'ws', 1)
            full_ws_url = f"{ws_url}/tts/websocket?api_key={self._api_key}&cartesia_version={API_VERSION}"
            
            try:
                self._ws_connection = await asyncio.wait_for(
                    self._ws_session.ws_connect(full_ws_url, heartbeat=30.0), timeout=5.0
                )
                return self._ws_connection
            except Exception as e:
                self.emit("error", f"Failed to establish WebSocket connection: {e}")
                raise

    async def _send_task(self, ws: aiohttp.ClientWebSocketResponse, text_iterator: AsyncIterator[str]):
        context_id = os.urandom(8).hex()
        
        voice_payload: dict[str, Any] = {}
        if isinstance(self._voice, str):
            voice_payload["mode"] = "id"
            voice_payload["id"] = self._voice
        else:
            voice_payload["mode"] = "embedding"
            voice_payload["embedding"] = self._voice

        base_payload = {
            "model_id": self.model, "language": self.language,
            "voice": voice_payload,
            "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": self.sample_rate},
            "add_timestamps": True, "context_id": context_id,
        }

        delimiters = {'.', '!', '?', '\n'}
        buffer = ""

        async def send_sentence(sentence: str) -> None:
            if not sentence:
                return
            payload = {**base_payload, "transcript": sentence + " ", "continue": True}
            await ws.send_str(json.dumps(payload))

        def first_delim_pos(buf: str) -> int:
            return min((p for p in (buf.find(d) for d in delimiters) if p != -1), default=-1)

        def over_thresholds(buf: str) -> bool:
            if len(buf) >= MIN_CHARS_FLUSH:
                return True
            if len(buf.split()) >= MIN_WORDS_FLUSH:
                return True
            return False

        aiter = text_iterator.__aiter__()
        while not self._interrupted:
            try:
                next_task = asyncio.create_task(aiter.__anext__())
                try:
                    text_chunk = await asyncio.wait_for(next_task, timeout=INACTIVITY_TIMEOUT_SEC)
                except asyncio.TimeoutError:
                    # Inactivity flush
                    next_task.cancel()
                    if buffer.strip() and not self._interrupted:
                        await send_sentence(buffer.strip())
                        buffer = ""
                    continue

                if not self._interrupted:
                    buffer += text_chunk

                # Greedy punctuation split first
                while True:
                    pos = first_delim_pos(buffer)
                    if pos == -1:
                        break
                    sentence = buffer[:pos + 1].strip()
                    buffer = buffer[pos + 1:]
                    if sentence:
                        await send_sentence(sentence)

                # Threshold-based flush
                if over_thresholds(buffer):
                    await send_sentence(buffer.strip())
                    buffer = ""

            except StopAsyncIteration:
                break

        if buffer.strip():
            await send_sentence(buffer.strip())
        final_payload = {**base_payload, "transcript": " ", "continue": False}
        await ws.send_str(json.dumps(final_payload))

    async def _receive_task(self, ws: aiohttp.ClientWebSocketResponse):
        while not self._interrupted:
            msg = await ws.receive()
            if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING):
                break
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            data = json.loads(msg.data)
            if data.get("type") == "error":
                error_details = json.dumps(data, indent=2)
                self.emit("error", f"Cartesia error: {error_details}")
                break
            if "data" in data and data["data"]:
                audio_chunk = base64.b64decode(data["data"])
                await self._stream_audio(audio_chunk)
            if data.get("done"):
                break
        await self._flush_audio_buffer()

    async def synthesize(
        self, text: AsyncIterator[str] | str, voice_id: Optional[Union[str, List[float]]] = None, **kwargs: Any,
    ) -> None:
        if voice_id:
            self._voice = voice_id

        if not self.audio_track or not self.loop:
            self.emit("error", "Audio track or event loop not set"); return
        
        self._interrupted = False
        self._current_tasks.clear()
        
        if isinstance(text, str):
            async def _string_iterator(): 
                yield text
            text_iterator = _string_iterator()
        else:
            text_iterator = text

        try:
            ws = await self._ensure_ws_connection()
            send_task = asyncio.create_task(self._send_task(ws, text_iterator))
            receive_task = asyncio.create_task(self._receive_task(ws))
            self._current_tasks.extend([send_task, receive_task])
            await asyncio.gather(send_task, receive_task)
        except Exception as e:
            self.emit("error", f"TTS synthesis failed: {str(e)}")
            if self._ws_connection and not self._ws_connection.closed: await self._ws_connection.close()
            self._ws_connection = None

    async def _stream_audio(self, audio_chunk: bytes):
        if self._interrupted:
            return
            
        if not self._first_chunk_sent and self._first_audio_callback:
            self._first_chunk_sent = True
            await self._first_audio_callback()

        self._audio_buffer.extend(audio_chunk)
        while len(self._audio_buffer) >= PLAYBACK_CHUNK_SIZE and not self._interrupted:
            playback_chunk = self._audio_buffer[:PLAYBACK_CHUNK_SIZE]
            self._audio_buffer = self._audio_buffer[PLAYBACK_CHUNK_SIZE:]
            if self.audio_track:
                await self.audio_track.add_new_bytes(playback_chunk)

    async def _flush_audio_buffer(self):
        if self._audio_buffer:
            chunk = self._audio_buffer
            self._audio_buffer = bytearray()
            if len(chunk) < PLAYBACK_CHUNK_SIZE:
                chunk.extend(b'\x00' * (PLAYBACK_CHUNK_SIZE - len(chunk)))
            if not self._first_chunk_sent and self._first_audio_callback:
                self._first_chunk_sent = True
                await self._first_audio_callback()
            if self.audio_track: await self.audio_track.add_new_bytes(chunk)

    async def aclose(self) -> None:
        await super().aclose()
        if self._ws_connection and not self._ws_connection.closed: await self._ws_connection.close()
        if self._ws_session and not self._ws_session.closed: await self._ws_session.close()

    async def interrupt(self) -> None:
        """Interrupt TTS synthesis"""
        self._interrupted = True
        
        for task in self._current_tasks:
            if not task.done():
                task.cancel()
        
        if self._ws_connection and not self._ws_connection.closed:
            await self._ws_connection.close()
            self._ws_connection = None
        
        if self.audio_track: 
            self.audio_track.interrupt()