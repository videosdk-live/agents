from __future__ import annotations

from typing import Any, AsyncIterator, Literal, Optional, Union
import os
import json
import aiohttp
import asyncio
import base64
from urllib.parse import urlencode

from videosdk.agents import TTS, segment_text

NEUPHONIC_DEFAULT_SAMPLE_RATE = 22050
NEUPHONIC_CHANNELS = 1
NEUPHONIC_BASE_URL = "wss://eu-west-1.api.neuphonic.com"
NEUPHONIC_SSE_BASE_URL = "https://eu-west-1.api.neuphonic.com"


class NeuphonicTTS(TTS):
    def __init__(
        self,
        *,
        lang_code: str = "en",
        voice_id: Optional[str] = None,
        speed: float = 0.8,
        sampling_rate: int = NEUPHONIC_DEFAULT_SAMPLE_RATE,
        encoding: Literal["pcm_linear", "pcm_mulaw"] = "pcm_linear",
        api_key: str | None = None,
        base_url: str = NEUPHONIC_BASE_URL,
    ) -> None:
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
        self._current_tasks: list[asyncio.Task] = []

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

    async def synthesize(
        self,
        text: AsyncIterator[str] | str,
        **kwargs: Any,
    ) -> None:
        try:
            if not self.audio_track or not self.loop:
                self.emit("error", "Audio track or event loop not set")
                return

            self._interrupted = False
            self._current_tasks.clear()

            if isinstance(text, AsyncIterator):
                await self._streaming_websocket_synthesis(text)
            else:
                await self._websocket_synthesis(text)

        except Exception as e:
            self.emit("error", f"TTS synthesis failed: {str(e)}")

    async def _streaming_websocket_synthesis(self, text: AsyncIterator[str]) -> None:
        """Streaming synthesis with single WebSocket connection for multiple text segments"""
        params = {
            "api_key": self.api_key,
            "speed": self.speed,
            "sampling_rate": self._sample_rate,
            "encoding": self.encoding
        }

        if self.voice_id:
            params["voice_id"] = self.voice_id

        query_string = urlencode(params)
        ws_url = f"{self.base_url}/speak/{self.lang_code}?{query_string}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(ws_url) as ws:
                    listener_task = asyncio.create_task(
                        self._listen_to_ws_messages(ws))
                    self._current_tasks.append(listener_task)

                    async for segment in segment_text(text):
                        if self._interrupted:
                            break
                        if segment.strip():
                            await ws.send_str(f"{segment} <STOP>")
                            await asyncio.sleep(0.01)

                    if not self._interrupted:
                        await listener_task

        except aiohttp.ClientError as e:
            self.emit("error", f"WebSocket connection failed: {str(e)}")
        except Exception as e:
            self.emit("error", f"Streaming synthesis failed: {str(e)}")

    async def _listen_to_ws_messages(self, ws) -> None:
        """Listen to WebSocket messages concurrently while sending text"""
        try:
            async for msg in ws:
                if self._interrupted:
                    break
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        if "data" in data and "audio" in data["data"]:
                            audio_data = base64.b64decode(
                                data["data"]["audio"])

                            if self.encoding == "pcm_linear":
                                await self._stream_audio_chunks(audio_data)
                            elif self.encoding == "pcm_mulaw":
                                await self._stream_audio_chunks(audio_data)

                    except json.JSONDecodeError:
                        self.emit(
                            "error", f"Invalid JSON response: {msg.data}")

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    self.emit(
                        "error", f"WebSocket connection error: {ws.exception()}")
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    break

        except Exception as e:
            self.emit("error", f"WebSocket message listening failed: {str(e)}")

    async def _websocket_synthesis(self, text: str) -> None:
        """WebSocket-based streaming synthesis"""
        params = {
            "api_key": self.api_key,
            "speed": self.speed,
            "sampling_rate": self._sample_rate,
            "encoding": self.encoding,
        }

        if self.voice_id:
            params["voice_id"] = self.voice_id

        query_string = urlencode(params)
        ws_url = f"{self.base_url}/speak/{self.lang_code}?{query_string}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(ws_url) as ws:
                    await ws.send_str(f"{text} <STOP>")

                    async for msg in ws:
                        if self._interrupted:
                            break
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                data = json.loads(msg.data)
                                if "data" in data and "audio" in data["data"]:
                                    audio_data = base64.b64decode(
                                        data["data"]["audio"])

                                    if self.encoding == "pcm_linear":
                                        await self._stream_audio_chunks(audio_data)
                                    elif self.encoding == "pcm_mulaw":
                                        await self._stream_audio_chunks(audio_data)

                            except json.JSONDecodeError:
                                self.emit(
                                    "error", f"Invalid JSON response: {msg.data}")

                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            self.emit(
                                "error", f"WebSocket connection error: {ws.exception()}")
                            break
                        elif msg.type == aiohttp.WSMsgType.CLOSED:
                            break

        except aiohttp.ClientError as e:
            self.emit("error", f"WebSocket connection failed: {str(e)}")
        except Exception as e:
            self.emit("error", f"Streaming synthesis failed: {str(e)}")

    async def _stream_audio_chunks(self, audio_bytes: bytes) -> None:
        """Stream audio data in chunks for smooth playback"""
        if self._interrupted:
            return

        chunk_duration_ms = 20
        bytes_per_sample = 2
        chunk_size = int(self._sample_rate * NEUPHONIC_CHANNELS *
                         bytes_per_sample * chunk_duration_ms / 1000)

        if chunk_size % 2 != 0:
            chunk_size += 1

        for i in range(0, len(audio_bytes), chunk_size):
            if self._interrupted:
                break
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

    async def _sse_synthesis(self, text: str) -> None:
        """SSE-based synthesis (alternative to WebSocket)"""
        url = f"{NEUPHONIC_SSE_BASE_URL}/sse/speak/{self.lang_code}"

        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        payload = {
            "text": text,
            "speed": self.speed,
            "sampling_rate": self._sample_rate,
            "encoding": self.encoding,
        }

        if self.voice_id:
            payload["voice_id"] = self.voice_id

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    response.raise_for_status()

                    async for line in response.content:
                        line_str = line.decode('utf-8').strip()

                        if line_str.startswith("data: "):
                            try:
                                json_data = json.loads(line_str[6:])
                                if "data" in json_data and "audio" in json_data["data"]:
                                    audio_data = base64.b64decode(
                                        json_data["data"]["audio"])
                                    await self._stream_audio_chunks(audio_data)
                            except json.JSONDecodeError:
                                continue

        except aiohttp.ClientResponseError as e:
            if e.status == 403:
                self.emit(
                    "error", "Neuphonic authentication failed. Please check your API key.")
            else:
                self.emit("error", f"Neuphonic HTTP error: {e.status}")
        except Exception as e:
            self.emit("error", f"SSE synthesis failed: {str(e)}")

    async def aclose(self) -> None:
        """Cleanup resources"""
        await super().aclose()

    async def interrupt(self) -> None:
        """Interrupt the TTS process"""
        self._interrupted = True

        for task in self._current_tasks:
            if not task.done():
                task.cancel()

        if self.audio_track:
            self.audio_track.interrupt()
