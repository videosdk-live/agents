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
DEFAULT_VOICE_ID = "f786b574-daa5-4673-aa0c-cbe3e8534c02"
API_VERSION = "2024-06-10"


class CartesiaTTS(TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        voice_id: Union[str, List[float]] = DEFAULT_VOICE_ID,
        language: str = "en",
        base_url: str = "https://api.cartesia.ai",
    ) -> None:
        """Initialize the Cartesia TTS plugin
        Args:
            api_key (str | None, optional): Cartesia API key. Uses CARTESIA_API_KEY environment variable if not provided. Defaults to None.
            model (str): The model to use for the TTS plugin. Defaults to "sonic-2".
            voice_id (Union[str, List[float]]): The voice ID to use for the TTS plugin. Defaults to "794f9389-aac1-45b6-b726-9d9369183238".
            api_key (str | None, optional): Cartesia API key. Uses CARTESIA_API_KEY environment variable if not provided. Defaults to None.
            language (str): The language to use for the TTS plugin. Defaults to "en".
            base_url (str): The base URL to use for the TTS plugin. Defaults to "https://api.cartesia.ai".
        """
        super().__init__(sample_rate=CARTESIA_SAMPLE_RATE, num_channels=CARTESIA_CHANNELS)

        self.model = model
        self.language = language
        self.base_url = base_url
        self._voice = voice_id
        self._first_chunk_sent = False
        self._interrupted = False

        api_key = api_key or os.getenv("CARTESIA_API_KEY")
        if not api_key:
            raise ValueError(
                "Cartesia API key must be provided either through api_key parameter or CARTESIA_API_KEY environment variable")
        self._api_key = api_key

        self._ws_session: aiohttp.ClientSession | None = None
        self._ws_connection: aiohttp.ClientWebSocketResponse | None = None
        self._connection_lock = asyncio.Lock()
        self._receive_task: asyncio.Task | None = None
        self._context_futures: dict[str, asyncio.Future[None]] = {}

    def reset_first_audio_tracking(self) -> None:
        """Reset the first audio tracking state for the next TTS task"""
        self._first_chunk_sent = False

    async def synthesize(
        self, text: AsyncIterator[str] | str, voice_id: Optional[Union[str, List[float]]] = None, **kwargs: Any,
    ) -> None:
        """Synthesize text to speech using Cartesia's streaming WebSocket API."""
        context_id = ""
        try:
            if not self.audio_track or not self.loop:
                self.emit("error", "Audio track or event loop not set")
                return

            if voice_id:
                self._voice = voice_id

            self._interrupted = False

            await self._ensure_ws_connection()
            if not self._ws_connection:
                raise RuntimeError("WebSocket connection is not available.")

            context_id = os.urandom(8).hex()
            done_future: asyncio.Future[None] = asyncio.get_event_loop().create_future()
            self._context_futures[context_id] = done_future

            async def _string_iterator(s: str) -> AsyncIterator[str]:
                yield s

            text_iterator = _string_iterator(text) if isinstance(text, str) else text
            send_task = asyncio.create_task(self._send_task(text_iterator, context_id))

            await done_future
            await send_task

        except Exception as e:
            self.emit("error", f"TTS synthesis failed: {str(e)}")
            raise 
        finally:
            if context_id and context_id in self._context_futures:
                del self._context_futures[context_id]

    async def _send_task(self, text_iterator: AsyncIterator[str], context_id: str):
        """The dedicated task for sending text chunks over the WebSocket."""
        has_sent_transcript = False
        try:
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

            async for text_chunk in text_iterator:
                if self._interrupted: break
                if text_chunk and text_chunk.strip():
                    if not has_sent_transcript:
                        pass

                    payload = {**base_payload, "transcript": text_chunk + " ", "continue": True}
                    if self._ws_connection and not self._ws_connection.closed:
                        await self._ws_connection.send_str(json.dumps(payload))
                    has_sent_transcript = True

        except Exception as e:
            future = self._context_futures.get(context_id)
            if future and not future.done():
                future.set_exception(e)
        finally:
            if has_sent_transcript and not self._interrupted:
                final_payload = {**base_payload, "transcript": " ", "continue": False}
                if self._ws_connection and not self._ws_connection.closed:
                    await self._ws_connection.send_str(json.dumps(final_payload))

    async def _receive_loop(self):
        """A single, long-running task that handles all incoming messages from the WebSocket."""
        try:
            while self._ws_connection and not self._ws_connection.closed:
                msg = await self._ws_connection.receive()
                if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR): break
                if msg.type != aiohttp.WSMsgType.TEXT: continue

                data = json.loads(msg.data)
                context_id = data.get("context_id")
                future = self._context_futures.get(context_id)

                if not future or future.done(): continue

                if data.get("type") == "error":
                    future.set_exception(RuntimeError(f"Cartesia API error: {json.dumps(data)}"))
                elif "data" in data and data["data"]:
                    await self._stream_audio(base64.b64decode(data["data"]))
                elif data.get("done"):
                    future.set_result(None)
        except Exception as e:
            for future in self._context_futures.values():
                if not future.done(): future.set_exception(e)
        finally:
            self._context_futures.clear()

    async def _ensure_ws_connection(self) -> None:
        """Establishes or re-establishes the WebSocket connection if needed."""
        async with self._connection_lock:
            if self._ws_connection and not self._ws_connection.closed: return

            if self._receive_task and not self._receive_task.done(): self._receive_task.cancel()
            if self._ws_connection: await self._ws_connection.close()
            if self._ws_session: await self._ws_session.close()

            try:
                self._ws_session = aiohttp.ClientSession()
                ws_url = self.base_url.replace('http', 'ws', 1)
                full_ws_url = f"{ws_url}/tts/websocket?api_key={self._api_key}&cartesia_version={API_VERSION}"

                self._ws_connection = await asyncio.wait_for(
                    self._ws_session.ws_connect(full_ws_url, heartbeat=30.0), timeout=5.0
                )
                self._receive_task = asyncio.create_task(self._receive_loop())
            except Exception as e:
                self.emit("error", f"Failed to establish WebSocket connection: {e}")
                raise

    async def _stream_audio(self, audio_chunk: bytes):
        """Streams a chunk of audio to the audio track."""
        if self._interrupted or not audio_chunk: return

        if not self._first_chunk_sent and self._first_audio_callback:
            self._first_chunk_sent = True
            await self._first_audio_callback()

        if self.audio_track:
            await self.audio_track.add_new_bytes(audio_chunk)

    async def interrupt(self) -> None:
        """Interrupts any ongoing TTS, stopping audio playback and network activity."""
        self._interrupted = True
        if self.audio_track: self.audio_track.interrupt()
        if self._ws_connection and not self._ws_connection.closed:
            await self._ws_connection.close()

    async def aclose(self) -> None:
        """Gracefully cleans up all resources."""
        await super().aclose()
        self._interrupted = True
        if self._receive_task and not self._receive_task.done(): self._receive_task.cancel()
        if self._ws_connection and not self._ws_connection.closed: await self._ws_connection.close()
        if self._ws_session and not self._ws_session.closed: await self._ws_session.close()