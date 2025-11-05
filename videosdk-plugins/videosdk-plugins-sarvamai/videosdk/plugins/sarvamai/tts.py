from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import time
from typing import Any, AsyncIterator, Optional

import aiohttp
import httpx
from videosdk.agents import TTS
import logging

logger = logging.getLogger(__name__)

SARVAM_SAMPLE_RATE = 24000
SARVAM_CHANNELS = 1
DEFAULT_MODEL = "bulbul:v2"
DEFAULT_SPEAKER = "anushka"
DEFAULT_LANGUAGE = "en-IN"
SARVAM_TTS_URL_STREAMING = "wss://api.sarvam.ai/text-to-speech/ws"
SARVAM_TTS_URL_HTTP = "https://api.sarvam.ai/text-to-speech"


class SarvamAITTS(TTS):
    """
    A unified Sarvam.ai Text-to-Speech (TTS) plugin that supports both real-time
    streaming via WebSockets and batch synthesis via HTTP. This version is optimized
    for robust, long-running sessions and responsive non-streaming playback.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        language: str = DEFAULT_LANGUAGE,
        speaker: str = DEFAULT_SPEAKER,
        enable_streaming: bool = True,
        sample_rate: int = SARVAM_SAMPLE_RATE,
        output_audio_codec: str = "linear16",
    ) -> None:
        """
        Initializes the SarvamAITTS plugin.

        Args:
            api_key (Optional[str]): The Sarvam.ai API key. If not provided, it will
                be read from the SARVAMAI_API_KEY environment variable.
            model (str): The TTS model to use.
            language (str): The target language code (e.g., "en-IN").
            speaker (str): The desired speaker for the voice.
            enable_streaming (bool): If True, uses WebSockets for low-latency streaming.
                If False, uses HTTP for batch synthesis.
            sample_rate (int): The audio sample rate.
            output_audio_codec (str): The desired output audio codec.
        """
        super().__init__(sample_rate=sample_rate, num_channels=SARVAM_CHANNELS)

        self.api_key = api_key or os.getenv("SARVAMAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Sarvam AI API key required. Provide either:\n"
                "1. api_key parameter, OR\n"
                "2. SARVAMAI_API_KEY environment variable"
            )

        self.model = model
        self.language = language
        self.speaker = speaker
        self.enable_streaming = enable_streaming
        self.output_audio_codec = output_audio_codec
        self.base_url_ws = SARVAM_TTS_URL_STREAMING
        self.base_url_http = SARVAM_TTS_URL_HTTP

        self._ws_session: aiohttp.ClientSession | None = None
        self._ws_connection: aiohttp.ClientWebSocketResponse | None = None
        self._receive_task: asyncio.Task | None = None
        self._connection_lock = asyncio.Lock()

        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=15.0, read=30.0, write=5.0, pool=5.0),
            follow_redirects=True,
        )

        self._interrupted = False
        self._first_chunk_sent = False
        self.ws_count = 0

    def reset_first_audio_tracking(self) -> None:
        """Resets tracking for the first audio chunk latency."""
        self._first_chunk_sent = False

    async def synthesize(
        self,
        text: AsyncIterator[str] | str,
        *,
        language: Optional[str] = None,
        speaker: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Main entry point for synthesizing audio. Routes to either streaming
        or batch (HTTP) mode. The HTTP mode uses a smart buffering strategy
        to balance responsiveness and audio quality.
        """
        try:
            if not self.audio_track or not self.loop:
                logger.error("error", "Audio track or event loop not initialized")
                return

            self.language = language or self.language
            self.speaker = speaker or self.speaker
            self._interrupted = False
            self.reset_first_audio_tracking()

            if self.enable_streaming:
                await self._stream_synthesis(text)
            else:
                if isinstance(text, str):
                    if text.strip():
                        await self._http_synthesis(text)
                else:
                    chunk_buffer = []
                    HTTP_CHUNK_BUFFER_SIZE = 4
                    LLM_PAUSE_TIMEOUT = 1.0 

                    text_iterator = text.__aiter__()
                    while not self._interrupted:
                        try:
                            chunk = await asyncio.wait_for(text_iterator.__anext__(), timeout=LLM_PAUSE_TIMEOUT)
                            
                            if chunk and chunk.strip():
                                chunk_buffer.append(chunk)
                            
                            if len(chunk_buffer) >= HTTP_CHUNK_BUFFER_SIZE:
                                combined_text = "".join(chunk_buffer)
                                await self._http_synthesis(combined_text)
                                chunk_buffer.clear()

                        except asyncio.TimeoutError:
                            if chunk_buffer:
                                combined_text = "".join(chunk_buffer)
                                await self._http_synthesis(combined_text)
                                chunk_buffer.clear()
                        
                        except StopAsyncIteration:
                            if chunk_buffer:
                                combined_text = "".join(chunk_buffer)
                                await self._http_synthesis(combined_text)
                            break

        except Exception as e:
            logger.error("error", f"Sarvam TTS synthesis failed: {e}")

    async def _stream_synthesis(self, text: AsyncIterator[str] | str) -> None:
        """
        Manages the WebSocket synthesis workflow, ensuring a fresh connection
        for each synthesis task to guarantee reliability.
        """
        try:
            # await self._close_ws_resources()
            await self._ensure_ws_connection()
            

            if isinstance(text, str):
                async def _str_iter():
                    yield text
                text_iter = _str_iter()
            else:
                text_iter = text

            await self._send_text_chunks(text_iter)
        except Exception as e:
            logger.error("error", f"WebSocket streaming failed: {e}. Trying HTTP fallback.")
            try:
                full_text = ""
                if isinstance(text, str):
                    full_text = text
                else:
                    async for chunk in text:
                        full_text += chunk
                
                if full_text.strip():
                    await self._http_synthesis(full_text.strip())
            except Exception as http_e:
                logger.error("error", f"HTTP fallback also failed: {http_e}")

    async def _ensure_ws_connection(self) -> None:
        """Establishes and maintains a persistent WebSocket connection."""
        async with self._connection_lock:
            if self._ws_connection and not self._ws_connection.closed:
                return
            try:
                self._ws_session = aiohttp.ClientSession()
                headers = {"Api-Subscription-Key": self.api_key}
                self._ws_connection = await asyncio.wait_for(
                    self._ws_session.ws_connect(
                        self.base_url_ws, headers=headers, heartbeat=20
                    ),
                    timeout=10.0,
                )
                self._receive_task = asyncio.create_task(self._recv_loop())
                await self._send_initial_config()
                self.ws_count = self.ws_count + 1
                logger.info(f"WS connection numbers: {self.ws_count}")
            except Exception as e:
                logger.error("error", f"Failed to connect to WebSocket: {e}")
                raise

    async def _send_initial_config(self) -> None:
        """Sends the initial configuration message to the WebSocket server."""
        config_payload = {
            "type": "config",
            "data": {
                "target_language_code": self.language,
                "speaker": self.speaker,
                "speech_sample_rate": str(self.sample_rate),
                "output_audio_codec": self.output_audio_codec,
            },
        }
        if self._ws_connection:
            await self._ws_connection.send_str(json.dumps(config_payload))

    async def _send_text_chunks(self, text_iterator: AsyncIterator[str]):
        """Sends text to the WebSocket, chunking it by word count or time."""
        if not self._ws_connection:
            raise ConnectionError("WebSocket is not connected.")
        try:
            buffer = []
            MIN_WORDS, MAX_DELAY = 4, 1.0
            last_send_time = asyncio.get_event_loop().time()

            async for text_chunk in text_iterator:
                if self._interrupted:
                    break

                words = re.findall(r"\b[\w'-]+\b", text_chunk)
                if not words:
                    continue

                buffer.extend(words)
                now = asyncio.get_event_loop().time()

                if len(buffer) >= MIN_WORDS or (now - last_send_time > MAX_DELAY):
                    combined_text = " ".join(buffer).strip()
                    if combined_text:
                        payload = {"type": "text", "data": {"text": combined_text}}
                        await self._ws_connection.send_str(json.dumps(payload))
                    buffer.clear()
                    last_send_time = now

            if buffer and not self._interrupted:
                combined_text = " ".join(buffer).strip()
                if combined_text:
                    payload = {"type": "text", "data": {"text": combined_text}}
                    await self._ws_connection.send_str(json.dumps(payload))
                    if not self._first_chunk_sent and hasattr(self, '_first_audio_callback') and self._first_audio_callback:
                        self._first_chunk_sent = True
                        asyncio.create_task(self._first_audio_callback())

            if not self._interrupted:
                await self._ws_connection.send_str(json.dumps({"type": "flush"}))
        except Exception as e:
            logger.error("error", f"Failed to send text chunks via WebSocket: {e}")

    async def _recv_loop(self):
        """Continuously listens for and processes incoming WebSocket messages."""
        try:
            while self._ws_connection and not self._ws_connection.closed:
                msg = await self._ws_connection.receive()
                if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                data = json.loads(msg.data)
                msg_type = data.get("type")

                if msg_type == "audio":

                    if not self._first_chunk_sent and hasattr(self, '_first_audio_callback') and self._first_audio_callback:
                        self._first_chunk_sent = True
                        asyncio.create_task(self._first_audio_callback())
                    
                    await self._handle_audio_data(data.get("data"))
                
                elif msg_type == "event" and data.get("data", {}).get("event_type") == "final":
                    logger.error("done", "TTS completed")
                
                elif msg_type == "error":
                    error_msg = data.get("data", {}).get("message", "Unknown WS error")
                    logger.error("error", f"Sarvam WebSocket error: {error_msg}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("error", f"WebSocket receive loop error: {e}")

    async def _handle_audio_data(self, audio_data: Optional[dict[str, Any]]):
        """Processes audio data received from the WebSocket."""
        if not audio_data or self._interrupted:
            return

        audio_b64 = audio_data.get("audio")
        if not audio_b64:
            return

        try:
            audio_bytes = base64.b64decode(audio_b64)
            if not self.audio_track:
                return


            await self.audio_track.add_new_bytes(audio_bytes)
        except Exception as e:
            logger.error("error", f"Failed to process WebSocket audio: {e}")


    async def _reinitialize_http_client(self):
        """Safely closes the current httpx client and creates a new one."""
        logger.info("Re-initializing HTTP client.")
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=15.0, read=30.0, write=5.0, pool=5.0),
            follow_redirects=True,
        )

    async def _http_synthesis(self, text: str) -> None:
        """Performs TTS synthesis using HTTP with a retry for connection errors."""
        payload = { "text": text, "target_language_code": self.language, "speaker": self.speaker, "speech_sample_rate": str(self.sample_rate), "model": self.model, "output_audio_codec": self.output_audio_codec }
        headers = { "Content-Type": "application/json", "api-subscription-key": self.api_key }
        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                if self._http_client.is_closed:
                    await self._reinitialize_http_client()
                response = await self._http_client.post(self.base_url_http, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                if not data.get("audios"):
                    logger.error("error", f"No audio data in HTTP response: {data}")
                    return
                audio_b64 = data["audios"][0]
                audio_bytes = base64.b64decode(audio_b64)
                if not self._first_chunk_sent and self._first_audio_callback:
                    self._first_chunk_sent = True
                    await self._first_audio_callback()

                await self._stream_http_audio(audio_bytes)
                return
            except httpx.HTTPStatusError as e:
                logger.error("error", f"HTTP error: {e.response.status_code} - {e.response.text}")
                logger.info(response)
                raise e
            except (httpx.NetworkError, httpx.ConnectError, httpx.ReadTimeout) as e:
                logger.warning(f"HTTP connection error on attempt {attempt + 1}: {e}")
                if attempt < max_attempts - 1:
                    await self._reinitialize_http_client()
                    continue
                else:
                    logger.error("error", f"HTTP synthesis failed after {max_attempts} connection attempts.")
                    raise e
            except Exception as e:
                logger.error("error", f"An unexpected HTTP synthesis error occurred: {e}")
                raise e

    async def _stream_http_audio(self, audio_bytes: bytes) -> None:
        """
        Streams decoded HTTP audio bytes to the audio track by sending two
        20ms chunks at a time (a 40ms block) to ensure real-time playback.
        """
        single_chunk_size = int(self.sample_rate * self.num_channels * 2 * 20 / 1000)
        
        block_size = single_chunk_size * 2
        
        raw_audio = self._remove_wav_header(audio_bytes)

        for i in range(0, len(raw_audio), block_size):
            if self._interrupted:
                break
            
            block = raw_audio[i : i + block_size]

            if 0 < len(block) < block_size:
                block += b"\x00" * (block_size - len(block))

                
            if self.audio_track:
                asyncio.create_task(self.audio_track.add_new_bytes(block))

    def _remove_wav_header(self, audio_bytes: bytes) -> bytes:
        """Removes the WAV header if present."""
        if audio_bytes.startswith(b"RIFF"):
            data_pos = audio_bytes.find(b"data")
            if data_pos != -1:
                return audio_bytes[data_pos + 8:]
        return audio_bytes

    async def interrupt(self) -> None:
        """Interrupts any ongoing TTS synthesis."""
        self._interrupted = True
        if self.audio_track:
            self.audio_track.interrupt()
        
    async def _close_ws_resources(self) -> None:
        """Helper to clean up all WebSocket-related resources."""
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
        if self._ws_connection and not self._ws_connection.closed:
            await self._ws_connection.close()
        if self._ws_session and not self._ws_session.closed:
            await self._ws_session.close()
        self._receive_task = self._ws_connection = self._ws_session = None

    async def aclose(self) -> None:
        """Gracefully closes all connections and cleans up resources."""
        self._interrupted = True
        await self._close_ws_resources()
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
        await super().aclose()
