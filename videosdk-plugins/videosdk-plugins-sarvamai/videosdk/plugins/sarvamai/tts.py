from __future__ import annotations
import asyncio
import base64
import json
import os
import re
import time
from typing import Any, AsyncIterator, Optional, Union
import logging

logger = logging.getLogger(__name__)
from PIL.Image import logger
import aiohttp
from videosdk.agents import TTS


SARVAM_SAMPLE_RATE = 24000
SARVAM_CHANNELS = 1
DEFAULT_MODEL = "bulbul:v2"
DEFAULT_SPEAKER = "anushka"
DEFAULT_LANGUAGE = "en-IN"
SARVAM_TTS_URL = "wss://api.sarvam.ai/text-to-speech/ws"


class SarvamAITTS(TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        language: str = DEFAULT_LANGUAGE,
        speaker: str = DEFAULT_SPEAKER,
        sample_rate: int = SARVAM_SAMPLE_RATE,
        base_url: str = SARVAM_TTS_URL,
        output_audio_codec: str = "linear16",
        output_audio_bitrate: str = "64k",
    ) -> None:
        """Sarvam.ai Text-to-Speech plugin for real-time streaming."""
        super().__init__(sample_rate=sample_rate, num_channels=SARVAM_CHANNELS)

        self.api_key = api_key or os.getenv("SARVAMAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Sarvam API key must be provided via parameter or SARVAMAI_API_KEY env var."
            )

        self.model = model
        self.language = language
        self.speaker = speaker
        self.base_url = base_url
        self.output_audio_codec = output_audio_codec
        self.output_audio_bitrate = output_audio_bitrate

        self._ws_session: aiohttp.ClientSession | None = None
        self._ws_connection: aiohttp.ClientWebSocketResponse | None = None
        self._connection_lock = asyncio.Lock()
        self._receive_task: asyncio.Task | None = None
        self._interrupted = False
        self._first_chunk_sent = False
        self._ttfb: Optional[float] = None

    def reset_first_audio_tracking(self) -> None:
        """Reset the first audio tracking for the next task."""
        self._first_chunk_sent = False
        self._ttfb = None

    async def _ensure_ws_connection(self) -> None:
        """Ensures a persistent WebSocket connection exists."""
        async with self._connection_lock:
            if self._ws_connection and not self._ws_connection.closed:
                return

            if self._receive_task and not self._receive_task.done():
                self._receive_task.cancel()
            if self._ws_connection:
                await self._ws_connection.close()
            if self._ws_session:
                await self._ws_session.close()

            try:
                self._ws_session = aiohttp.ClientSession()
                headers = {"Api-Subscription-Key": self.api_key}
                self._ws_connection = await asyncio.wait_for(
                    self._ws_session.ws_connect(self.base_url, headers=headers, heartbeat=20),
                    timeout=5.0,
                )
                self._receive_task = asyncio.create_task(self._receive_loop())
                await self._send_initial_config()
            except Exception as e:
                self.emit("error", f"Failed to connect to Sarvam TTS: {e}")
                raise

    async def _send_initial_config(self) -> None:
        """Sends initial configuration to Sarvam TTS once after connecting."""
        config_payload = {
            "type": "config",
            "data": {
                "target_language_code": self.language,
                "speaker": self.speaker,
                "speech_sample_rate": str(self.sample_rate),
                "output_audio_codec": self.output_audio_codec,
                "output_audio_bitrate": self.output_audio_bitrate,
            },
        }
        await self._ws_connection.send_str(json.dumps(config_payload))

    async def synthesize(
        self,
        text: AsyncIterator[str] | str,
        *,
        language: Optional[str] = None,
        speaker: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Stream text-to-speech synthesis using Sarvam TTS WebSocket."""
        try:
            if not self.audio_track or not self.loop:
                self.emit("error", "Audio track or event loop not initialized")
                return

            await self._ensure_ws_connection()
            if not self._ws_connection:
                raise RuntimeError("WebSocket connection not available")

            self._interrupted = False
            self.reset_first_audio_tracking()

            if language:
                self.language = language
            if speaker:
                self.speaker = speaker

            # Prepare text iterator
            if isinstance(text, str):
                async def _string_iter():
                    yield text
                text_iter = _string_iter()
            else:
                text_iter = text

            send_task = asyncio.create_task(self._send_text_chunks(text_iter))
            await send_task
        except Exception as e:
            self.emit("error", f"TTS synthesis failed: {e}")

    async def _send_text_chunks(self, text_iterator: AsyncIterator[str]):
        """Send text chunks to Sarvam TTS WebSocket in grouped batches (4–5 words per chunk)."""
        try:
            buffer = []
            MIN_WORDS = 4        
            MAX_DELAY = 1.0      
            last_send_time = asyncio.get_event_loop().time()

            async for text_chunk in text_iterator:
                if self._interrupted:
                    break

                if not text_chunk or not text_chunk.strip():
                    continue

                words = re.findall(r"\b[\w'-]+\b", text_chunk)
                if not words:
                    continue

                buffer.extend(words)
                now = asyncio.get_event_loop().time()

                if len(buffer) >= MIN_WORDS or (now - last_send_time > MAX_DELAY):
                    combined_text = " ".join(buffer).strip()
                    payload = {"type": "text", "data": {"text": combined_text}}
                    await self._ws_connection.send_str(json.dumps(payload))
                    logger.debug(f"[SarvamTTS] Sent chunk: {combined_text!r}")
                    buffer.clear()
                    last_send_time = now

            if buffer and not self._interrupted:
                combined_text = " ".join(buffer).strip()
                payload = {"type": "text", "data": {"text": combined_text}}
                await self._ws_connection.send_str(json.dumps(payload))
                logger.debug(f"[SarvamTTS] Final flush: {combined_text!r}")

            if not self._interrupted:
                await self._ws_connection.send_str(json.dumps({"type": "flush"}))

        except Exception as e:
            self.emit("error", f"Failed to send text chunks: {e}")

    async def _receive_loop(self):
        """Handles all incoming messages from Sarvam TTS."""
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
                    await self._handle_audio(data.get("data"))
                elif msg_type == "event":
                    if data.get("data", {}).get("event_type") == "final":
                        self.emit("done", "TTS completed")
                elif msg_type == "error":
                    error_msg = data.get("data", {}).get("message", "Unknown error")
                    self.emit("error", f"Sarvam API error: {error_msg}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.emit("error", f"Receive loop error: {e}")

    async def _handle_audio(self, audio_data: Optional[dict[str, Any]]):
        """Handle audio data from Sarvam WebSocket."""
        if not audio_data:
            return

        if self._ttfb is None:
            self._ttfb = time.time()
            self.emit("ttfb", self._ttfb)

        audio_b64 = audio_data.get("audio")
        if not audio_b64:
            return

        try:
            audio_bytes = base64.b64decode(audio_b64)
            if self._interrupted or not self.audio_track:
                return

            if not self._first_chunk_sent and self._first_audio_callback:
                self._first_chunk_sent = True
                await self._first_audio_callback()

            await self.audio_track.add_new_bytes(audio_bytes)
        except Exception as e:
            self.emit("error", f"Failed to process audio: {e}")

    async def interrupt(self) -> None:
        """Interrupt the ongoing TTS session."""
        self._interrupted = True
        if self.audio_track:
            self.audio_track.interrupt()
        if self._ws_connection and not self._ws_connection.closed:
            await self._ws_connection.close()

    async def aclose(self) -> None:
        """Gracefully close the TTS connection."""
        await super().aclose()
        self._interrupted = True
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
        if self._ws_connection and not self._ws_connection.closed:
            await self._ws_connection.close()
        if self._ws_session and not self._ws_session.closed:
            await self._ws_session.close()
