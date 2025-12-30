from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import aiohttp
import numpy as np

from videosdk.agents import STT,STTResponse,SpeechData,SpeechEventType,global_event_emitter
try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


GLADIA_API_URL = "https://api.gladia.io/v2/live"


class GladiaSTT(STT):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "solaria-1",
        language: str = "en",
        input_sample_rate: int = 48000,
        output_sample_rate: int = 16000,
        encoding: str = "wav/pcm",
        bit_depth: int = 16,
        channels: int = 1,
        receive_partial_transcripts: bool = False,
    ) -> None:
        """Initialize the Gladia STT plugin with WebSocket support.
        Args:
            api_key: Gladia API key
            model: The model to use (default: "fast", options: "fast" or "accurate")
            language: The language code (default: "en")
            input_sample_rate: Input sample rate (default: 48000)
            output_sample_rate: Output sample rate for Gladia (default: 16000)
            encoding: Audio encoding format (default: "wav/pcm")
            bit_depth: Bit depth (default: 16)
            channels: Number of audio channels (default: 1 for mono)
            receive_partial_transcripts: Whether to receive partial transcripts (default: False)
        """
        super().__init__()

        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is not installed. Please install it with 'pip install scipy'")

        self.api_key = api_key or os.getenv("GLADIA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gladia API key must be provided either through api_key parameter "
                "or GLADIA_API_KEY environment variable"
            )

        self.model = model
        self.language = language
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.encoding = encoding
        self.bit_depth = bit_depth
        self.channels = channels
        self.receive_partial_transcripts = receive_partial_transcripts

        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._ws_task: asyncio.Task | None = None

        self._is_speaking = False
        self._session_id: str | None = None
        self._ws_url: str | None = None
        self._is_connected = False
        self._init_lock = asyncio.Lock()

    async def _initialize_session(self) -> None:
        async with self._init_lock:
            if self._ws_url is not None:
                return

            if self._session is None:
                self._session = aiohttp.ClientSession()

            config = {
                "encoding": self.encoding,
                "sample_rate": self.output_sample_rate,
                "bit_depth": self.bit_depth,
                "channels": self.channels,
                "language_config": {
                    "languages": [self.language],
                },
                "messages_config": {
                    "receive_partial_transcripts": self.receive_partial_transcripts
                },
            }

            if self.model:
                config["model"] = self.model

            headers = {
                "Content-Type": "application/json",
                "x-gladia-key": self.api_key,
            }

            async with self._session.post(
                GLADIA_API_URL,
                headers=headers,
                json=config,
            ) as response:
                if response.status not in (200, 201):
                    error_text = await response.text()
                    raise RuntimeError(
                        f"Failed to initialize Gladia session: {response.status} - {error_text}"
                    )

                data = await response.json()
                self._session_id = data["id"]
                self._ws_url = data["url"]

    async def _ensure_websocket(self) -> aiohttp.ClientWebSocketResponse:
        """Ensure WebSocket connection is established."""
        if self._ws is None or self._ws.closed:
            await self._connect_websocket()

        if self._ws is None:
            raise RuntimeError("Failed to establish WebSocket connection")

        return self._ws

    async def _connect_websocket(self) -> None:
        if self._is_connected:
            return

        if self._ws_url is None:
            await self._initialize_session()

        self._ws = await self._session.ws_connect(self._ws_url)
        self._is_connected = True
        self._ws_task = asyncio.create_task(self._process_messages())

    async def process_audio(self, audio_frames: bytes, **kwargs: Any) -> None:
        """Process audio frames and send to WebSocket."""
        if not self._is_connected:
            await asyncio.sleep(0.1)

        resampled_audio = self._resample_audio(audio_frames)
        if not resampled_audio:
            return

        ws = await self._ensure_websocket()
        await ws.send_bytes(resampled_audio)

    async def _process_messages(self) -> None:
        """Process incoming WebSocket messages."""
        if self._ws is None:
            return

        async for msg in self._ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                await self._handle_message(data)
            elif msg.type in (
                aiohttp.WSMsgType.ERROR,
                aiohttp.WSMsgType.CLOSED,
            ):
                self._is_connected = False
                break

    async def _handle_message(self, data: dict) -> None:
        """Handle different message types from Gladia API."""
        msg_type = data.get("type")

        if msg_type == "transcript":
            transcript_data = data.get("data", {})
            utterance = transcript_data.get("utterance", {})

            transcript_text = utterance.get("text", "").strip()
            language = utterance.get("language", self.language)
            is_final = transcript_data.get("is_final", True)

            if transcript_text and self._transcript_callback:
                if not self._is_speaking:
                    self._is_speaking = True
                    global_event_emitter.emit("speech_started")

                event_type = (
                    SpeechEventType.FINAL
                    if is_final
                    else SpeechEventType.INTERIM
                )

                event = STTResponse(
                    event_type=event_type,
                    data=SpeechData(
                        text=transcript_text,
                        language=language,
                        confidence=1.0,
                    ),
                )

                await self._transcript_callback(event)

                if is_final and self._is_speaking:
                    self._is_speaking = False
                    global_event_emitter.emit("speech_stopped")

    def _resample_audio(self, audio_bytes: bytes) -> bytes:
        """Resample audio from input sample rate to output sample rate and convert to mono."""
        if not audio_bytes:
            return b""

        raw_audio = np.frombuffer(audio_bytes, dtype=np.int16)
        if raw_audio.size == 0:
            return b""

        if raw_audio.size % 2 == 0 and self.channels == 1:
            stereo_audio = raw_audio.reshape(-1, 2)
            mono_audio = stereo_audio.astype(np.float32).mean(axis=1)
        else:
            mono_audio = raw_audio.astype(np.float32)

        if self.input_sample_rate != self.output_sample_rate:
            output_length = int(
                len(mono_audio)
                * self.output_sample_rate
                / self.input_sample_rate
            )
            resampled_data = signal.resample(mono_audio, output_length)
        else:
            resampled_data = mono_audio

        resampled_data = np.clip(resampled_data, -32767, 32767)
        return resampled_data.astype(np.int16).tobytes()

    async def stop_stream(self) -> None:
        """Send stop stream message to Gladia API."""
        if self._ws and not self._ws.closed:
            await self._ws.send_str(json.dumps({"type": "stop_recording"}))

    async def aclose(self) -> None:
        """Close WebSocket connection and cleanup."""
        await self.stop_stream()

        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass

        if self._ws and not self._ws.closed:
            await self._ws.close()

        if self._session and not self._session.closed:
            await self._session.close()

        self._is_connected = False
        await super().aclose()
