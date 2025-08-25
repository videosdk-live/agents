from __future__ import annotations

from typing import Any, AsyncIterator, Optional, Union, Literal
import os
import asyncio
import base64
import httpx
from dataclasses import dataclass

from videosdk.agents import TTS, segment_text

GOOGLE_SAMPLE_RATE = 24000
GOOGLE_CHANNELS = 1
GOOGLE_TTS_ENDPOINT = "https://texttospeech.googleapis.com/v1/text:synthesize"


@dataclass
class GoogleVoiceConfig:
    languageCode: str = "en-US"
    name: str = "en-US-Chirp3-HD-Aoede"
    ssmlGender: str = "FEMALE"


class GoogleTTS(TTS):
    def __init__(
        self,
        *,
        speed: float = 1.0,
        pitch: float = 0.0,
        api_key: str | None = None,
        response_format: Literal["pcm"] = "pcm",
        voice_config: GoogleVoiceConfig | None = None,
    ) -> None:
        super().__init__(sample_rate=GOOGLE_SAMPLE_RATE, num_channels=GOOGLE_CHANNELS)

        self.speed = speed
        self.pitch = pitch
        self.response_format = response_format
        self.audio_track = None
        self.loop = None
        self._first_chunk_sent = False
        self.voice_config = voice_config or GoogleVoiceConfig()
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Google TTS API key required. Provide either:\n"
                "1. api_key parameter, OR\n"
                "2. GOOGLE_API_KEY environment variable"
            )

        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=15.0, read=30.0,
                                  write=5.0, pool=5.0),
            follow_redirects=True,
        )

    def reset_first_audio_tracking(self) -> None:
        """Reset the first audio tracking state for next TTS task"""
        self._first_chunk_sent = False

    async def synthesize(
        self,
        text: AsyncIterator[str] | str,
        **kwargs: Any,
    ) -> None:
        try:
            if isinstance(text, AsyncIterator):
                async for segment in segment_text(text):
                    await self._synthesize_audio(segment)
            else:
                await self._synthesize_audio(text)

            if not self.audio_track or not self.loop:
                self.emit("error", "Audio track or loop not initialized")
                return

        except Exception as e:
            self.emit("error", f"Google TTS synthesis failed: {str(e)}")

    async def _synthesize_audio(self, text: str) -> None:
        """Synthesize text to speech using Google TTS REST API"""
        try:
            voice_config = {
                "languageCode": self.voice_config.languageCode,
                "name": self.voice_config.name,
            }

            if not self.voice_config.name.startswith("en-US-Studio"):
                voice_config["ssmlGender"] = self.voice_config.ssmlGender

            payload = {
                "input": {"text": text},
                "voice": voice_config,
                "audioConfig": {
                    "audioEncoding": "LINEAR16",
                    "speakingRate": self.speed,
                    "pitch": self.pitch,
                    "sampleRateHertz": GOOGLE_SAMPLE_RATE,
                },
            }

            response = await self._http_client.post(
                GOOGLE_TTS_ENDPOINT, params={"key": self.api_key}, json=payload
            )
            response.raise_for_status()

            response_data = response.json()
            audio_content = response_data.get("audioContent")
            if not audio_content:
                self.emit("error", "No audio content received from Google TTS")
                return

            audio_bytes = base64.b64decode(audio_content)

            if not audio_bytes:
                self.emit("error", "Decoded audio bytes are empty")
                return

            await self._stream_audio_chunks(audio_bytes)

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                self.emit(
                    "error", "Google TTS authentication failed. Please check your API key.")
            elif e.response.status_code == 400:
                try:
                    error_data = e.response.json()
                    error_msg = error_data.get("error", {}).get(
                        "message", "Bad request")
                    self.emit(
                        "error", f"Google TTS request error: {error_msg}")
                except:
                    self.emit(
                        "error", "Google TTS bad request. Please check your configuration.")
            else:
                self.emit(
                    "error", f"Google TTS HTTP error: {e.response.status_code}")
            raise

    async def _stream_audio_chunks(self, audio_bytes: bytes) -> None:
        """Stream audio data in chunks to avoid beeps and ensure smooth playback"""
        chunk_size = 960
        audio_data = self._remove_wav_header(audio_bytes)

        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]

            if len(chunk) < chunk_size and len(chunk) > 0:
                padding_needed = chunk_size - len(chunk)
                chunk += b'\x00' * padding_needed

            if len(chunk) == chunk_size:
                if not self._first_chunk_sent and self._first_audio_callback:
                    self._first_chunk_sent = True
                    await self._first_audio_callback()

                asyncio.create_task(self.audio_track.add_new_bytes(chunk))
                await asyncio.sleep(0.001)

    def _remove_wav_header(self, audio_bytes: bytes) -> bytes:
        """Remove WAV header if present to get raw PCM data"""
        if audio_bytes.startswith(b"RIFF"):
            data_pos = audio_bytes.find(b"data")
            if data_pos != -1:
                return audio_bytes[data_pos + 8:]

        return audio_bytes

    async def aclose(self) -> None:
        if self._http_client:
            await self._http_client.aclose()
        await super().aclose()

    async def interrupt(self) -> None:
        if self.audio_track:
            self.audio_track.interrupt()
