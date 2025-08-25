from __future__ import annotations

from typing import Any, AsyncIterator, Optional, Union
import os
import asyncio
import base64
import httpx

from videosdk.agents import TTS, segment_text

SARVAMAI_SAMPLE_RATE = 22050
SARVAMAI_CHANNELS = 1
SARVAMAI_TTS_ENDPOINT = "https://api.sarvam.ai/text-to-speech"

DEFAULT_MODEL = "bulbul:v2"
DEFAULT_SPEAKER = "anushka"
DEFAULT_TARGET_LANGUAGE = "en-IN"


class SarvamAITTS(TTS):
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        speaker: str = DEFAULT_SPEAKER,
        target_language_code: str = DEFAULT_TARGET_LANGUAGE,
        pitch: float = 0.0,
        pace: float = 1.0,
        loudness: float = 1.2,
        enable_preprocessing: bool = True,
        api_key: str | None = None,
    ) -> None:
        super().__init__(
            sample_rate=SARVAMAI_SAMPLE_RATE, num_channels=SARVAMAI_CHANNELS
        )

        self.model = model
        self.speaker = speaker
        self.target_language_code = target_language_code
        self.pitch = pitch
        self.pace = pace
        self.loudness = loudness
        self.enable_preprocessing = enable_preprocessing
        self.audio_track = None
        self.loop = None

        self._first_chunk_sent = False

        self.api_key = api_key or os.getenv("SARVAMAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Sarvam AI API key required. Provide either:\n"
                "1. api_key parameter, OR\n"
                "2. SARVAMAI_API_KEY environment variable"
            )

        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=15.0, read=30.0, write=5.0, pool=5.0),
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
            if not self.audio_track or not self.loop:
                self.emit("error", "Audio track or loop not initialized")
                return

            if isinstance(text, AsyncIterator):
                async for segment in segment_text(text):
                    await self._synthesize_audio(segment)
            else:
                await self._synthesize_audio(text)

        except Exception as e:
            self.emit("error", f"Sarvam AI TTS synthesis failed: {str(e)}")

    async def _synthesize_audio(self, text: str) -> None:
        try:
            payload = {
                "inputs": [text],
                "target_language_code": self.target_language_code,
                "speaker": self.speaker,
                "pitch": self.pitch,
                "pace": self.pace,
                "loudness": self.loudness,
                "speech_sample_rate": SARVAMAI_SAMPLE_RATE,
                "enable_preprocessing": self.enable_preprocessing,
                "model": self.model,
            }

            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "api-subscription-key": self.api_key,
            }

            response = await self._http_client.post(
                SARVAMAI_TTS_ENDPOINT, headers=headers, json=payload
            )
            response.raise_for_status()

            response_data = response.json()
            if "audios" not in response_data or not response_data["audios"]:
                self.emit("error", "No audio data found in response from Sarvam AI")
                return

            audio_content = response_data["audios"][0]
            if not audio_content:
                self.emit("error", "No audio content received from Sarvam AI")
                return

            audio_bytes = base64.b64decode(audio_content)

            if not audio_bytes:
                self.emit("error", "Decoded audio bytes are empty")
                return

            await self._stream_audio_chunks(audio_bytes)

        except httpx.HTTPStatusError as e:
            self.emit(
                "error",
                f"Sarvam AI TTS HTTP error: {e.response.status_code} - {e.response.text}",
            )
            raise

    async def _stream_audio_chunks(self, audio_bytes: bytes) -> None:
        chunk_size = int(SARVAMAI_SAMPLE_RATE * SARVAMAI_CHANNELS * 2 * 20 / 1000)

        audio_data = self._remove_wav_header(audio_bytes)

        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i : i + chunk_size]

            if len(chunk) < chunk_size and len(chunk) > 0:
                padding_needed = chunk_size - len(chunk)
                chunk += b"\x00" * padding_needed

            if len(chunk) == chunk_size:
                if not self._first_chunk_sent and self._first_audio_callback:
                    self._first_chunk_sent = True
                    asyncio.create_task(self._first_audio_callback())

                asyncio.create_task(self.audio_track.add_new_bytes(chunk))
                await asyncio.sleep(0.001)

    def _remove_wav_header(self, audio_bytes: bytes) -> bytes:
        if audio_bytes.startswith(b"RIFF"):
            data_pos = audio_bytes.find(b"data")
            if data_pos != -1:
                return audio_bytes[data_pos + 8 :]

        return audio_bytes

    async def aclose(self) -> None:
        if self._http_client:
            await self._http_client.aclose()
        await super().aclose()

    async def interrupt(self) -> None:
        if self.audio_track:
            self.audio_track.interrupt()
