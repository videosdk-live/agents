from __future__ import annotations

from typing import Any, AsyncIterator, Literal, Optional
import httpx
import os
import asyncio

from videosdk.agents import TTS, segment_text

GROQ_TTS_SAMPLE_RATE = 24000
GROQ_TTS_CHANNELS = 1

DEFAULT_MODEL = "playai-tts"
DEFAULT_VOICE = "Fritz-PlayAI"
GROQ_TTS_ENDPOINT = "https://api.groq.com/openai/v1/audio/speech"

SAMPLE_RATE_MAP = {
    8000: 8000,
    16000: 16000,
    22050: 22050,
    24000: 24000,
    32000: 32000,
    44100: 44100,
    48000: 48000,
}


class GroqTTS(TTS):
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        voice: str = DEFAULT_VOICE,
        speed: float = 1.0,
        api_key: str | None = None,
        response_format: Literal["flac", "mp3", "mulaw", "ogg", "wav"] = "wav",
        sample_rate: int = 24000,
    ) -> None:
        if sample_rate not in SAMPLE_RATE_MAP:
            raise ValueError(
                f"Invalid sample rate: {sample_rate}. Must be one of: {list(SAMPLE_RATE_MAP.keys())}"
            )

        if not 0.5 <= speed <= 5.0:
            raise ValueError(f"Speed must be between 0.5 and 5.0, got {speed}")

        super().__init__(sample_rate=sample_rate, num_channels=GROQ_TTS_CHANNELS)

        self.model = model
        self.voice = voice
        self.speed = speed
        self.audio_track = None
        self.loop = None
        self.response_format = response_format
        self._groq_sample_rate = sample_rate
        self._first_chunk_sent = False

        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key must be provided either through api_key parameter or GROQ_API_KEY environment variable"
            )

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=15.0, read=30.0, write=5.0, pool=5.0),
            follow_redirects=True,
            limits=httpx.Limits(
                max_connections=50,
                max_keepalive_connections=50,
                keepalive_expiry=120,
            ),
        )

    def reset_first_audio_tracking(self) -> None:
        """Reset the first audio tracking state for next TTS task"""
        self._first_chunk_sent = False

    async def synthesize(
        self,
        text: AsyncIterator[str] | str,
        voice_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Convert text to speech using Groq's TTS API and stream to audio track

        Args:
            text: Text to convert to speech
            voice_id: Optional voice override
            **kwargs: Additional provider-specific arguments
        """
        try:
            if isinstance(text, AsyncIterator):
                async for segment in segment_text(text):
                    await self._synthesize_audio(segment, voice_id)
            else:
                await self._synthesize_audio(text, voice_id)

            if not self.audio_track or not self.loop:
                self.emit("error", "Audio track or event loop not set")
                return

        except Exception as e:
            self.emit("error", f"TTS synthesis failed: {str(e)}")

    async def _synthesize_audio(
        self, text: str, voice_id: Optional[str] = None
    ) -> None:
        """Call Groq API to synthesize audio"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.model,
                "input": text,
                "voice": voice_id or self.voice,
                "response_format": self.response_format,
                "sample_rate": self._groq_sample_rate,
                "speed": self.speed,
            }

            async with self._client.stream(
                "POST",
                GROQ_TTS_ENDPOINT,
                headers=headers,
                json=payload,
            ) as response:
                response.raise_for_status()

                if self.response_format == "wav":
                    audio_data = b""
                    async for chunk in response.aiter_bytes():
                        audio_data += chunk

                    pcm_data = self._extract_pcm_from_wav(audio_data)
                    await self._stream_audio_chunks(pcm_data)
                else:
                    self.emit(
                        "error",
                        f"Format {self.response_format} requires decoding, which is not implemented yet",
                    )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                self.emit(
                    "error",
                    "Groq API authentication failed. Please check your API key.",
                )
            elif e.response.status_code == 400:
                try:
                    error_data = e.response.json()
                    error_msg = error_data.get("error", {}).get(
                        "message", "Bad request"
                    )
                    self.emit("error", f"Groq TTS request error: {error_msg}")
                except:
                    self.emit("error", f"Groq TTS bad request: {e.response.text}")
            elif e.response.status_code == 429:
                self.emit(
                    "error", "Groq TTS rate limit exceeded. Please try again later."
                )
            else:
                self.emit(
                    "error",
                    f"Groq TTS HTTP error {e.response.status_code}: {e.response.text}",
                )
        except Exception as e:
            self.emit("error", f"Groq TTS API call failed: {str(e)}")

    async def _stream_audio_chunks(self, audio_bytes: bytes) -> None:
        """Stream audio data in chunks at 24kHz"""
        chunk_size = int(24000 * GROQ_TTS_CHANNELS * 2 * 20 / 1000)

        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i : i + chunk_size]

            if len(chunk) < chunk_size and len(chunk) > 0:
                padding_needed = chunk_size - len(chunk)
                chunk += b"\x00" * padding_needed

            if chunk:
                if not self._first_chunk_sent and self._first_audio_callback:
                    self._first_chunk_sent = True
                    await self._first_audio_callback()

                asyncio.create_task(self.audio_track.add_new_bytes(chunk))
                await asyncio.sleep(0.001)

    def _extract_pcm_from_wav(self, wav_data: bytes) -> bytes:
        """Extract PCM data from WAV file format"""
        if len(wav_data) < 44:
            return wav_data

        if wav_data[:4] != b"RIFF":
            return wav_data

        data_pos = wav_data.find(b"data")
        if data_pos == -1:
            return wav_data

        return wav_data[data_pos + 8 :]

    async def aclose(self) -> None:
        """Cleanup resources"""
        await self._client.aclose()
        await super().aclose()

    async def interrupt(self) -> None:
        """Interrupt the TTS process"""
        if self.audio_track:
            self.audio_track.interrupt()
