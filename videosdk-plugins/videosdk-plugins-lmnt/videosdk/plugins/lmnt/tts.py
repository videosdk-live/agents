from __future__ import annotations

from typing import Any, AsyncIterator, Literal, Optional, Union
import httpx
import os
import asyncio

from videosdk.agents import TTS, segment_text

LMNT_API_BASE_URL = "https://api.lmnt.com"
LMNT_SAMPLE_RATE = 24000
LMNT_CHANNELS = 1

DEFAULT_MODEL = "blizzard"
DEFAULT_VOICE = "ava"
DEFAULT_LANGUAGE = "auto"
DEFAULT_FORMAT = "wav"

_LanguageCode = Union[
    Literal["auto", "de", "en", "es", "fr", "hi", "id", "it", "ja",
            "ko", "nl", "pl", "pt", "ru", "sv", "th", "tr", "uk", "vi", "zh"],
    str
]
_FormatType = Union[Literal["aac", "mp3", "mulaw", "raw", "wav"], str]
_SampleRate = Union[Literal[8000, 16000, 24000], int]


class LMNTTTS(TTS):
    def __init__(
        self,
        *,
        voice: str = DEFAULT_VOICE,
        model: str = DEFAULT_MODEL,
        language: _LanguageCode = DEFAULT_LANGUAGE,
        format: _FormatType = DEFAULT_FORMAT,
        sample_rate: _SampleRate = LMNT_SAMPLE_RATE,
        seed: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 0.8,
        api_key: Optional[str] = None,
        base_url: str = LMNT_API_BASE_URL,
    ) -> None:
        super().__init__(sample_rate=sample_rate, num_channels=LMNT_CHANNELS)

        self.voice = voice
        self.model = model
        self.language = language
        self.format = format
        self.output_sample_rate = sample_rate
        self.seed = seed
        self.temperature = temperature
        self.top_p = top_p
        self.base_url = base_url
        self.audio_track = None
        self.loop = None
        self._first_chunk_sent = False
        self._interrupted = False

        self.api_key = api_key or os.getenv("LMNT_API_KEY")
        if not self.api_key:
            raise ValueError(
                "LMNT API key must be provided either through api_key parameter "
                "or LMNT_API_KEY environment variable"
            )

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=15.0, read=30.0,
                                  write=5.0, pool=5.0),
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
        Convert text to speech using LMNT's TTS API and stream to audio track

        Args:
            text: Text to convert to speech
            voice_id: Optional voice override (uses voice from __init__ if not provided)
            **kwargs: Additional provider-specific arguments
        """
        try:
            if not self.audio_track or not self.loop:
                self.emit("error", "Audio track or event loop not set")
                return

            self._interrupted = False

            if isinstance(text, AsyncIterator):
                async for segment in segment_text(text):
                    if self._interrupted:
                        break
                    await self._synthesize_segment(segment, voice_id, **kwargs)
            else:
                if not self._interrupted:
                    await self._synthesize_segment(text, voice_id, **kwargs)

        except Exception as e:
            self.emit("error", f"TTS synthesis failed: {str(e)}")

    async def _synthesize_segment(self, text: str, voice_id: Optional[str] = None, **kwargs: Any) -> None:
        """Synthesize a single text segment"""
        if not text.strip() or self._interrupted:
            return

        target_voice = voice_id or self.voice

        payload = {
            "voice": target_voice,
            "text": text,
            "model": kwargs.get("model", self.model),
            "language": kwargs.get("language", self.language),
            "format": kwargs.get("format", self.format),
            "sample_rate": kwargs.get("sample_rate", self.output_sample_rate),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
        }

        seed = kwargs.get("seed", self.seed)
        if seed is not None:
            payload["seed"] = seed

        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}/v1/ai/speech/bytes"

        async with self._client.stream(
            "POST",
            url,
            headers=headers,
            json=payload
        ) as response:
            if response.status_code == 400:
                error_data = await response.aread()
                try:
                    import json
                    error_json = json.loads(error_data.decode())
                    error_msg = error_json.get("error", "Bad request")
                except:
                    error_msg = "Bad request"
                self.emit("error", f"LMNT API error: {error_msg}")
                return
            elif response.status_code == 401:
                self.emit(
                    "error", "LMNT API authentication failed. Please check your API key.")
                return
            elif response.status_code != 200:
                self.emit(
                    "error", f"LMNT API error: HTTP {response.status_code}")
                return

            header_processed = False
            accumulated_data = b""

            async for chunk in response.aiter_bytes():
                if self._interrupted:
                    break
                if chunk:
                    accumulated_data += chunk

                    if not header_processed and len(accumulated_data) >= 44:
                        if accumulated_data.startswith(b'RIFF'):
                            data_pos = accumulated_data.find(b'data')
                            if data_pos != -1:
                                accumulated_data = accumulated_data[data_pos + 8:]
                        header_processed = True

                    if header_processed:
                        chunk_size = int(
                            self.output_sample_rate * LMNT_CHANNELS * 2 * 20 / 1000)  # 20ms chunks
                        while len(accumulated_data) >= chunk_size:
                            audio_chunk = accumulated_data[:chunk_size]
                            accumulated_data = accumulated_data[chunk_size:]

                            if not self._first_chunk_sent and self._first_audio_callback:
                                self._first_chunk_sent = True
                                await self._first_audio_callback()

                            self.loop.create_task(
                                self.audio_track.add_new_bytes(audio_chunk))
                            await asyncio.sleep(0.01)

            if accumulated_data and header_processed:
                chunk_size = int(self.output_sample_rate *
                                 LMNT_CHANNELS * 2 * 20 / 1000)
                if len(accumulated_data) < chunk_size:
                    accumulated_data += b'\x00' * \
                        (chunk_size - len(accumulated_data))

                if not self._first_chunk_sent and self._first_audio_callback:
                    self._first_chunk_sent = True
                    await self._first_audio_callback()

                self.loop.create_task(
                    self.audio_track.add_new_bytes(accumulated_data))

    async def aclose(self) -> None:
        """Cleanup resources"""
        await self._client.aclose()
        await super().aclose()

    async def interrupt(self) -> None:
        """Interrupt the TTS process"""
        self._interrupted = True
        if self.audio_track:
            self.audio_track.interrupt()
