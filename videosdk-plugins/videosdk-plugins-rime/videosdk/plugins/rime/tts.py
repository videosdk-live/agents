from __future__ import annotations

from typing import Any, AsyncIterator, Optional, Union
import os
import asyncio
import httpx

from videosdk.agents import TTS, segment_text

RIME_SAMPLE_RATE = 24000
RIME_CHANNELS = 1
RIME_TTS_ENDPOINT = "https://users.rime.ai/v1/rime-tts"

DEFAULT_MODEL = "mist"
DEFAULT_SPEAKER = "river"
DEFAULT_LANGUAGE = "eng"

KNOWN_SPEAKERS = {
    "mist": ["river", "storm", "brook", "ember", "iris", "pearl"],
    "mistv2": ["river", "storm", "brook", "ember", "iris", "pearl"]
}


class RimeTTS(TTS):
    def __init__(
        self,
        *,
        speaker: str = DEFAULT_SPEAKER,
        model_id: str = DEFAULT_MODEL,
        lang: str = DEFAULT_LANGUAGE,
        sampling_rate: int = RIME_SAMPLE_RATE,
        speed_alpha: float = 1.0,
        reduce_latency: bool = True,
        pause_between_brackets: bool = False,
        phonemize_between_brackets: bool = False,
        inline_speed_alpha: str | None = None,
        api_key: str | None = None,
    ) -> None:
        actual_sample_rate = sampling_rate
        super().__init__(sample_rate=actual_sample_rate, num_channels=RIME_CHANNELS)

        self.speaker = speaker
        self.model_id = model_id
        self.lang = lang
        self.sampling_rate = sampling_rate
        self.speed_alpha = speed_alpha
        self.reduce_latency = reduce_latency
        self.pause_between_brackets = pause_between_brackets
        self.phonemize_between_brackets = phonemize_between_brackets
        self.inline_speed_alpha = inline_speed_alpha
        self.audio_track = None
        self.loop = None
        self._first_chunk_sent = False

        self.api_key = api_key or os.getenv("RIME_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Rime AI API key required. Provide either:\n"
                "1. api_key parameter, OR\n"
                "2. RIME_API_KEY environment variable"
            )

        if model_id in KNOWN_SPEAKERS and speaker not in KNOWN_SPEAKERS[model_id]:
            available = ", ".join(KNOWN_SPEAKERS[model_id])
            print(
                f" Warning: Speaker '{speaker}' may not be available for model '{model_id}'. "
                f"Known speakers: {available}"
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
            if not self.audio_track or not self.loop:
                self.emit("error", "Audio track or loop not initialized")
                return

            if isinstance(text, AsyncIterator):
                async for segment in segment_text(text):
                    await self._synthesize_audio(segment)
            else:
                await self._synthesize_audio(text)

        except Exception as e:
            self.emit("error", f"Rime TTS synthesis failed: {str(e)}")

    async def _synthesize_audio(self, text: str) -> None:
        """Synthesize text to speech using Rime AI streaming API"""
        try:
            if len(text) > 500:
                self.emit(
                    "error", f"Text exceeds 500 character limit. Got {len(text)} characters.")
                return

            payload = {
                "speaker": self.speaker,
                "text": text,
                "modelId": self.model_id,
                "lang": self.lang,
                "samplingRate": self.sampling_rate,
                "speedAlpha": self.speed_alpha,
                "reduceLatency": self.reduce_latency,
                "pauseBetweenBrackets": self.pause_between_brackets,
                "phonemizeBetweenBrackets": self.phonemize_between_brackets,
            }

            if self.inline_speed_alpha:
                payload["inlineSpeedAlpha"] = self.inline_speed_alpha

            headers = {
                "Accept": "audio/pcm",
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            async with self._http_client.stream(
                "POST", RIME_TTS_ENDPOINT, headers=headers, json=payload
            ) as response:
                response.raise_for_status()

                async for chunk in response.aiter_bytes():
                    if chunk:
                        await self._process_audio_chunk(chunk)

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                self.emit(
                    "error", "Rime TTS authentication failed. Please check your API key.")
            elif e.response.status_code == 400:
                error_text = e.response.text
                if "doesn't match list" in error_text:
                    available = ", ".join(
                        KNOWN_SPEAKERS.get(self.model_id, []))
                    self.emit("error", f"Speaker '{self.speaker}' not available for model '{self.model_id}'. "
                              f"Try one of: {available}")
                else:
                    self.emit("error", f"Rime TTS bad request: {error_text}")
            else:
                self.emit(
                    "error", f"Rime TTS HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            self.emit("error", f"Rime TTS request failed: {str(e)}")
            raise

    async def _process_audio_chunk(self, audio_chunk: bytes) -> None:
        """Process individual audio chunks in real-time for minimal latency"""
        if not audio_chunk:
            return

        processed_chunk = self._remove_wav_header(audio_chunk)

        if not processed_chunk:
            return

        if not self._first_chunk_sent and self._first_audio_callback:
            self._first_chunk_sent = True
            await self._first_audio_callback()

        if self.audio_track and self.loop:
            asyncio.create_task(
                self.audio_track.add_new_bytes(processed_chunk))

    def _remove_wav_header(self, audio_bytes: bytes) -> bytes:
        """Remove WAV header if present to get raw PCM data"""
        if audio_bytes.startswith(b"RIFF"):
            data_pos = audio_bytes.find(b"data")
            if data_pos != -1:
                return audio_bytes[data_pos + 8:]

        return audio_bytes

    async def aclose(self) -> None:
        """Cleanup HTTP client resources"""
        if self._http_client:
            await self._http_client.aclose()
        await super().aclose()

    async def interrupt(self) -> None:
        """Interrupt the TTS audio stream"""
        if self.audio_track:
            self.audio_track.interrupt()
