from __future__ import annotations

import asyncio
import os
import wave
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

import httpx
import numpy as np

from videosdk.agents import TTS, segment_text
from scipy import signal


CAMB_AI_TTS_ENDPOINT = "https://client.camb.ai/apis/tts-stream"
CAMB_AI_SAMPLE_RATE = 48000
OUTPUT_SAMPLE_RATE = 24000
RESAMPLE_RATIO = CAMB_AI_SAMPLE_RATE // OUTPUT_SAMPLE_RATE 

CAMB_AI_CHANNELS = 1

DEFAULT_SPEECH_MODEL = "mars-pro"
DEFAULT_VOICE_ID = 147320
DEFAULT_LANGUAGE = "en-us"


@dataclass
class InferenceOptions:
    """Model sampling controls — trade off stability, variation, and latency."""
    stability: float | None = None
    temperature: float | None = None
    inference_steps: int | None = None
    speaker_similarity: float | None = None
    localize_speaker_weight: float | None = None
    acoustic_quality_boost: bool | None = None

    def to_dict(self) -> dict:
        d = {}
        if self.stability is not None:
            d["stability"] = self.stability
        if self.temperature is not None:
            d["temperature"] = self.temperature
        if self.inference_steps is not None:
            d["inference_steps"] = self.inference_steps
        if self.speaker_similarity is not None:
            d["speaker_similarity"] = self.speaker_similarity
        if self.localize_speaker_weight is not None:
            d["localize_speaker_weight"] = self.localize_speaker_weight
        if self.acoustic_quality_boost is not None:
            d["acoustic_quality_boost"] = self.acoustic_quality_boost
        return d


@dataclass
class VoiceSettings:
    """Voice behaviour preferences."""
    enhance_reference_audio_quality: bool = False
    maintain_source_accent: bool = False

    def to_dict(self) -> dict:
        return {
            "enhance_reference_audio_quality": self.enhance_reference_audio_quality,
            "maintain_source_accent": self.maintain_source_accent,
        }


@dataclass
class OutputConfiguration:
    """Audio output format & pacing options."""
    format: str = "wav"
    duration: float | None = None
    sample_rate: int | None = None

    def to_dict(self) -> dict:
        d: dict = {"format": self.format}
        if self.duration is not None:
            d["duration"] = self.duration
        if self.sample_rate is not None:
            d["sample_rate"] = self.sample_rate
        return d


class CambAITTS(TTS):
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        speech_model: str = DEFAULT_SPEECH_MODEL,
        voice_id: int = DEFAULT_VOICE_ID,
        language: str = DEFAULT_LANGUAGE,
        user_instructions: Optional[str] = None,
        enhance_named_entities_pronunciation: bool = False,
        output_configuration: Optional[OutputConfiguration] = None,
        voice_settings: Optional[VoiceSettings] = None,
        inference_options: Optional[InferenceOptions] = None,
    ) -> None:
        """
        CambAI Text-to-Speech plugin for VideoSDK agents.

        Args:
            api_key: CambAI API key. Falls back to CAMBAI_API_KEY env var.
            speech_model: "mars-pro" | "mars-flash" | "mars-instruct".
            voice_id: Numeric voice profile ID from /list-voices.
            language: BCP-47 locale string (e.g. "en-us").
            user_instructions: Style/tone guidance (only for mars-instruct).
            enhance_named_entities_pronunciation: Improve brand/name pronunciation.
            output_configuration: Format, duration, and sample-rate settings.
            voice_settings: Accent & reference-audio quality preferences.
            inference_options: Stability, temperature, steps, similarity controls.
        """
        if user_instructions is not None and speech_model != "mars-instruct":
            raise ValueError(
                "user_instructions is only supported when speech_model='mars-instruct'."
            )

        # Tell the base class we're delivering 24 kHz audio (post-resample)
        super().__init__(sample_rate=OUTPUT_SAMPLE_RATE, num_channels=CAMB_AI_CHANNELS)

        self.speech_model = speech_model
        self.voice_id = voice_id
        self.language = language
        self.user_instructions = user_instructions
        self.enhance_named_entities_pronunciation = enhance_named_entities_pronunciation
        self.output_configuration = output_configuration or OutputConfiguration()
        self.voice_settings = voice_settings or VoiceSettings()
        self.inference_options = inference_options or InferenceOptions()

        self.audio_track = None
        self.loop = None
        self._first_chunk_sent = False

        self.api_key = api_key or os.getenv("CAMBAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "CambAI API key must be provided via the api_key parameter "
                "or the CAMBAI_API_KEY environment variable."
            )

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=15.0, read=60.0, write=10.0, pool=5.0),
            follow_redirects=True,
            limits=httpx.Limits(
                max_connections=50,
                max_keepalive_connections=50,
                keepalive_expiry=120,
            ),
        )

    @staticmethod
    def _resample_pcm(audio_bytes: bytes) -> bytes:
        """ Resample 48 kHz mono 16-bit PCM to 24 kHz mono 16-bit PCM. """
        if not audio_bytes:
            return b""

        # Decode raw bytes to int16 samples
        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        cutoff = 12000.0 / CAMB_AI_SAMPLE_RATE 
        num_taps = 65 
        n = np.arange(num_taps)
        mid = num_taps // 2

        # Ideal sinc kernel
        with np.errstate(invalid="ignore"):
            h = np.where(
                n == mid,
                2.0 * cutoff,
                np.sin(2.0 * np.pi * cutoff * (n - mid)) / (np.pi * (n - mid)),
            )
        h *= np.hamming(num_taps)
        h /= h.sum()
        filtered = np.convolve(samples, h, mode="same")
        decimated = filtered[::RESAMPLE_RATIO]
        resampled = np.clip(decimated, -32768, 32767).astype(np.int16)
        return resampled.tobytes()
    
    def reset_first_audio_tracking(self) -> None:
        """Reset first-audio tracking state before a new TTS task."""
        self._first_chunk_sent = False

    async def synthesize(
        self,
        text: AsyncIterator[str] | str,
        voice_id: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        try:
            if isinstance(text, AsyncIterator):
                async for segment in segment_text(text):
                    await self._synthesize_segment(segment, voice_id)
            else:
                await self._synthesize_segment(text, voice_id)
        except Exception as e:
            self.emit("error", f"TTS synthesis failed: {e}")

    async def send_text(self, text: str) -> None:
        """Synthesize a plain text string directly"""
        await self._synthesize_segment(text)

    async def _synthesize_segment(
        self, text: str, voice_id: Optional[int] = None
    ) -> None:
        """Call the CambAI streaming TTS API for a single text segment."""
        if not text.strip():
            return

        if len(text) > 3000:
            for chunk in [text[i:i + 3000] for i in range(0, len(text), 3000)]:
                await self._synthesize_segment(chunk, voice_id)
            return

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        payload: dict = {
            "text": text,
            "language": self.language,
            "voice_id": voice_id if voice_id is not None else self.voice_id,
            "speech_model": self.speech_model,
            "enhance_named_entities_pronunciation": self.enhance_named_entities_pronunciation,
            "output_configuration": self.output_configuration.to_dict(),
            "voice_settings": self.voice_settings.to_dict(),
            "inference_options": self.inference_options.to_dict(),
        }

        if self.user_instructions is not None:
            payload["user_instructions"] = self.user_instructions

        try:
            async with self._client.stream(
                "POST",
                CAMB_AI_TTS_ENDPOINT,
                headers=headers,
                json=payload,
            ) as response:
                response.raise_for_status()

                audio_data = b""
                async for chunk in response.aiter_bytes():
                    if chunk:
                        audio_data += chunk

            if self.output_configuration.format == "wav":
                pcm_data = self._extract_pcm_from_wav(audio_data)
                resampled_pcm = self._resample_pcm(pcm_data)
                await self._stream_audio_chunks(resampled_pcm)
            else:
                self.emit(
                    "error",
                    f"Format '{self.output_configuration.format}' requires decoding, "
                    "which is not yet implemented.",
                )

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 401:
                self.emit("error", "CambAI authentication failed — check your API key.")
            elif status == 400:
                try:
                    msg = e.response.json().get("detail", e.response.text)
                except Exception:
                    msg = e.response.text
                self.emit("error", f"CambAI bad request (400): {msg}")
            elif status == 422:
                try:
                    msg = e.response.json()
                except Exception:
                    msg = e.response.text
                self.emit("error", f"CambAI validation error (422): {msg}")
            elif status == 429:
                self.emit("error", "CambAI rate limit exceeded — please retry later.")
            else:
                self.emit("error", f"CambAI HTTP {status}: {e.response.text}")
        except httpx.TimeoutException as e:
            self.emit("error", f"CambAI request timed out: {e}")
        except Exception as e:
            self.emit("error", f"CambAI TTS API call failed: {e}")

    async def _stream_audio_chunks(self, audio_bytes: bytes) -> None:
        """Push 24 kHz PCM audio to the audio track in ~20 ms frames."""
        if not audio_bytes:
            return

        if not self.audio_track or not self.loop:
            self.emit("error", "audio_track or loop not set before streaming.")
            return

        if not self._first_chunk_sent:
            self._first_chunk_sent = True
            if hasattr(self, "_first_audio_callback") and self._first_audio_callback:
                asyncio.create_task(self._first_audio_callback())

        # 20 ms frame at 24 kHz mono 16-bit = 24000 × 1 × 2 × 0.02 = 960 bytes
        chunk_size = int(OUTPUT_SAMPLE_RATE * CAMB_AI_CHANNELS * 2 * 20 / 1000)

        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i: i + chunk_size]
            if 0 < len(chunk) < chunk_size:
                chunk += b"\x00" * (chunk_size - len(chunk))

            if chunk:
                asyncio.create_task(self.audio_track.add_new_bytes(chunk))
                await asyncio.sleep(0.001)


    @staticmethod
    def _extract_pcm_from_wav(self, wav_data: bytes) -> bytes:
        """Extract PCM data from WAV file format"""
        if len(wav_data) < 44:
            return wav_data

        if wav_data[:4] != b"RIFF":
            return wav_data

        data_pos = wav_data.find(b"data")
        if data_pos == -1:
            return wav_data

        return wav_data[data_pos + 8:]

    async def aclose(self) -> None:
        """Cleanup resources"""
        await self._client.aclose()
        await super().aclose()

    async def interrupt(self) -> None:
        if self.audio_track:
            self.audio_track.interrupt()
        self.reset_first_audio_tracking()