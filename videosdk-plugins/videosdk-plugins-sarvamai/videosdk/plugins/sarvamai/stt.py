from __future__ import annotations

import asyncio
import io
import os
import time
import wave
from typing import Any, Optional

import httpx
import numpy as np

from videosdk.agents import STT, STTResponse, SpeechData, SpeechEventType, global_event_emitter

try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

SARVAM_STT_API_URL = "https://api.sarvam.ai/speech-to-text"
DEFAULT_MODEL = "saarika:v2"

class SarvamAISTT(STT):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        language: str = "en-IN",
        input_sample_rate: int = 48000,
        output_sample_rate: int = 16000,
        silence_threshold: float = 0.01,
        silence_duration: float = 0.8,
    ) -> None:
        super().__init__()
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is not installed. Please install it with 'pip install scipy'")

        self.api_key = api_key or os.getenv("SARVAMAI_API_KEY")
        if not self.api_key:
            raise ValueError("Sarvam AI API key must be provided either through api_key parameter or SARVAMAI_API_KEY environment variable")

        self.model = model
        self.language = language
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.silence_threshold_bytes = int(silence_threshold * 32767)
        self.silence_duration_frames = int(silence_duration * self.input_sample_rate)

        self._http_client = httpx.AsyncClient(timeout=httpx.Timeout(connect=15.0, read=30.0, write=5.0, pool=5.0))
        self._audio_buffer = bytearray()
        self._is_speaking = False
        self._silence_frames = 0
        self._lock = asyncio.Lock()

    async def process_audio(self, audio_frames: bytes, **kwargs: Any) -> None:
        async with self._lock:
            is_silent_chunk = self._is_silent(audio_frames)
            
            if not is_silent_chunk:
                if not self._is_speaking:
                    self._is_speaking = True
                    global_event_emitter.emit("speech_started")
                self._audio_buffer.extend(audio_frames)
                self._silence_frames = 0
            else:
                if self._is_speaking:
                    self._silence_frames += len(audio_frames) // 4 
                    if self._silence_frames > self.silence_duration_frames:
                        global_event_emitter.emit("speech_stopped")
                        asyncio.create_task(self._transcribe_buffer())
                        self._is_speaking = False
                        self._silence_frames = 0

    def _is_silent(self, audio_chunk: bytes) -> bool:
        """Simple VAD: check if the max amplitude is below a threshold."""
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
        return np.max(np.abs(audio_data)) < self.silence_threshold_bytes

    async def _transcribe_buffer(self):
        async with self._lock:
            if not self._audio_buffer:
                return
            audio_to_send = self._audio_buffer
            self._audio_buffer = bytearray()
        
        try:
            resampled_audio = self._resample_audio(audio_to_send)
            wav_audio = self._create_wav_in_memory(resampled_audio)

            headers = {"api-subscription-key": self.api_key}
            data = {"model": self.model, "language_code": self.language}
            files = {'file': ('audio.wav', wav_audio, 'audio/wav')}

            response = await self._http_client.post(SARVAM_STT_API_URL, headers=headers, data=data, files=files)
            response.raise_for_status()

            response_data = response.json()
            transcript = response_data.get("transcript", "")
            
            if transcript and self._transcript_callback:
                event = STTResponse(
                    event_type=SpeechEventType.FINAL,
                    data=SpeechData(text=transcript, language=self.language, confidence=1.0)
                )
                await self._transcript_callback(event)
        except httpx.HTTPStatusError as e:
            self.emit("error", f"Sarvam STT API error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            self.emit("error", f"Error during transcription: {e}")

    def _resample_audio(self, audio_bytes: bytes) -> bytes:
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
        resampled_data = signal.resample(audio_data, int(len(audio_data) * self.output_sample_rate / self.input_sample_rate))
        return resampled_data.astype(np.int16).tobytes()

    def _create_wav_in_memory(self, pcm_data: bytes) -> io.BytesIO:
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(self.output_sample_rate) 
            wf.writeframes(pcm_data)
        wav_buffer.seek(0)
        return wav_buffer

    async def aclose(self) -> None:
        if self._is_speaking and self._audio_buffer:
            await self._transcribe_buffer()
            await asyncio.sleep(1)

        if self._http_client:
            await self._http_client.aclose()
