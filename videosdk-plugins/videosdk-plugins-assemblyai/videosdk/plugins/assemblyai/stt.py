from __future__ import annotations

import asyncio
import io
import os
import wave
from typing import Any

import aiohttp
import numpy as np
from videosdk.agents import STT, STTResponse, SpeechData, SpeechEventType, global_event_emitter

try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

ASSEMBLYAI_API_URL = "https://api.assemblyai.com/v2"

class AssemblyAISTT(STT):
    """
    VideoSDK Agent Framework STT plugin for AssemblyAI.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        language_code: str = "en_us",
        input_sample_rate: int = 48000,
        target_sample_rate: int = 16000,
        silence_threshold: float = 0.015,
        silence_duration: float = 0.8,
    ) -> None:
        super().__init__()
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is not installed. Please install it with 'pip install scipy'")

        self.api_key = api_key or os.getenv("ASSEMBLYAI_API_KEY")
        if not self.api_key:
            raise ValueError("AssemblyAI API key must be provided either through the 'api_key' parameter or the 'ASSEMBLYAI_API_KEY' environment variable.")

        self.language_code = language_code
        self.input_sample_rate = input_sample_rate
        self.target_sample_rate = target_sample_rate
        self.silence_threshold_bytes = int(silence_threshold * 32767)
        self.silence_duration_frames = int(silence_duration * self.input_sample_rate)

        self._session = aiohttp.ClientSession(headers={"Authorization": self.api_key})
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
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
        return np.max(np.abs(audio_data)) < self.silence_threshold_bytes

    async def _transcribe_buffer(self):
        async with self._lock:
            if not self._audio_buffer:
                return
            audio_to_send = self._audio_buffer
            self._audio_buffer = bytearray()
        
        try:
            resampled_audio_bytes = self._resample_audio(audio_to_send)
            wav_audio = self._create_wav_in_memory(resampled_audio_bytes)

            upload_url = f"{ASSEMBLYAI_API_URL}/upload"
            async with self._session.post(upload_url, data=wav_audio) as response:
                response.raise_for_status()
                upload_data = await response.json()
                audio_url = upload_data["upload_url"]

            transcript_url = f"{ASSEMBLYAI_API_URL}/transcript"
            payload = {"audio_url": audio_url, "language_code": self.language_code}
            async with self._session.post(transcript_url, json=payload) as response:
                response.raise_for_status()
                transcript_data = await response.json()
                transcript_id = transcript_data["id"]

            poll_url = f"{ASSEMBLYAI_API_URL}/transcript/{transcript_id}"
            while True:
                await asyncio.sleep(1)
                async with self._session.get(poll_url) as response:
                    response.raise_for_status()
                    result = await response.json()
                    if result["status"] == "completed":
                        if result.get("text") and self._transcript_callback:
                            event = STTResponse(
                                event_type=SpeechEventType.FINAL,
                                data=SpeechData(text=result["text"], language=self.language_code, confidence=result.get("confidence", 1.0))
                            )
                            await self._transcript_callback(event)
                        break
                    elif result["status"] == "error":
                        raise Exception(f"AssemblyAI transcription failed: {result.get('error')}")

        except Exception as e:
            print(f"!!! ASSEMBLYAI PLUGIN FATAL ERROR: {e} ({type(e).__name__}) !!!")
            self.emit("error", f"AssemblyAI transcription error: {e}")

    def _resample_audio(self, audio_bytes: bytes) -> bytes:
        raw_audio = np.frombuffer(audio_bytes, dtype=np.int16)
        if raw_audio.size == 0: return b''
        stereo_audio = raw_audio.reshape(-1, 2)
        mono_audio = stereo_audio.astype(np.float32).mean(axis=1)
        resampled_data = signal.resample(mono_audio, int(len(mono_audio) * self.target_sample_rate / self.input_sample_rate))
        return resampled_data.astype(np.int16).tobytes()
        
    def _create_wav_in_memory(self, pcm_data: bytes) -> io.BytesIO:
        """Creates a WAV file in memory from raw PCM data."""
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.target_sample_rate)
            wf.writeframes(pcm_data)
        wav_buffer.seek(0)
        return wav_buffer

    async def aclose(self) -> None:
        if self._is_speaking and self._audio_buffer:
            await self._transcribe_buffer()
            await asyncio.sleep(1)

        if self._session and not self._session.closed:
            await self._session.close()