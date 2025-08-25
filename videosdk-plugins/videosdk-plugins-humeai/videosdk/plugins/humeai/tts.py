from __future__ import annotations

from typing import Any, AsyncIterator, Literal, Optional
import os
import httpx
import asyncio
import json
import base64
import numpy as np

from videosdk.agents import TTS, segment_text

try:
    from scipy import signal

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

API_BASE_URL = "https://api.hume.ai/v0"


class HumeAITTS(TTS):
    def __init__(
        self,
        *,
        voice: Optional[str] = "Serene Assistant",
        speed: float = 1.0,
        api_key: Optional[str] = None,
        response_format: Literal["pcm", "mp3", "wav"] = "pcm",
        instant_mode: bool = True,
    ) -> None:
        super().__init__(sample_rate=24000, num_channels=1)

        self.voice = voice
        self.speed = speed
        self.response_format = response_format
        self.instant_mode = instant_mode
        self.audio_track = None
        self.loop = None
        self._first_chunk_sent = False

        if self.instant_mode and not self.voice:
            raise ValueError("Voice required for instant mode")

        self.api_key = api_key or os.getenv("HUMEAI_API_KEY")
        if not self.api_key:
            raise ValueError("HUMEAI_API_KEY required")

        self._session = httpx.AsyncClient(timeout=30.0)

    def reset_first_audio_tracking(self) -> None:
        """Reset the first audio tracking state for next TTS task"""
        self._first_chunk_sent = False

    async def synthesize(
        self,
        text: AsyncIterator[str] | str,
        voice_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        try:
            if not self.audio_track or not self.loop:
                self.emit("error", "Audio track not set")
                return

            if isinstance(text, AsyncIterator):
                async for segment in segment_text(text):
                    await self._synthesize_audio(segment, voice_id, **kwargs)
            else:
                await self._synthesize_audio(text, voice_id, **kwargs)

        except Exception as e:
            self.emit("error", f"Synthesis failed: {str(e)}")

    async def _synthesize_audio(self, text: str, voice_id: Optional[str] = None, **kwargs: Any) -> None:
        """Synthesize a single text segment"""
        if not text.strip():
            return

        utterance = {
            "text": text,
            "speed": kwargs.get("speed", self.speed)
        }
        if self.instant_mode:
            utterance["voice"] = {
                "name": voice_id or self.voice, "provider": "HUME_AI"}

        payload = {
            "utterances": [utterance],
            "format": {"type": self.response_format},
            "instant_mode": self.instant_mode,
            "strip_headers": False,
        }

        await self._stream_synthesis(payload)

    async def _stream_synthesis(self, payload: dict) -> None:
        """Stream audio from Hume AI API"""
        url = f"{API_BASE_URL}/tts/stream/json"
        headers = {
            "X-Hume-Api-Key": self.api_key,
            "Content-Type": "application/json"
        }

        try:
            async with self._session.stream(
                "POST", url, headers=headers, json=payload
            ) as response:
                response.raise_for_status()

                buffer = b""
                async for chunk in response.aiter_bytes():
                    lines = (buffer + chunk).split(b"\n")
                    buffer = lines.pop()

                    for line in lines:
                        if line.strip():
                            try:
                                data = json.loads(line)
                                if "audio" in data and data["audio"]:
                                    audio_bytes = base64.b64decode(
                                        data["audio"])
                                    if self.response_format == "wav":
                                        audio_bytes = self._remove_wav_header(
                                            audio_bytes)
                                    await self._stream_audio_chunks(audio_bytes)
                            except json.JSONDecodeError:
                                continue

                if buffer.strip():
                    try:
                        data = json.loads(buffer)
                        if "audio" in data and data["audio"]:
                            audio_bytes = base64.b64decode(data["audio"])
                            if self.response_format == "wav":
                                audio_bytes = self._remove_wav_header(
                                    audio_bytes)
                            await self._stream_audio_chunks(audio_bytes)
                    except json.JSONDecodeError:
                        pass

        except Exception as e:
            self.emit("error", f"Streaming failed: {str(e)}")

    async def _stream_audio_chunks(self, audio_bytes: bytes) -> None:
        """Stream audio with 48kHz->24kHz resampling"""
        if not audio_bytes:
            return

        try:
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            if len(audio_array) == 0:
                return

            resampled_audio = audio_array[::2]
            audio_bytes = resampled_audio.tobytes()

            chunk_size = 960
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i : i + chunk_size]
                if len(chunk) < chunk_size and len(chunk) > 0:
                    chunk += b"\x00" * (chunk_size - len(chunk))
                if chunk:
                    if not self._first_chunk_sent and self._first_audio_callback:
                        self._first_chunk_sent = True
                        await self._first_audio_callback()

                    asyncio.create_task(self.audio_track.add_new_bytes(chunk))
                    await asyncio.sleep(0.001)
        except Exception as e:
            self.emit("error", f"Audio streaming failed: {str(e)}")

    def _remove_wav_header(self, audio_bytes: bytes) -> bytes:
        """Remove WAV header if present"""
        if audio_bytes.startswith(b"RIFF"):
            data_pos = audio_bytes.find(b"data")
            if data_pos != -1:
                return audio_bytes[data_pos + 8 :]
        return audio_bytes

    async def aclose(self) -> None:
        """Cleanup resources"""
        if self._session:
            await self._session.aclose()
        await super().aclose()

    async def interrupt(self) -> None:
        """Interrupt TTS"""
        if self.audio_track:
            self.audio_track.interrupt()
