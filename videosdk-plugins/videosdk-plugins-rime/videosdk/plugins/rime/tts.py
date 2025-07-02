from __future__ import annotations

from typing import Any, AsyncIterator, Optional, Union
import os
import asyncio
import httpx

from videosdk.agents import TTS

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
        reduce_latency: bool = False,
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

        self.api_key = api_key or os.getenv("RIME_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "Rime AI API key required. Provide either:\n"
                "1. api_key parameter, OR\n"
                "2. RIME_API_KEY environment variable"
            )

        if model_id in KNOWN_SPEAKERS and speaker not in KNOWN_SPEAKERS[model_id]:
            available = ", ".join(KNOWN_SPEAKERS[model_id])
            print(f" Warning: Speaker '{speaker}' may not be available for model '{model_id}'. "
                  f"Known speakers: {available}")

        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=15.0, read=30.0, write=5.0, pool=5.0),
            follow_redirects=True,
        )

    async def synthesize(
        self,
        text: AsyncIterator[str] | str,
        **kwargs: Any,
    ) -> None:
        try:
            if isinstance(text, AsyncIterator):
                full_text = ""
                async for chunk in text:
                    full_text += chunk
            else:
                full_text = text

            if not self.audio_track or not self.loop:
                self.emit("error", "Audio track or loop not initialized")
                return

            await self._synthesize_audio(full_text)

        except Exception as e:
            self.emit("error", f"Rime TTS synthesis failed: {str(e)}")

    async def _synthesize_audio(self, text: str) -> None:
        """Synthesize text to speech using Rime AI streaming API"""
        try:
            if len(text) > 500:
                self.emit("error", f"Text exceeds 500 character limit. Got {len(text)} characters.")
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
                "POST",
                RIME_TTS_ENDPOINT,
                headers=headers,
                json=payload
            ) as response:
                response.raise_for_status()
                
                audio_data = b""
                async for chunk in response.aiter_bytes():
                    if chunk:
                        audio_data += chunk
                
                if not audio_data:
                    self.emit("error", "No audio data received from Rime TTS")
                    return
                
                await self._stream_audio_chunks(audio_data)
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                self.emit("error", "Rime TTS authentication failed. Please check your API key.")
            elif e.response.status_code == 400:
                error_text = e.response.text
                if "doesn't match list" in error_text:
                    available = ", ".join(KNOWN_SPEAKERS.get(self.model_id, []))
                    self.emit("error", f"Speaker '{self.speaker}' not available for model '{self.model_id}'. "
                              f"Try one of: {available}")
                else:
                    self.emit("error", f"Rime TTS bad request: {error_text}")
            else:
                self.emit("error", f"Rime TTS HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            self.emit("error", f"Rime TTS request failed: {str(e)}")
            raise

    async def _stream_audio_chunks(self, audio_bytes: bytes) -> None:
        """Stream audio data in chunks to avoid beeps and ensure smooth playback"""
        chunk_size = int(self.sampling_rate * RIME_CHANNELS * 2 * 20 / 1000)
        
        audio_data = self._remove_wav_header(audio_bytes)
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            
            if len(chunk) < chunk_size and len(chunk) > 0:
                padding_needed = chunk_size - len(chunk)
                chunk += b'\x00' * padding_needed
            
            if len(chunk) == chunk_size:
                self.loop.create_task(self.audio_track.add_new_bytes(chunk))
                await asyncio.sleep(0.001)

    def _remove_wav_header(self, audio_bytes: bytes) -> bytes:
        """Remove WAV header if present to get raw PCM data"""
        if audio_bytes.startswith(b'RIFF'):
            data_pos = audio_bytes.find(b'data')
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