from __future__ import annotations

from typing import Any, AsyncIterator, Optional, Literal
import os
import asyncio
import aiofiles
import tempfile
from videosdk.agents import TTS

SMALLESTAI_SAMPLE_RATE = 24000
SMALLESTAI_CHANNELS = 1
DEFAULT_MODEL = "lightning"
DEFAULT_VOICE_ID = "emily"


class SmallestAITTS(TTS):
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        voice_id: str = DEFAULT_VOICE_ID,
        speed: float = 1.0,
        consistency: float = 0.5,
        similarity: float = 0.0,
        enhancement: bool = False,
        add_wav_header: bool = False,
        api_key: str | None = None,
    ) -> None:
        super().__init__(sample_rate=SMALLESTAI_SAMPLE_RATE, num_channels=SMALLESTAI_CHANNELS)

        self.model = model
        self.voice_id = voice_id
        self.speed = speed
        self.consistency = consistency
        self.similarity = similarity
        self.enhancement = enhancement
        self.add_wav_header = add_wav_header

        self.audio_track = None
        self.loop = None

        self.api_key = api_key or os.getenv("SMALLEST_API_KEY")
        if not self.api_key:
            raise ValueError(
                "SmallestAI API key required. Provide either:\n"
                "1. api_key parameter, OR\n"
                "2. SMALLEST_API_KEY environment variable"
            )

        
        try:
            from smallestai.waves import AsyncWavesClient
        except ImportError:
            raise ImportError(
                "SmallestAI package not found. Install it with: pip install smallestai"
            )

        self._client = AsyncWavesClient(
            api_key=self.api_key,
            model=self.model,
            sample_rate=SMALLESTAI_SAMPLE_RATE,
            voice_id=self.voice_id,
            speed=self.speed,
            consistency=self.consistency,
            similarity=self.similarity,
            enhancement=self.enhancement,
            add_wav_header=self.add_wav_header,
        )

    async def synthesize(
        self,
        text: AsyncIterator[str] | str,
        voice_id: Optional[str] = None,
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
                self.emit("error", "Audio track or event loop not set")
                return

           
            target_voice = voice_id or self.voice_id

            await self._synthesize_audio(full_text, target_voice, **kwargs)

        except Exception as e:
            self.emit("error", f"SmallestAI TTS synthesis failed: {str(e)}")

    async def _synthesize_audio(self, text: str, voice_id: str, **kwargs: Any) -> None:
        """Synthesize text to speech using SmallestAI API"""
        try:
            
            synthesis_kwargs = {
                "voice_id": voice_id,
                "speed": kwargs.get("speed", self.speed),
                "consistency": kwargs.get("consistency", self.consistency),
                "similarity": kwargs.get("similarity", self.similarity),
                "enhancement": kwargs.get("enhancement", self.enhancement),
                "sample_rate": kwargs.get("sample_rate", SMALLESTAI_SAMPLE_RATE),
                "add_wav_header": kwargs.get("add_wav_header", self.add_wav_header),
            }
            
            async with self._client as tts:
                audio_bytes = await tts.synthesize(text, **synthesis_kwargs)
                
                if not audio_bytes:
                    self.emit("error", "No audio data received from SmallestAI")
                    return

                await self._stream_audio_chunks(audio_bytes)

        except Exception as e:
            self.emit("error", f"SmallestAI synthesis failed: {str(e)}")
            raise

    async def _stream_audio_chunks(self, audio_bytes: bytes) -> None:
        """Stream audio data in chunks to ensure smooth playback"""
        chunk_size = int(SMALLESTAI_SAMPLE_RATE * SMALLESTAI_CHANNELS * 2 * 20 / 1000)
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
        """Cleanup resources"""
        if hasattr(self, '_client'):
            try:
                
                if hasattr(self._client, 'aclose'):
                    await self._client.aclose()
            except Exception:
                pass  
        await super().aclose()

    async def interrupt(self) -> None:
        """Interrupt the TTS process"""
        if self.audio_track:
            self.audio_track.interrupt()