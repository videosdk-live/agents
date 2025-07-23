from __future__ import annotations

from typing import Any, AsyncIterator, Literal, Optional, Union
import httpx
import os
import openai
import asyncio

from videosdk.agents import TTS

OPENAI_TTS_SAMPLE_RATE = 24000
OPENAI_TTS_CHANNELS = 1

DEFAULT_MODEL = "gpt-4o-mini-tts"
DEFAULT_VOICE = "ash"
_RESPONSE_FORMATS = Union[Literal["mp3", "opus", "aac", "flac", "wav", "pcm"], str]

class OpenAITTS(TTS):
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        voice: str = DEFAULT_VOICE,
        speed: float = 1.0,
        instructions: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        response_format: str = "pcm"
    ) -> None:
        super().__init__(sample_rate=OPENAI_TTS_SAMPLE_RATE, num_channels=OPENAI_TTS_CHANNELS)
        
        self.model = model
        self.voice = voice
        self.speed = speed
        self.instructions = instructions
        self.audio_track = None
        self.loop = None
        self.response_format = response_format
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided either through api_key parameter or OPENAI_API_KEY environment variable")
        
        self._client = openai.AsyncClient(
            max_retries=0,
            api_key=self.api_key,
            base_url=base_url or None,
            http_client=httpx.AsyncClient(
                timeout=httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=50,
                    max_keepalive_connections=50,
                    keepalive_expiry=120,
                ),
            ),
        )
    
    async def synthesize(
        self,
        text: AsyncIterator[str] | str,
        voice_id: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Convert text to speech using OpenAI's TTS API and stream to audio track
        
        Args:
            text: Text to convert to speech
            voice_id: Optional voice override
            **kwargs: Additional provider-specific arguments
        """
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

            audio_data = b""
            async with self._client.audio.speech.with_streaming_response.create(
                model=self.model,
                voice=voice_id or self.voice,
                input=full_text,
                speed=self.speed,
                response_format=self.response_format,
                **({"instructions": self.instructions} if self.instructions else {})
            ) as response:
                async for chunk in response.iter_bytes():
                    if chunk:
                        audio_data += chunk


            if audio_data:
                await self._stream_audio_chunks(audio_data)

        except openai.APIError as e:
            self.emit("error", str(e))
        except Exception as e:
            self.emit("error", f"TTS synthesis failed: {str(e)}")

    async def _stream_audio_chunks(self, audio_bytes: bytes) -> None:
        """Stream audio data in chunks for smooth playback"""
        chunk_size = int(OPENAI_TTS_SAMPLE_RATE * OPENAI_TTS_CHANNELS * 2 * 20 / 1000) 
        
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i + chunk_size]
            
            if len(chunk) < chunk_size and len(chunk) > 0:
                padding_needed = chunk_size - len(chunk)
                chunk += b'\x00' * padding_needed
            
            if len(chunk) == chunk_size:
                self.loop.create_task(self.audio_track.add_new_bytes(chunk))
                await asyncio.sleep(0.001)

    async def aclose(self) -> None:
        """Cleanup resources"""
        await self._client.close()
        await super().aclose()

    async def interrupt(self) -> None:
        """Interrupt the TTS process"""
        if self.audio_track:
            self.audio_track.interrupt()