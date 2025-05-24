from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, AsyncIterator, Literal, Optional, Union
import httpx
import os
import openai
import sounddevice as sd
import numpy as np
import io
from pydub import AudioSegment

from videosdk.agents.tts.tts import TTS
from videosdk.agents import CustomAudioStreamTrack

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
        audio_track: CustomAudioStreamTrack | None = None,
        response_format: str = "mp3"
    ) -> None:
        super().__init__(sample_rate=OPENAI_TTS_SAMPLE_RATE, num_channels=OPENAI_TTS_CHANNELS)
        
        self.model = model
        self.voice = voice
        self.speed = speed
        self.instructions = instructions
        self.audio_track = audio_track
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
        Convert text to speech using OpenAI's TTS API
        
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
            # TODO: Remove this temporary code
            # Temporary for testing purposes
            async def _play_sound():
                try:
                    async with self._client.audio.speech.with_streaming_response.create(
                        model=self.model,
                        voice=voice_id or self.voice,
                        input=full_text,
                        speed=self.speed,
                        response_format=self.response_format,
                        **({"instructions": self.instructions} if self.instructions else {})
                    ) as response:
                        if self.response_format == "pcm":
                            audio_data = bytearray()
                            async for chunk in response.iter_bytes():
                                audio_data.extend(chunk)
                            
                            samples = np.frombuffer(audio_data, dtype=np.int16)
                            
                            sd.play(samples, OPENAI_TTS_SAMPLE_RATE)
                            sd.wait()
                        else:
                            audio_data = bytearray()
                            async for chunk in response.iter_bytes():
                                audio_data.extend(chunk)

                            audio_segment = AudioSegment.from_file(
                                io.BytesIO(audio_data),
                                format=self.response_format
                            )

                            samples = np.array(audio_segment.get_array_of_samples())
                            if audio_segment.channels == 2:
                                samples = samples.reshape((-1, 2))

                            sd.play(samples, audio_segment.frame_rate)
                            sd.wait()

                except Exception as e:
                    self.emit("error", f"Audio playback failed: {str(e)}")

            await _play_sound()

        except openai.APIError as e:
            self.emit("error", str(e))
        except Exception as e:
            self.emit("error", f"TTS synthesis failed: {str(e)}")


    async def aclose(self) -> None:
        """Cleanup resources"""
        await self._client.close()
        await super().aclose()