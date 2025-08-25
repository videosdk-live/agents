from __future__ import annotations

from typing import Any, AsyncIterator, Optional, Union
import os
import httpx
import io
import asyncio
from pydub import AudioSegment

from videosdk.agents import TTS, segment_text

PAPLA_SAMPLE_RATE = 24000
PAPLA_CHANNELS = 1
AUDIO_FORMAT = "mp3"

API_BASE_URL = "https://api.papla.media/v1"
DEFAULT_MODEL = "papla_p1"
DEFAULT_VOICE_ID = "6ce54263-cff6-457d-a72d-1387d0f28f6c"


class PaplaTTS(TTS):
    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str = API_BASE_URL,
    ) -> None:
        super().__init__(sample_rate=PAPLA_SAMPLE_RATE, num_channels=PAPLA_CHANNELS)

        self.model_id = model_id
        self.audio_track = None
        self.loop = None
        self.base_url = base_url
        self._first_chunk_sent = False

        self.api_key = api_key or os.getenv("PAPLA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Papla API key must be provided either through the 'api_key' "
                "parameter or the 'PAPLA_API_KEY' environment variable."
            )

        self._client = httpx.AsyncClient(
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
        voice_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Convert text to speech using Papla's streaming TTS API.
        This now includes decoding the received MP3 audio to raw PCM.
        """
        try:
            if not self.audio_track or not self.loop:
                self.emit(
                    "error", "Audio track or event loop not set by the framework.")
                return

            if isinstance(text, AsyncIterator):
                async for segment in segment_text(text):
                    await self._synthesize_segment(segment, voice_id, **kwargs)
            else:
                await self._synthesize_segment(text, voice_id, **kwargs)

        except Exception as e:
            self.emit("error", f"Papla TTS synthesis failed: {str(e)}")

    async def _synthesize_segment(self, text: str, voice_id: Optional[str] = None, **kwargs: Any) -> None:
        """Synthesize a single text segment"""
        if not text.strip():
            return

        target_voice = voice_id or DEFAULT_VOICE_ID
        url = f"{self.base_url}/text-to-speech/{target_voice}/stream"

        headers = {
            "papla-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "text": text,
            "model_id": self.model_id,
        }

        async with self._client.stream("POST", url, headers=headers, json=payload) as response:
            response.raise_for_status()

            mp3_data = b""
            async for chunk in response.aiter_bytes():
                if chunk:
                    mp3_data += chunk

            if mp3_data:
                asyncio.create_task(self._decode_and_stream_pcm(mp3_data))

    async def _decode_and_stream_pcm(self, audio_bytes: bytes) -> None:
        """Decodes compressed audio (MP3) into raw PCM and streams it to the audio track."""
        try:
            audio = AudioSegment.from_file(
                io.BytesIO(audio_bytes), format=AUDIO_FORMAT)

            audio = audio.set_frame_rate(PAPLA_SAMPLE_RATE)
            audio = audio.set_channels(PAPLA_CHANNELS)
            audio = audio.set_sample_width(2)

            pcm_data = audio.raw_data

            chunk_size = int(PAPLA_SAMPLE_RATE *
                             PAPLA_CHANNELS * 2 * 20 / 1000)

            for i in range(0, len(pcm_data), chunk_size):
                chunk = pcm_data[i:i + chunk_size]

                if 0 < len(chunk) < chunk_size:
                    padding = b"\x00" * (chunk_size - len(chunk))
                    chunk += padding

                if len(chunk) == chunk_size and self.audio_track:
                    if not self._first_chunk_sent and self._first_audio_callback:
                        self._first_chunk_sent = True
                        await self._first_audio_callback()

                    asyncio.create_task(self.audio_track.add_new_bytes(chunk))
                    await asyncio.sleep(0.01)

        except Exception as e:
            self.emit(
                "error", f"Failed to decode or stream Papla audio: {str(e)}")

    async def aclose(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        await super().aclose()

    async def interrupt(self) -> None:
        if self.audio_track:
            self.audio_track.interrupt()
