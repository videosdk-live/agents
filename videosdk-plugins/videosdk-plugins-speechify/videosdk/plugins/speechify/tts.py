from __future__ import annotations

from typing import Any, AsyncIterator, Literal, Optional
import os
import httpx
import io
import asyncio

from pydub import AudioSegment

from videosdk.agents import TTS, segment_text

SPEECHIFY_SAMPLE_RATE = 24000
SPEECHIFY_CHANNELS = 1
SPEECHIFY_STREAM_ENDPOINT = "https://api.sws.speechify.com/v1/audio/stream"


class SpeechifyTTS(TTS):
    def __init__(
        self,
        *,
        voice_id: str = "kristy",
        api_key: Optional[str] = None,
        model: Literal[
            "simba-base", "simba-english", "simba-multilingual", "simba-turbo"
        ] = "simba-english",
        language: Optional[str] = None,
        audio_format: Literal["mp3", "ogg", "aac"] = "mp3",
    ) -> None:
        super().__init__(
            sample_rate=SPEECHIFY_SAMPLE_RATE, num_channels=SPEECHIFY_CHANNELS
        )

        self.voice_id = voice_id
        self.model = model
        self.language = language
        self.audio_format = audio_format
        self.audio_track = None
        self.loop = None
        self._first_chunk_sent = False
        self._current_synthesis_task: asyncio.Task | None = None
        self._interrupted = False

        self.api_key = api_key or os.getenv("SPEECHIFY_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Speechify API key required. Provide either:\n"
                "1. api_key parameter, OR\n"
                "2. SPEECHIFY_API_KEY environment variable"
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

            self._interrupted = False

            if isinstance(text, AsyncIterator):
                async for segment in segment_text(text):
                    if self._interrupted:
                        break
                    await self._stream_synthesis(segment)
            else:
                if not self._interrupted:
                    await self._stream_synthesis(text)

        except Exception as e:
            self.emit("error", f"Speechify TTS synthesis failed: {str(e)}")

    async def _stream_synthesis(self, text: str) -> None:
        """Synthesize text to speech using Speechify stream endpoint"""
        if not text.strip() or self._interrupted:
            return

        try:
            headers = {
                "Accept": f"audio/{self.audio_format}",
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "input": text,
                "voice_id": self.voice_id,
                "model": self.model,
            }

            if self.language:
                payload["language"] = self.language

            async with self._http_client.stream(
                "POST",
                SPEECHIFY_STREAM_ENDPOINT,
                headers=headers,
                json=payload
            ) as response:
                response.raise_for_status()

                audio_data = b""
                async for chunk in response.aiter_bytes():
                    if self._interrupted:
                        break
                    if chunk:
                        audio_data += chunk

                if audio_data and not self._interrupted:
                    await self._decode_and_stream(audio_data)

        except httpx.HTTPStatusError as e:
            if not self._interrupted:
                error_msg = f"HTTP error {e.response.status_code}"
                try:
                    error_data = e.response.json()
                    if isinstance(error_data, dict) and "error" in error_data:
                        error_msg = f"{error_msg}: {error_data['error']}"
                except:
                    pass
                self.emit(
                    "error", f"Speechify stream synthesis failed: {error_msg}")
        except Exception as e:
            if not self._interrupted:
                self.emit("error", f"Stream synthesis failed: {str(e)}")

    async def _decode_and_stream(self, audio_bytes: bytes) -> None:
        """Decode compressed audio to PCM and stream it"""
        if self._interrupted:
            return

        try:
            audio = AudioSegment.from_file(
                io.BytesIO(audio_bytes),
                format=self.audio_format
            )

            audio = audio.set_frame_rate(SPEECHIFY_SAMPLE_RATE)
            audio = audio.set_channels(SPEECHIFY_CHANNELS)
            audio = audio.set_sample_width(2)

            pcm_data = audio.raw_data

            chunk_size = int(SPEECHIFY_SAMPLE_RATE *
                             SPEECHIFY_CHANNELS * 2 * 20 / 1000)  # 20ms chunks

            for i in range(0, len(pcm_data), chunk_size):
                if self._interrupted:
                    break

                chunk = pcm_data[i:i + chunk_size]

                if len(chunk) < chunk_size and len(chunk) > 0:
                    padding_needed = chunk_size - len(chunk)
                    chunk += b'\x00' * padding_needed

                if len(chunk) == chunk_size:
                    if not self._first_chunk_sent and self._first_audio_callback:
                        self._first_chunk_sent = True
                        await self._first_audio_callback()

                    asyncio.create_task(
                        self.audio_track.add_new_bytes(chunk))
                    await asyncio.sleep(0.001)

        except Exception as e:
            if not self._interrupted:
                self.emit("error", f"Audio decoding failed: {str(e)}")

    async def aclose(self) -> None:
        if self._http_client:
            await self._http_client.aclose()
        await super().aclose()

    async def interrupt(self) -> None:
        """Interrupt TTS synthesis"""
        self._interrupted = True
        if self._current_synthesis_task and not self._current_synthesis_task.done():
            self._current_synthesis_task.cancel()
        if self.audio_track:
            self.audio_track.interrupt()
