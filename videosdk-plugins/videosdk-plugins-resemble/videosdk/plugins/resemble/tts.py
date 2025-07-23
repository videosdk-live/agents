from __future__ import annotations

from typing import Any, AsyncIterator, Optional
import os
import asyncio
import httpx
from dataclasses import dataclass

from videosdk.agents import TTS

RESEMBLE_HTTP_STREAMING_URL = "https://f.cluster.resemble.ai/stream"
DEFAULT_VOICE_UUID = "55592656"
DEFAULT_SAMPLE_RATE = 22050
DEFAULT_PRECISION = "PCM_16"

class ResembleTTS(TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        voice_uuid: str = DEFAULT_VOICE_UUID,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        precision: str = DEFAULT_PRECISION,
    ) -> None:
        super().__init__(sample_rate=sample_rate, num_channels=1)

        self.api_key = api_key or os.getenv("RESEMBLE_API_KEY")
        if not self.api_key:
            raise ValueError("Resemble API key is required. Provide either `api_key` or set `RESEMBLE_API_KEY` environment variable.")
        
        self.voice_uuid = voice_uuid
        self.precision = precision

        self.audio_track = None
        self.loop = None
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
                self.emit("error", "Audio track or event loop not set")
                return

            await self._http_stream_synthesis(full_text)

        except Exception as e:
            self.emit("error", f"Resemble TTS synthesis failed: {str(e)}")

    async def _http_stream_synthesis(self, text: str) -> None:
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "voice_uuid": self.voice_uuid,
            "data": text,
            "precision": self.precision,
            "sample_rate": self.sample_rate,
        }

        try:
            async with self._http_client.stream(
                "POST", 
                RESEMBLE_HTTP_STREAMING_URL,
                headers=headers, 
                json=payload
            ) as response:
                response.raise_for_status()

                audio_data = b""
                header_processed = False

                async for chunk in response.aiter_bytes():
                    if not header_processed:
                        audio_data += chunk
                        data_pos = audio_data.find(b'data')
                        if data_pos != -1:
                            header_size = data_pos + 8
                            audio_data = audio_data[header_size:]
                            header_processed = True
                    else:
                        if chunk:
                            audio_data += chunk

                if audio_data:
                    await self._stream_audio_chunks(audio_data)
                        
        except httpx.HTTPStatusError as e:
            self.emit("error", f"HTTP error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            self.emit("error", f"HTTP streaming synthesis failed: {str(e)}")

    async def _stream_audio_chunks(self, audio_bytes: bytes) -> None:
        """Stream audio data in chunks for smooth playback """
        chunk_size = int(self.sample_rate * 1 * 2 * 20 / 1000)  
        
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i + chunk_size]
            
            if len(chunk) < chunk_size and len(chunk) > 0:
                padding_needed = chunk_size - len(chunk)
                chunk += b'\x00' * padding_needed
            
            if len(chunk) == chunk_size:
                self.loop.create_task(self.audio_track.add_new_bytes(chunk))
                await asyncio.sleep(0.001)

    async def aclose(self) -> None:
        if self._http_client:
            await self._http_client.aclose()
        await super().aclose()

    async def interrupt(self) -> None:
        if self.audio_track:
            self.audio_track.interrupt()
