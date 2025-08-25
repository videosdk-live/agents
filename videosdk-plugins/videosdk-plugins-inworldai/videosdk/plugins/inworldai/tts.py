from __future__ import annotations

from typing import Any, AsyncIterator, Optional
import os
import base64
import json
import httpx
import asyncio

from videosdk.agents import TTS, segment_text

INWORLD_SAMPLE_RATE = 24000
INWORLD_CHANNELS = 1
INWORLD_TTS_STREAMING_ENDPOINT = "https://api.inworld.ai/tts/v1/voice:stream"

DEFAULT_MODEL = "inworld-tts-1"
DEFAULT_VOICE = "Hades"
DEFAULT_TEMPERATURE = 0.8


class InworldAITTS(TTS):
    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL,
        voice_id: str = DEFAULT_VOICE,
        temperature: float = DEFAULT_TEMPERATURE,
        api_key: str | None = None,
        audio_encoding: str = "LINEAR16",
        sample_rate: int = INWORLD_SAMPLE_RATE,
    ) -> None:
        super().__init__(sample_rate=sample_rate, num_channels=INWORLD_CHANNELS)

        self.model_id = model_id
        self.voice_id = voice_id
        self.temperature = temperature
        self.audio_encoding = audio_encoding
        self.audio_track = None
        self.loop = None
        self._first_chunk_sent = False

        self.api_key = api_key or os.getenv("INWORLD_API_KEY")
        if not self.api_key:
            raise ValueError(
                "InworldAI API key must be provided either through:\n"
                "1. api_key parameter, OR\n"
                "2. INWORLD_API_KEY environment variable"
            )

        self._auth_header = f"Basic {self.api_key}"

        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=15.0, read=30.0,
                                  write=5.0, pool=5.0),
            follow_redirects=True,
            limits=httpx.Limits(
                max_connections=50,
                max_keepalive_connections=50,
                keepalive_expiry=120,
            ),
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
        Convert text to speech using InworldAI's streaming TTS API

        Args:
            text: Text to convert to speech
            voice_id: Optional voice override
            **kwargs: Additional provider-specific arguments
        """
        try:
            if not self.audio_track or not self.loop:
                self.emit("error", "Audio track or event loop not set")
                return

            if isinstance(text, AsyncIterator):
                async for segment in segment_text(text):
                    await self._synthesize_streaming(segment, voice_id)
            else:
                await self._synthesize_streaming(text, voice_id)

        except Exception as e:
            self.emit("error", f"InworldAI TTS synthesis failed: {str(e)}")

    async def _synthesize_streaming(
        self, text: str, voice_id: Optional[str] = None
    ) -> None:
        """Synthesize text using the streaming endpoint"""
        try:
            payload = {
                "text": text,
                "voiceId": voice_id or self.voice_id,
                "modelId": self.model_id,
                "audioConfig": {
                    "temperature": self.temperature,
                    "audioEncoding": self.audio_encoding,
                    "sampleRateHertz": self._sample_rate,
                },
            }

            headers = {
                "Authorization": self._auth_header,
                "Content-Type": "application/json",
                "Accept": "application/json",
            }

            async with self._http_client.stream(
                "POST",
                INWORLD_TTS_STREAMING_ENDPOINT,
                headers=headers,
                json=payload,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    try:
                        data = json.loads(line)

                        if "error" in data:
                            error = data["error"]
                            self.emit(
                                "error", f"InworldAI API error: {error.get('message', 'Unknown error')}")
                            return

                        if "result" in data and "audioContent" in data["result"]:
                            audio_content_b64 = data["result"]["audioContent"]
                            if audio_content_b64:
                                audio_bytes = base64.b64decode(
                                    audio_content_b64)

                                await self._stream_audio_chunk(audio_bytes)

                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        self.emit(
                            "error", f"Error processing stream chunk: {str(e)}")

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                self.emit(
                    "error", "InworldAI authentication failed. Please check your API key.")
            elif e.response.status_code == 400:
                try:
                    error_data = e.response.json()
                    error_msg = error_data.get("error", {}).get(
                        "message", "Bad request")
                    self.emit("error", f"InworldAI request error: {error_msg}")
                except:
                    self.emit(
                        "error", "InworldAI bad request. Please check your parameters.")
            else:
                self.emit(
                    "error", f"InworldAI HTTP error: {e.response.status_code}")
            raise

    async def _stream_audio_chunk(self, audio_bytes: bytes) -> None:
        """Stream a single audio chunk, removing WAV header if present"""
        if not audio_bytes:
            return

        audio_data = self._remove_wav_header(audio_bytes)

        if audio_data:
            if not self._first_chunk_sent and self._first_audio_callback:
                self._first_chunk_sent = True
                await self._first_audio_callback()

            asyncio.create_task(self.audio_track.add_new_bytes(audio_data))
            await asyncio.sleep(0.001)

    def _remove_wav_header(self, audio_bytes: bytes) -> bytes:
        """Remove WAV header if present to get raw PCM data"""
        if audio_bytes.startswith(b"RIFF"):
            data_pos = audio_bytes.find(b"data")
            if data_pos != -1:
                return audio_bytes[data_pos + 8:]

        return audio_bytes

    async def aclose(self) -> None:
        """Cleanup resources"""
        if self._http_client:
            await self._http_client.aclose()
        await super().aclose()

    async def interrupt(self) -> None:
        """Interrupt the TTS process"""
        if self.audio_track:
            self.audio_track.interrupt()
