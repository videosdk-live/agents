from __future__ import annotations

from typing import Any, AsyncIterator, Literal, Optional, Union
import os
import httpx
import asyncio
import json
import aiohttp
from dataclasses import dataclass

from videosdk.agents import TTS

ELEVENLABS_SAMPLE_RATE = 24000
ELEVENLABS_CHANNELS = 1

DEFAULT_MODEL = "eleven_flash_v2_5"
DEFAULT_VOICE_ID = "EXAVITQu4vr4xnSDxMaL"
API_BASE_URL = "https://api.elevenlabs.io/v1"


@dataclass
class VoiceSettings:
    stability: float = 0.71
    similarity_boost: float = 0.5
    style: float = 0.0
    use_speaker_boost: bool = True


class ElevenLabsTTS(TTS):
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        voice: str = DEFAULT_VOICE_ID,
        speed: float = 1.0,
        api_key: str | None = None,
        response_format: str = "pcm_24000",
        voice_settings: VoiceSettings | None = None,
        base_url: str = API_BASE_URL,
        enable_streaming: bool = False,
    ) -> None:
        super().__init__(sample_rate=ELEVENLABS_SAMPLE_RATE, num_channels=ELEVENLABS_CHANNELS)

        self.model = model
        self.voice = voice
        self.speed = speed
        self.audio_track = None
        self.loop = None
        self.response_format = response_format
        self.base_url = base_url
        self.enable_streaming = enable_streaming
        self.voice_settings = voice_settings or VoiceSettings()

        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError("ElevenLabs API key must be provided either through api_key parameter or ELEVENLABS_API_KEY environment variable")

        self._session = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=15.0, read=30.0, write=5.0, pool=5.0),
            follow_redirects=True,
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

            target_voice = voice_id or self.voice

            if self.enable_streaming:
                await self._stream_synthesis(full_text, target_voice)
            else:
                await self._chunked_synthesis(full_text, target_voice)

        except Exception as e:
            self.emit("error", f"TTS synthesis failed: {str(e)}")

    async def _chunked_synthesis(self, text: str, voice_id: str) -> None:
        """Non-streaming synthesis using the standard API"""
        url = f"{self.base_url}/text-to-speech/{voice_id}/stream"
        
        params = {
            "model_id": self.model,
            "output_format": self.response_format,
        }
        
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        
        payload = {
            "text": text,
            "voice_settings": {
                "stability": self.voice_settings.stability,
                "similarity_boost": self.voice_settings.similarity_boost,
                "style": self.voice_settings.style,
                "use_speaker_boost": self.voice_settings.use_speaker_boost,
            },
        }

        try:
            async with self._session.stream(
                "POST", 
                url, 
                headers=headers, 
                json=payload,
                params=params
            ) as response:
                response.raise_for_status()
                
                audio_data = b""
                async for chunk in response.aiter_bytes():
                    if chunk:
                        audio_data += chunk

                if audio_data:
                    await self._stream_audio_chunks(audio_data)
                        
        except httpx.HTTPStatusError as e:
            self.emit("error", f"HTTP error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            self.emit("error", f"Chunked synthesis failed: {str(e)}")

    async def _stream_synthesis(self, text: str, voice_id: str) -> None:
        """WebSocket-based streaming synthesis"""
        ws_url = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"
        
        params = {
            "model_id": self.model,
            "output_format": self.response_format,
        }
        
        param_string = "&".join([f"{k}={v}" for k, v in params.items()])
        full_ws_url = f"{ws_url}?{param_string}"
        
        headers = {"xi-api-key": self.api_key}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(full_ws_url, headers=headers) as ws:
                    init_message = {
                        "text": " ",
                        "voice_settings": {
                            "stability": self.voice_settings.stability,
                            "similarity_boost": self.voice_settings.similarity_boost,
                            "style": self.voice_settings.style,
                            "use_speaker_boost": self.voice_settings.use_speaker_boost,
                        },
                    }
                    await ws.send_str(json.dumps(init_message))
                    
                    text_message = {"text": f"{text} "}
                    await ws.send_str(json.dumps(text_message))
            
                    eos_message = {"text": ""}
                    await ws.send_str(json.dumps(eos_message))
                    
                    audio_data = b""
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            if data.get("audio"):
                                import base64
                                audio_chunk = base64.b64decode(data["audio"])
                                audio_data += audio_chunk
                            elif data.get("isFinal"):
                                break
                            elif data.get("error"):
                                self.emit("error", f"WebSocket error: {data['error']}")
                                break
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            self.emit("error", f"WebSocket connection error: {ws.exception()}")
                            break

                    if audio_data:
                        await self._stream_audio_chunks(audio_data)
                            
        except Exception as e:
            self.emit("error", f"Streaming synthesis failed: {str(e)}")

    async def _stream_audio_chunks(self, audio_bytes: bytes) -> None:
        """Stream audio data in chunks for smooth playback"""
        chunk_size = int(ELEVENLABS_SAMPLE_RATE * ELEVENLABS_CHANNELS * 2 * 20 / 1000)
        
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
        if self._session:
            await self._session.aclose()
        await super().aclose()

    async def interrupt(self) -> None:
        """Interrupt the TTS process"""
        if self.audio_track:
            self.audio_track.interrupt()
