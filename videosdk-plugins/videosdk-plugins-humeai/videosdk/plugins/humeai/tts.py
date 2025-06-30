from __future__ import annotations

from typing import Any, AsyncIterator, Literal, Optional, Union, List, Dict
import os
import httpx
import asyncio
import json
import base64
from dataclasses import dataclass

from videosdk.agents import TTS

HUMEAI_SAMPLE_RATE = 24000
HUMEAI_CHANNELS = 1

# Note: In instant mode, only "Serene Assistant" is available
# For different voices, disable instant_mode and use custom voice IDs
DEFAULT_VOICE = "Serene Assistant"
DEFAULT_MODEL = "instant"  # Hume's instant mode
API_BASE_URL = "https://api.hume.ai/v0"

_RESPONSE_FORMATS = Union[Literal["pcm", "mp3", "wav"], str]

@dataclass
class Utterance:
    """Single utterance for text-to-speech conversion"""
    text: str
    description: Optional[str] = None
    voice: Optional[str] = None
    speed: Optional[float] = None
    trailing_silence: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values"""
        result = {"text": self.text}
        if self.description is not None:
            result["description"] = self.description
        if self.voice is not None:
            result["voice"] = self.voice
        if self.speed is not None:
            result["speed"] = self.speed
        if self.trailing_silence is not None:
            result["trailing_silence"] = self.trailing_silence
        return result

@dataclass
class Context:
    """Context utterances for consistent speech style and prosody"""
    utterances: List[Utterance]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format expected by Hume AI API"""
        return {
            "utterances": [utt.to_dict() for utt in self.utterances]
        }

class HumeAITTS(TTS):
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        voice: str = DEFAULT_VOICE,
        speed: float = 2.0,
        api_key: str | None = None,
        base_url: str = API_BASE_URL,
        response_format: str = "pcm",
        instant_mode: bool = True
    ) -> None:
        super().__init__(sample_rate=HUMEAI_SAMPLE_RATE, num_channels=HUMEAI_CHANNELS)
        
        self.model = model
        self.voice = voice
        self.speed = speed
        self.audio_track = None
        self.loop = None
        self.response_format = response_format
        self.base_url = base_url
        self.instant_mode = instant_mode
        
        self.api_key = api_key or os.getenv("HUMEAI_API_KEY")
        if not self.api_key:
            raise ValueError("Hume AI API key must be provided either through api_key parameter or HUMEAI_API_KEY environment variable")
        
        self._session = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=15.0, read=30.0, write=5.0, pool=5.0),
            follow_redirects=True,
        )
    
    async def synthesize(
        self,
        text: AsyncIterator[str] | str,
        voice_id: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Convert text to speech using Hume AI's streaming API and stream to audio track
        
        Args:
            text: Text to convert to speech
            voice_id: Optional voice override
            **kwargs: Additional provider-specific arguments including:
                - context: Context object for consistent speech style
                - trailing_silence: Float seconds of silence to add after speech
                - description: Voice description for dynamic voice generation
                - speed: Override instance speed for this synthesis
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

            # Use provided voice_id or fall back to instance voice
            target_voice = voice_id or self.voice

            # Build format object
            if self.response_format == "mp3":
                format_obj = {"type": "mp3"}
            elif self.response_format == "wav":
                format_obj = {"type": "wav"}
            else:  # pcm
                format_obj = {"type": "pcm"}

            # Build utterance with support for trailing_silence
            utterance = {
                "text": full_text,
                "speed": kwargs.get("speed", self.speed)
            }
            
            # Add trailing_silence if provided
            if "trailing_silence" in kwargs:
                utterance["trailing_silence"] = kwargs["trailing_silence"]
            
            # Add voice configuration based on mode
            if self.instant_mode:
                # Instant mode uses voice name
                utterance["voice"] = {
                    "name": target_voice,
                    "provider": "HUME_AI"
                }
            else:
                # Non-instant mode can use voice ID or description
                if target_voice and target_voice != "Serene Assistant":
                    # If it looks like a voice ID (UUID format), use it directly
                    if len(target_voice) == 36 and "-" in target_voice:
                        utterance["voice"] = target_voice
                    else:
                        # Otherwise use as description
                        utterance["description"] = target_voice
                
                # Allow description override from kwargs
                if "description" in kwargs:
                    utterance["description"] = kwargs["description"]

            # Build request payload
            payload = {
                "utterances": [utterance],
                "format": format_obj,
                "instant_mode": self.instant_mode,
                "num_generations": 1,
                "split_utterances": True,
                "strip_headers": False,
            }

            # Add context if provided
            if "context" in kwargs:
                context = kwargs["context"]
                if isinstance(context, Context):
                    payload["context"] = context.to_dict()
                elif isinstance(context, dict):
                    payload["context"] = context
                else:
                    self.emit("error", "Context must be a Context object or dictionary")

            await self._stream_synthesis(payload)

        except Exception as e:
            self.emit("error", f"TTS synthesis failed: {str(e)}")

    async def _stream_synthesis(self, payload: dict) -> None:
        """Stream audio synthesis from Hume AI API"""
        # Both instant and non-instant mode use the same endpoint
        url = f"{self.base_url}/tts/stream/json"
        
        headers = {
            "X-Hume-Api-Key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        try:
            async with self._session.stream(
                "POST",
                url,
                headers=headers,
                json=payload
            ) as response:
                response.raise_for_status()
                
                # Process streaming JSON response
                buffer = b""
                async for chunk in response.aiter_bytes():
                    buffer += chunk
                    
                    # Try to parse complete JSON objects from buffer
                    while b'\n' in buffer:
                        line, buffer = buffer.split(b'\n', 1)
                        line = line.strip()
                        
                        if not line:
                            continue
                        
                        try:
                            data = json.loads(line)
                            
                            # Process audio chunk
                            if "audio" in data and data["audio"]:
                                audio_bytes = base64.b64decode(data["audio"])
                                
                                # Handle different formats
                                if self.response_format == "wav":
                                    audio_bytes = self._remove_wav_header(audio_bytes)
                                
                                self.loop.create_task(
                                    self.audio_track.add_new_bytes(audio_bytes)
                                )
                                
                        except json.JSONDecodeError:
                            buffer = line + b'\n' + buffer
                            break
                
                # Process any remaining data in buffer
                if buffer.strip():
                    try:
                        data = json.loads(buffer)
                        if "audio" in data and data["audio"]:
                            audio_bytes = base64.b64decode(data["audio"])
                            
                            if self.response_format == "wav":
                                audio_bytes = self._remove_wav_header(audio_bytes)
                            
                            self.loop.create_task(
                                self.audio_track.add_new_bytes(audio_bytes)
                            )
                    except json.JSONDecodeError:
                        pass
                        
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                self.emit("error", "Hume AI authentication failed. Please check your API key.")
            elif e.response.status_code == 400:
                try:
                    error_data = e.response.json()
                    error_msg = error_data.get("error", {}).get("message", "Bad request")
                    self.emit("error", f"Hume AI request error: {error_msg}")
                except:
                    self.emit("error", "Hume AI bad request. Please check your configuration.")
            else:
                self.emit("error", f"Hume AI HTTP error: {e.response.status_code}")
        except httpx.TimeoutException:
            self.emit("error", "Request timeout")
        except Exception as e:
            self.emit("error", f"Streaming synthesis failed: {str(e)}")

    def _remove_wav_header(self, audio_bytes: bytes) -> bytes:
        """Remove WAV header if present to get raw PCM data"""
        if audio_bytes.startswith(b'RIFF'):
            data_pos = audio_bytes.find(b'data')
            if data_pos != -1:
                return audio_bytes[data_pos + 8:]
        return audio_bytes

    async def aclose(self) -> None:
        """Cleanup resources"""
        if self._session:
            await self._session.aclose()
        await super().aclose()

    async def interrupt(self) -> None:
        """Interrupt the TTS process"""
        if self.audio_track:
            self.audio_track.interrupt() 