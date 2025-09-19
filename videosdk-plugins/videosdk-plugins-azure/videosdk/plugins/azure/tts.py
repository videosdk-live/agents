from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Literal, AsyncIterator, Optional, Any

import httpx

from videosdk.agents import TTS, segment_text 
import logging

logger = logging.getLogger(__name__)

@dataclass
class VoiceTuning:
    """Configuration for speech tuning (rate, volume, pitch)."""

    _rate: Literal["x-slow", "slow", "medium", "fast", "x-fast"] | float | None = None
    _volume: Literal["silent", "x-soft", "soft", "medium", "loud", "x-loud"] | float | None = None
    _pitch: Literal["x-low", "low", "medium", "high", "x-high"] | None = None

    @property
    def rate(self):
        return self._rate

    @rate.setter
    def rate(self, value):
        if value:
            if isinstance(value, float) and not 0.5 <= value <= 2.0:
                raise ValueError("Rate must be a float between 0.5 and 2.0")
            if isinstance(value, str) and value not in ["x-slow", "slow", "medium", "fast", "x-fast"]:
                raise ValueError("Rate must be one of 'x-slow', 'slow', 'medium', 'fast', 'x-fast'")
        self._rate = value

    @property
    def volume(self):
        return self._volume
    
    @volume.setter
    def volume(self, value):
        if value:
            if isinstance(value, float) and not 0 <= value <= 100.0:
                raise ValueError("Volume must be a float between 0 and 100")
            if isinstance(value, str) and value not in ["silent", "x-soft", "soft", "medium", "loud", "x-loud"]:
                raise ValueError("Volume must be one of 'silent', 'x-soft', 'soft', 'medium', 'loud', 'x-loud'")
        self._volume = value

    @property
    def pitch(self):
        return self._pitch

    @pitch.setter
    def pitch(self, value):
        if value and value not in ["x-low", "low", "medium", "high", "x-high"]:
            raise ValueError("Pitch must be one of 'x-low', 'low', 'medium', 'high', 'x-high'")
        self._pitch = value

    def __init__(self, rate=None, volume=None, pitch=None):
        self.rate = rate
        self.volume = volume
        self.pitch = pitch


@dataclass
class SpeakingStyle:
    """Configuration for speech expressive style."""
    style: str
    _degree: float | None = None

    @property
    def degree(self):
        return self._degree

    @degree.setter
    def degree(self, value: float | None):
        if value is not None and not 0.1 <= value <= 2.0:
            raise ValueError("Style degree must be between 0.1 and 2.0")
        self._degree = value
    
    def __init__(self, style: str, degree: float | None = None):
        self.style = style
        self.degree = degree


class AzureTTS(TTS):
    """
    Initialize the Azure TTS plugin.

    Args:
        voice (str): Name of the Azure neural voice to use (default: "en-US-EmmaNeural").
            For a full list of available voices, see:
            https://eastus2.tts.speech.microsoft.com/cognitiveservices/voices/list
            (Requires: curl --location --request GET with header 'Ocp-Apim-Subscription-Key')
        language (str, optional): Language code for the voice (e.g., "en-US"). If not provided, defaults to the voice's language.
        tuning (VoiceTuning, optional): VoiceTuning object to control speech rate, volume, and pitch.
        style (SpeakingStyle, optional): SpeakingStyle object for expressive speech synthesis.
        speech_key (str, optional): Azure Speech API key. If not provided, uses the AZURE_SPEECH_KEY environment variable.
        speech_region (str, optional): Azure Speech region. If not provided, uses the AZURE_SPEECH_REGION environment variable.
        speech_endpoint (str, optional): Custom endpoint URL. If not provided, uses the AZURE_SPEECH_ENDPOINT environment variable.
        deployment_id (str, optional): Custom deployment ID for model deployment scenarios.
        speech_auth_token (str, optional): Azure Speech authorization token for token-based authentication.

    """
    FIXED_SAMPLE_RATE = 24000
    AZURE_OUTPUT_FORMAT = "raw-24khz-16bit-mono-pcm"

    def __init__(
        self,
        *,
        voice: str = "en-US-EmmaNeural",
        language: str | None = None,
        tuning: Optional[VoiceTuning] = None,
        style: Optional[SpeakingStyle] = None,
        speech_key: str | None = None,
        speech_region: str | None = None,
        speech_endpoint: str | None = None,
        deployment_id: str | None = None,
        speech_auth_token: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            sample_rate=self.FIXED_SAMPLE_RATE,
            num_channels=1,
        )
        
        self.speech_key = speech_key or os.environ.get("AZURE_SPEECH_KEY")
        self.speech_region = speech_region or os.environ.get("AZURE_SPEECH_REGION")
        self.speech_endpoint = speech_endpoint or os.environ.get(
            "AZURE_SPEECH_ENDPOINT"
        )
        self.speech_auth_token = speech_auth_token
        self.deployment_id = deployment_id

        has_endpoint = bool(self.speech_endpoint)
        has_key_and_region = bool(self.speech_key and self.speech_region)
        has_token_and_region = bool(self.speech_auth_token and self.speech_region)

        if not (has_endpoint or has_key_and_region or has_token_and_region):
            raise ValueError(
                "Authentication requires one of: speech_endpoint, (speech_key & speech_region), or (speech_auth_token & speech_region)."
            )

        self.voice = voice
        self.language = language
        self.tuning = tuning
        self.style = style

        self._first_chunk_sent = False
        self._interrupted = False
        self._http_client: Optional[httpx.AsyncClient] = None
    
    
    def reset_first_audio_tracking(self) -> None:
        self._first_chunk_sent = False

    def _get_endpoint_url(self) -> str:
        if self.speech_endpoint:
            base = self.speech_endpoint.rstrip("/")
            if not base.endswith("/cognitiveservices/v1"):
                base = f"{base}/cognitiveservices/v1"
        else:
            base = f"https://{self.speech_region}.tts.speech.microsoft.com/cognitiveservices/v1"

        if self.deployment_id:
            return f"{base}?deploymentId={self.deployment_id}"
        return base

    def _get_http_client(self) -> httpx.AsyncClient:
        if not self._http_client:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=15.0, read=30.0, write=5.0, pool=5.0
                ),
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=50,
                    max_keepalive_connections=50,
                    keepalive_expiry=120,
                ),
            )
        return self._http_client

    async def synthesize(
        self,
        text: AsyncIterator[str] | str,
        voice_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        try:
            if not self.audio_track or not self.loop:
                self.emit("error", "Audio track or event loop not set")
                return

            self._interrupted = False

            if isinstance(text, AsyncIterator):
                async for segment in segment_text(text):
                    if self._interrupted:
                        break
                    await self._synthesize_segment(segment, voice_id, **kwargs)
            else:
                if not self._interrupted:
                    await self._synthesize_segment(text, voice_id, **kwargs)

        except Exception as e:
            logger.error("Azure TTS synthesis failed: %s", str(e), exc_info=True)
            self.emit("error", f"Azure TTS synthesis failed: {str(e)}")

    async def _synthesize_segment(
        self, text: str, voice_id: Optional[str] = None, **kwargs: Any
    ) -> None:
        if not text.strip() or self._interrupted:
            return

        try:
            
            headers = {
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": self.AZURE_OUTPUT_FORMAT,
                "User-Agent": "VideoSDK Agents", 
            }

            if self.speech_auth_token:
                headers["Authorization"] = f"Bearer {self.speech_auth_token}"
            elif self.speech_key:
                headers["Ocp-Apim-Subscription-Key"] = self.speech_key

            ssml_data = self._build_ssml(text, voice_id or self.voice)

            response = await self._get_http_client().post(
                url=self._get_endpoint_url(),
                headers=headers,
                content=ssml_data,
            )
            response.raise_for_status()

            audio_data = b""
            async for chunk in response.aiter_bytes(chunk_size=8192):
                if self._interrupted: 
                    break
                if chunk: 
                    audio_data += chunk

            if audio_data and not self._interrupted:
                await self._stream_audio_chunks(audio_data)
                
        except httpx.TimeoutException:
            logger.error("Azure TTS request timeout")
            self.emit("error", "Azure TTS request timeout")
        except httpx.HTTPStatusError as e:
            logger.error("Azure TTS HTTP error: %s - %s", e.response.status_code, e.response.text)
            self.emit("error", f"Azure TTS HTTP error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            if not self._interrupted:
                logger.error("Azure TTS synthesis failed: %s", str(e), exc_info=True)
                self.emit("error", f"Azure TTS synthesis failed: {str(e)}")

    def _build_ssml(self, text: str, voice: str) -> str:
        lang = self.language or "en-US"
        ssml = (
            f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
            f'xmlns:mstts="http://www.w3.org/2001/mstts" xml:lang="{lang}">'
        )
        ssml += f'<voice name="{voice}">'

        if self.style:
            degree = f' styledegree="{self.style.degree}"' if self.style.degree else ""
            ssml += f'<mstts:express-as style="{self.style.style}"{degree}>'

        if self.tuning:
            t = self.tuning
            rate_attr = f' rate="{t.rate}"' if t.rate is not None else ""
            vol_attr = f' volume="{t.volume}"' if t.volume is not None else ""
            pitch_attr = f' pitch="{t.pitch}"' if t.pitch is not None else ""
            ssml += f"<prosody{rate_attr}{vol_attr}{pitch_attr}>{text}</prosody>"
        else:
            ssml += text

        if self.style:
            ssml += "</mstts:express-as>"

        ssml += "</voice></speak>"
        return ssml

    async def _stream_audio_chunks(self, audio_bytes: bytes) -> None:
        chunk_size = int(self.FIXED_SAMPLE_RATE * 2 * 20 / 1000)
        for i in range(0, len(audio_bytes), chunk_size):
            if self._interrupted: break
            chunk = audio_bytes[i : i + chunk_size]
            if len(chunk) < chunk_size and len(chunk) > 0:
                padding_needed = chunk_size - len(chunk)
                chunk += b"\x00" * padding_needed
            if len(chunk) == chunk_size:
                if not self._first_chunk_sent and self._first_audio_callback:
                    self._first_chunk_sent = True
                    await self._first_audio_callback()
                if self.audio_track:
                    asyncio.create_task(self.audio_track.add_new_bytes(chunk))
                await asyncio.sleep(0.001)

    async def interrupt(self) -> None:
        self._interrupted = True
        if self.audio_track:
            self.audio_track.interrupt()

    async def aclose(self) -> None:
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        await super().aclose()