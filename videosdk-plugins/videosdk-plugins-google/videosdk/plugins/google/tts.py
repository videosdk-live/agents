from __future__ import annotations

from typing import Any, AsyncIterator, Literal
import os
import asyncio
import base64
import httpx
from dataclasses import dataclass, field

from videosdk.agents import TTS, segment_text

GOOGLE_SAMPLE_RATE = 24000
GOOGLE_CHANNELS = 1
GOOGLE_TTS_ENDPOINT = "https://texttospeech.googleapis.com/v1/text:synthesize"

VERTEXAI_TTS_ENDPOINT = "https://texttospeech.googleapis.com/v1beta1/text:synthesize"


@dataclass
class GoogleVoiceConfig:
    languageCode: str = "en-US"
    name: str = "en-US-Chirp3-HD-Aoede"
    ssmlGender: str = "FEMALE"


@dataclass
class VertexAIConfig:
    project_id: str | None = None
    location: str = "us-central1"

def apply_custom_pronunciations_ssml(
    text: str,
    custom_prons: list[dict[str, str]],
) -> str:
    """Wrap text in SSML and replace known phrases with <phoneme> tags."""
    try:
        if not text:
            return "<speak></speak>"
        if not custom_prons:
            return f"<speak>{text}</speak>"

        for item in custom_prons:
            if not isinstance(item, dict):
                continue
            for phrase, pronunciation in item.items():
                if not phrase or not pronunciation:
                    continue
                try:
                    safe_pron = pronunciation.replace('"', "&quot;")
                    text = text.replace(
                        phrase,
                        f'<phoneme alphabet="ipa" ph="{safe_pron}">{phrase}</phoneme>',
                    )
                except Exception as e:
                    continue

        return f"<speak>{text}</speak>"

    except Exception as e:
        return f"<speak>{text}</speak>"


def _get_vertexai_token_sync() -> str:
    """Obtain and refresh an ADC bearer token."""
    try:
        import google.auth
        import google.auth.transport.requests as google_requests
    except ImportError as exc:
        raise ImportError(
            "google-auth is required for Vertex AI support. "
            "Install it with: pip install google-auth"
        ) from exc

    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    request = google_requests.Request()
    credentials.refresh(request)

    if not credentials.token:
        raise RuntimeError(
            "ADC token is empty after refresh. "
            "Ensure GOOGLE_APPLICATION_CREDENTIALS is set or run "
            "'gcloud auth application-default login'."
        )
    return credentials.token


async def _get_vertexai_token() -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _get_vertexai_token_sync)

class GoogleTTS(TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        speed: float = 1.0,
        pitch: float = 0.0,
        response_format: Literal["pcm"] = "pcm",
        voice_config: GoogleVoiceConfig | None = None,
        custom_pronunciations: list[dict] | None = None,
        vertexai: bool = False,
        vertexai_config: VertexAIConfig | None = None,
    ) -> None:
        """Initialize the Google TTS plugin.

        Args:
            api_key: Google API key (standard Cloud TTS only; not used for Vertex AI).
            speed: Speaking rate multiplier. Defaults to 1.0.
            pitch: Pitch shift in semitones. Defaults to 0.0.
            response_format: Audio format returned. Currently only "pcm" is supported.
            voice_config: Voice language / name / gender settings.
            custom_pronunciations (list): Use IPA chart for pronunctiations.
            vertexai: Use Vertex AI TTS instead of the standard Cloud TTS API.
            vertexai_config: Project / location settings for Vertex AI.
        """
        super().__init__(sample_rate=GOOGLE_SAMPLE_RATE, num_channels=GOOGLE_CHANNELS)

        self.speed = speed
        self.pitch = pitch
        self.response_format = response_format
        self.audio_track = None
        self.loop = None
        self._first_chunk_sent = False
        self.voice_config = voice_config or GoogleVoiceConfig()
        self.custom_pronunciations = custom_pronunciations
        self.vertexai = vertexai
        self.vertexai_config = vertexai_config or VertexAIConfig()

        if self.vertexai:
            project_id = (
                self.vertexai_config.project_id
                or os.getenv("GOOGLE_CLOUD_PROJECT")
                or os.getenv("GCLOUD_PROJECT")
            )

            if project_id is None:
                service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
                if service_account_path:
                    try:
                        from google.oauth2 import service_account
                        creds = service_account.Credentials.from_service_account_file(service_account_path)
                        project_id = creds.project_id
                    except Exception:
                        pass

            if project_id is None:
                raise ValueError(
                    "Vertex AI TTS requires a GCP project ID. Provide one of:\n"
                    "1. vertexai_config=VertexAIConfig(project_id='my-project')\n"
                    "2. GOOGLE_CLOUD_PROJECT environment variable\n"
                    "3. GOOGLE_APPLICATION_CREDENTIALS pointing to a service-account file"
                )
            location = (
                self.vertexai_config.location
                or os.getenv("GOOGLE_CLOUD_LOCATION")
                or "us-central1"
            )
            self.vertexai_config.project_id = project_id
            self.vertexai_config.location = location
            self.api_key = None

        else:
            self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

            if not self.api_key:
                raise ValueError(
                    "Google TTS API key required. Provide either:\n"
                    "1. api_key parameter, OR\n"
                    "2. GOOGLE_API_KEY environment variable"
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
            if isinstance(text, AsyncIterator):
                async for segment in segment_text(text):
                    await self._synthesize_audio(segment)
            else:
                await self._synthesize_audio(text)

            if not self.audio_track or not self.loop:
                self.emit("error", "Audio track or loop not initialized")
                return

        except Exception as e:
            self.emit("error", f"Google TTS synthesis failed: {str(e)}")
            raise

    async def _synthesize_audio(self, text: str) -> None:
        """Route synthesis to Vertex AI or standard Cloud TTS."""
        if self.vertexai:
            await self._synthesize_vertexai(text)
        else:
            await self._synthesize_standard(text)

    async def _synthesize_standard(self, text: str) -> None:
        """Synthesize using the standard Google Cloud TTS REST API."""
        try:
            payload = {
                "input": self._build_input_payload(text),
                "voice": self._build_voice_config(),
                "audioConfig": {
                    "audioEncoding": "LINEAR16",
                    "speakingRate": self.speed,
                    "pitch": self.pitch,
                    "sampleRateHertz": GOOGLE_SAMPLE_RATE,
                },
            }

            response = await self._http_client.post(
                GOOGLE_TTS_ENDPOINT, params={"key": self.api_key}, json=payload
            )
            
            response.raise_for_status()
            await self._handle_response(response)

        except httpx.HTTPStatusError as e:
            self._handle_http_error(e)
            raise

    async def _synthesize_vertexai(self, text: str) -> None:
        """Synthesize using the Vertex AI TTS API (ADC authentication)."""
        try:
            token = await _get_vertexai_token()

            payload = {
                "input": self._build_input_payload(text),
                "voice": self._build_voice_config(),
                "audioConfig": {
                    "audioEncoding": "LINEAR16",
                    "speakingRate": self.speed,
                    "pitch": self.pitch,
                    "sampleRateHertz": GOOGLE_SAMPLE_RATE,
                },
            }
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }

            response = await self._http_client.post(
                VERTEXAI_TTS_ENDPOINT,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()

            await self._handle_response(response)

        except httpx.HTTPStatusError as e:
            self._handle_http_error(e)
            raise


    def _build_input_payload(self, text: str) -> dict:
        """Return either an SSML or plain-text input dict."""
        if self.custom_pronunciations:
            ssml_text = apply_custom_pronunciations_ssml(text, self.custom_pronunciations)
            return {"ssml": ssml_text}
        return {"text": text}

    def _build_voice_config(self) -> dict:
        cfg = {
            "languageCode": self.voice_config.languageCode,
            "name": self.voice_config.name,
        }
        if not self.voice_config.name.startswith("en-US-Studio"):
            cfg["ssmlGender"] = self.voice_config.ssmlGender
        return cfg

    async def _handle_response(self, response: httpx.Response) -> None:
        response_data = response.json()
        audio_content = response_data.get("audioContent")
        if not audio_content:
            self.emit("error", "No audio content received from Google TTS")
            return

        audio_bytes = base64.b64decode(audio_content)

        if not audio_bytes:
            self.emit("error", "Decoded audio bytes are empty")
            return

        await self._stream_audio_chunks(audio_bytes)

    def _handle_http_error(self, e: httpx.HTTPStatusError) -> None:
        status = e.response.status_code
        try:
            body = e.response.json()
        except Exception:
            body = e.response.text

        if status == 401:
            self.emit("error", f"Google TTS authentication failed. Detail: {body}")
        elif status == 403:
            self.emit("error", f"Google TTS permission denied (403). Detail: {body}")
        elif status == 400:
            error_msg = (
                body.get("error", {}).get("message", "Bad request")
                if isinstance(body, dict)
                else body
            )
            self.emit("error", f"Google TTS request error: {error_msg}")
        else:
            self.emit("error", f"Google TTS HTTP error: {status}")

    async def _stream_audio_chunks(self, audio_bytes: bytes) -> None:
        """Stream audio data in chunks to avoid beeps and ensure smooth playback"""
        chunk_size = 960
        audio_data = self._remove_wav_header(audio_bytes)

        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]

            if len(chunk) < chunk_size and len(chunk) > 0:
                padding_needed = chunk_size - len(chunk)
                chunk += b'\x00' * padding_needed

            if len(chunk) == chunk_size:
                if not self._first_chunk_sent and self._first_audio_callback:
                    self._first_chunk_sent = True
                    await self._first_audio_callback()

                asyncio.create_task(self.audio_track.add_new_bytes(chunk))
                await asyncio.sleep(0.001)

    def _remove_wav_header(self, audio_bytes: bytes) -> bytes:
        """Remove WAV header if present to get raw PCM data"""
        if audio_bytes.startswith(b"RIFF"):
            data_pos = audio_bytes.find(b"data")
            if data_pos != -1:
                return audio_bytes[data_pos + 8:]

        return audio_bytes

    async def aclose(self) -> None:
        if self._http_client:
            await self._http_client.aclose()
        await super().aclose()

    async def interrupt(self) -> None:
        if self.audio_track:
            self.audio_track.interrupt()
