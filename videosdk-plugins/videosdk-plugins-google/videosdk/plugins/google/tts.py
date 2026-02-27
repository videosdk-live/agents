from __future__ import annotations

from typing import Any, AsyncIterator, Literal
import os
import asyncio
from dataclasses import dataclass
import logging

from videosdk.agents import TTS, segment_text
logger = logging.getLogger(__name__)

GOOGLE_SAMPLE_RATE = 24000
GOOGLE_CHANNELS = 1


@dataclass
class GoogleVoiceConfig:
    languageCode: str = "en-US"
    name: str = "en-US-Chirp3-HD-Aoede"
    ssmlGender: str = "FEMALE"


@dataclass
class VertexAIConfig:
    project_id: str | None = None
    location: str = "us-central1"


class GoogleTTS(TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        speed: float = 1.0,
        pitch: float = 0.0,
        response_format: Literal["pcm"] = "pcm",
        voice_config: GoogleVoiceConfig | None = None,
        custom_pronunciations: list[dict] | dict | None = None,
        vertexai: bool = False,
        vertexai_config: VertexAIConfig | None = None,
        streaming: bool = True,
    ) -> None:
        """Initialize the Google TTS plugin.

        Args:
            api_key (Optional[str], optional): Google API key. Defaults to None.
            speed (float): The speed to use for the TTS plugin. Defaults to 1.0.
            pitch (float): The pitch to use for the TTS plugin. Defaults to 0.0.
            response_format (Literal["pcm"]): The response format to use for the TTS plugin. Defaults to "pcm".
            voice_config (GoogleVoiceConfig | None): The voice configuration to use for the TTS plugin. Defaults to None.
            custom_pronunciations: IPA pronunciation overrides,
                                   e.g. [{"tomato": "təˈmeɪtoʊ"}].
            vertexai: Use Vertex AI TTS endpoint with ADC authentication.
            vertexai_config: Project / location settings for Vertex AI.
            streaming: Use gRPC StreamingSynthesize for lower latency.
                       Compatible with vertexai=True — routes over gRPC to the regional endpoint.

        Requires: pip install google-cloud-texttospeech
        """
        super().__init__(sample_rate=GOOGLE_SAMPLE_RATE, num_channels=GOOGLE_CHANNELS)

        try:
            from google.cloud import texttospeech_v1
        except ImportError as exc:
            raise ImportError(
                "google-cloud-texttospeech is required. "
                "Install it with: pip install google-cloud-texttospeech"
            ) from exc

        self._tts = texttospeech_v1

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
        self.streaming = streaming
        if self.streaming and self.vertexai:
            raise ValueError("Streaming and vertexai cannot be used together.")
        resolved_voice = (voice_config or GoogleVoiceConfig()).name
        if streaming and not self._is_chirp3_hd_voice(resolved_voice):
            raise ValueError(
                f"Streaming synthesis only supports Chirp 3 HD voices "
                f"(e.g. 'en-US-Chirp3-HD-Aoede'). "
                f"Got: '{resolved_voice}'. "
                f"See https://cloud.google.com/text-to-speech/docs/chirp3-hd for available voices."
            )

        self._client = self._build_client(api_key)

    @staticmethod
    def _is_chirp3_hd_voice(name: str) -> bool:
        return "chirp3-hd" in name.lower()

    def _build_client(self, api_key: str | None) -> Any:
        """Construct a TextToSpeechAsyncClient."""
        from google.api_core.client_options import ClientOptions

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
                        creds = service_account.Credentials.from_service_account_file(
                            service_account_path
                        )
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

            return self._tts.TextToSpeechAsyncClient(
                client_options=ClientOptions(
                    api_endpoint=f"{location}-texttospeech.googleapis.com"
                )
            )

        else:
            resolved_key = api_key or os.getenv("GOOGLE_API_KEY")
            if not resolved_key:
                raise ValueError(
                    "Google TTS API key required. Provide either:\n"
                    "1. api_key parameter, OR\n"
                    "2. GOOGLE_API_KEY environment variable"
                )
            return self._tts.TextToSpeechAsyncClient(
                client_options=ClientOptions(api_key=resolved_key)
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
            if self.streaming:
                await self._synthesize_streaming(text)
            elif isinstance(text, AsyncIterator):
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
        """Single-request synthesis via SynthesizeSpeech."""
        tts = self._tts

        if self.custom_pronunciations:
            synthesis_input = tts.SynthesisInput(
                text=text,
                custom_pronunciations=self._build_custom_pronunciations_proto(),
            )
        else:
            synthesis_input = tts.SynthesisInput(text=text)
        is_studio = self.voice_config.name.startswith("en-US-Studio")

        voice_params = tts.VoiceSelectionParams(
            language_code=self.voice_config.languageCode,
            name=self.voice_config.name,
        )
        if not is_studio:
            voice_params.ssml_gender = tts.SsmlVoiceGender[self.voice_config.ssmlGender]
        response = await self._client.synthesize_speech(
            input=synthesis_input,
            voice=voice_params,
            audio_config=tts.AudioConfig(
                audio_encoding=tts.AudioEncoding.LINEAR16,
                speaking_rate=self.speed,
                pitch=self.pitch,
                sample_rate_hertz=GOOGLE_SAMPLE_RATE,
            ),
        )

        if not response.audio_content:
            self.emit("error", "No audio content received from Google TTS")
            return

        await self._stream_audio_chunks(response.audio_content)

    async def _synthesize_streaming(self, text: AsyncIterator[str] | str) -> None:
        """Bidirectional gRPC streaming via StreamingSynthesize."""
        tts = self._tts

        streaming_config_kwargs: dict = dict(
            voice=tts.VoiceSelectionParams(
                language_code=self.voice_config.languageCode,
                name=self.voice_config.name,
            ),
            streaming_audio_config=tts.StreamingAudioConfig(
                audio_encoding=tts.AudioEncoding.PCM,
                sample_rate_hertz=GOOGLE_SAMPLE_RATE,
                speaking_rate=self.speed,
            ),
        )
        if self.custom_pronunciations:
            streaming_config_kwargs["custom_pronunciations"] = (
                self._build_custom_pronunciations_proto()
            )

        streaming_config = tts.StreamingSynthesizeConfig(**streaming_config_kwargs)

        async def request_generator() -> AsyncIterator[Any]:
            yield tts.StreamingSynthesizeRequest(streaming_config=streaming_config)
            if isinstance(text, str):
                yield tts.StreamingSynthesizeRequest(
                    input=tts.StreamingSynthesisInput(text=text)
                )
            else:
                async for chunk in text:
                    if chunk:
                        yield tts.StreamingSynthesizeRequest(
                            input=tts.StreamingSynthesisInput(text=chunk)
                        )

        try:
            async for response in await self._client.streaming_synthesize(
                request_generator()
            ):
                if response.audio_content:
                    await self._stream_audio_chunks(response.audio_content, has_wav_header=False)
        except Exception as e:
            self.emit("error", f"Google TTS streaming error: {str(e)}")
            raise

    def _build_custom_pronunciations_proto(self) -> Any:
        """Convert self.custom_pronunciations to a CustomPronunciations proto."""
        tts = self._tts
        params = []
        try:
            from google.cloud.texttospeech_v1.types import CustomPronunciationParams as _CPP
            PE = _CPP.PhoneticEncoding
            ENCODING_MAP = {
                "ipa":    PE.PHONETIC_ENCODING_IPA,
                "x-sampa": PE.PHONETIC_ENCODING_X_SAMPA,
            }
        except (ImportError, AttributeError):
            ENCODING_MAP = {"ipa": 1, "x-sampa": 2}

        if not self.custom_pronunciations:
            return tts.CustomPronunciations(pronunciations=[])

        raw = self.custom_pronunciations
        entries: list[tuple[str, str, Any]] = []

        if isinstance(raw, dict):
            for phrase, pronunciation in raw.items():
                entries.append((phrase, pronunciation, ENCODING_MAP["ipa"]))
        else:
            for item in raw:
                if not isinstance(item, dict):
                    continue
                if "phrase" in item and "pronunciation" in item:
                    enc_key = item.get("encoding", "ipa").lower()
                    enc = ENCODING_MAP.get(enc_key, ENCODING_MAP["ipa"])
                    if enc_key not in ENCODING_MAP:
                        logger.warning(
                            f"Unknown encoding '{enc_key}' for phrase '{item['phrase']}'. "
                            f"Supported: {list(ENCODING_MAP.keys())}. Falling back to IPA.",
                            UserWarning, stacklevel=3,
                        )
                    entries.append((item["phrase"], item["pronunciation"], enc))
                else:
                    for phrase, pronunciation in item.items():
                        entries.append((phrase, pronunciation, ENCODING_MAP["ipa"]))


        if self.voice_config.languageCode.lower() != "en-us":
            logger.warning(
                f"custom_pronunciations is only supported for en-US. "
                f"Got '{self.voice_config.languageCode}' — pronunciations will be ignored.",
                UserWarning,
                stacklevel=3,
            )

        for phrase, pronunciation, encoding in entries:
            if not phrase or not pronunciation:
                continue
            try:
                params.append(
                    tts.CustomPronunciationParams(
                        phrase=phrase,
                        pronunciation=pronunciation,
                        phonetic_encoding=encoding,
                    )
                )
            except Exception as e:
                logger.warning(
                    f"Skipping custom pronunciation for '{phrase}': {e}",
                    UserWarning,
                    stacklevel=3,
                )

        if not params:
            logger.warning(
                "custom_pronunciations was set but no valid entries were built. "
                "Check your phrase/pronunciation format.",
                UserWarning,
                stacklevel=3,
            )

        return tts.CustomPronunciations(pronunciations=params)


    async def _stream_audio_chunks(
        self, audio_bytes: bytes, has_wav_header: bool = True
    ) -> None:
        """Chunk raw PCM and forward to the audio track."""
        chunk_size = 960
        audio_data = self._remove_wav_header(audio_bytes) if has_wav_header else audio_bytes

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
        if self._client:
            await self._client.transport.close()
        await super().aclose()

    async def interrupt(self) -> None:
        if self.audio_track:
            self.audio_track.interrupt()
