from __future__ import annotations

from typing import Any, AsyncIterator, Literal, Optional, Union
import os
import asyncio
import logging
import numpy as np
from dataclasses import dataclass

from videosdk.agents import TTS, segment_text

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

try:
    from scipy import signal

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


VIDEOSDK_TTS_SAMPLE_RATE = 24000
VIDEOSDK_TTS_CHANNELS = 1

DEFAULT_VOICE = "Joanna"
DEFAULT_ENGINE = "neural"
DEFAULT_OUTPUT_FORMAT = "pcm"
DEFAULT_SAMPLE_RATE = "24000"  # AWS Polly supports 8000, 16000, 22050, 24000

logger = logging.getLogger(__name__)


@dataclass
class PollyVoiceConfig:
    """Configuration for AWS Polly voice settings"""

    voice_id: str = DEFAULT_VOICE
    engine: str = DEFAULT_ENGINE
    output_format: str = DEFAULT_OUTPUT_FORMAT
    sample_rate: str = DEFAULT_SAMPLE_RATE
    speed: float = 1.0
    pitch: float = 0.0
    volume: float = 1.0


class AWSPollyTTS(TTS):
    """
    AWS Polly TTS implementation (plug-and-play for VideoSDK Agents).
    Usage:
        tts = AWSPollyTTS()  # All config from env, or override via kwargs
    """

    def __init__(
        self,
        *,
        voice: str = "Joanna",
        engine: str = "neural",
        region: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        speed: float = 1.0,
        pitch: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(sample_rate=VIDEOSDK_TTS_SAMPLE_RATE,
                         num_channels=VIDEOSDK_TTS_CHANNELS)
        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is not installed. Please install it with 'pip install boto3'")
        self.voice = voice
        self.engine = engine
        self.region = region or os.getenv("AWS_DEFAULT_REGION")
        self.aws_access_key_id = aws_access_key_id or os.getenv(
            "AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = aws_secret_access_key or os.getenv(
            "AWS_SECRET_ACCESS_KEY")
        self.aws_session_token = aws_session_token or os.getenv(
            "AWS_SESSION_TOKEN")
        self.speed = speed
        self.pitch = pitch
        self._first_chunk_sent = False
        if not self.region:
            raise ValueError(
                "AWS region must be specified via parameter or AWS_DEFAULT_REGION env var")
        if not self.aws_access_key_id or not self.aws_secret_access_key:
            raise ValueError(
                "AWS credentials must be provided or set as environment variables.")
        client_kwargs = {
            "service_name": "polly",
            "region_name": self.region,
            "aws_access_key_id": self.aws_access_key_id,
            "aws_secret_access_key": self.aws_secret_access_key,
        }
        if self.aws_session_token:
            client_kwargs["aws_session_token"] = self.aws_session_token
        self._client = boto3.client(**client_kwargs)

    def reset_first_audio_tracking(self) -> None:
        """Reset the first audio tracking state for next TTS task"""
        self._first_chunk_sent = False

    async def synthesize(self, text_or_generator: Union[str, AsyncIterator[str]], **kwargs) -> None:
        if not self.audio_track or not self.loop:
            logger.error("Audio track or event loop not initialized.")
            return

        try:
            if isinstance(text_or_generator, str):
                await self._process_text_segment(text_or_generator)
            else:
                async for segment in segment_text(text_or_generator):
                    await self._process_text_segment(segment)

        except (BotoCoreError, ClientError) as e:
            logger.error(f"AWS Polly API error: {e}")
        except Exception as e:
            logger.error(f"Error in AWSPollyTTS synthesis: {e}")

    async def _process_text_segment(self, text_segment: str) -> None:
        """Process individual text segments for streaming TTS"""
        if not text_segment.strip():
            return

        ssml_text = self._build_ssml(text_segment)

        response = await asyncio.to_thread(
            self._client.synthesize_speech,
            Text=ssml_text,
            TextType="ssml",
            OutputFormat="pcm",
            VoiceId=self.voice,
            SampleRate="16000",
            Engine=self.engine
        )

        audio_stream = response.get("AudioStream")
        if audio_stream:
            audio_data = await asyncio.to_thread(audio_stream.read)
            await self._stream_audio(audio_data)

    async def _stream_audio(self, audio_data: bytes):
        if not audio_data:
            return

        try:
            if SCIPY_AVAILABLE:
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                target_length = int(len(audio_array) * 24000 / 16000)
                resampled_audio = signal.resample(audio_array, target_length)

                resampled_audio = np.clip(
                    resampled_audio, -32768, 32767).astype(np.int16)
                audio_data = resampled_audio.tobytes()

                logger.debug(
                    f"Resampled audio from {len(audio_array)} to {len(resampled_audio)} samples")
            else:
                logger.warning(
                    "scipy not available, using original audio without resampling")

            chunk_size = int(self.sample_rate *
                             self.num_channels * 2 * 20 / 1000)

            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i : i + chunk_size]
                if len(chunk) < chunk_size:
                    chunk += b'\x00' * (chunk_size - len(chunk))

                if self.audio_track and self.loop:
                    if not self._first_chunk_sent and self._first_audio_callback:
                        self._first_chunk_sent = True
                        await self._first_audio_callback()

                    asyncio.create_task(self.audio_track.add_new_bytes(chunk))
                    await asyncio.sleep(0.01)

        except Exception as e:
            logger.error(f"Error in audio streaming: {e}")
            chunk_size = int(self.sample_rate *
                             self.num_channels * 2 * 20 / 1000)

            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i : i + chunk_size]
                if len(chunk) < chunk_size:
                    chunk += b'\x00' * (chunk_size - len(chunk))

                if self.audio_track and self.loop:
                    if not self._first_chunk_sent and self._first_audio_callback:
                        self._first_chunk_sent = True
                        await self._first_audio_callback()

                    self.loop.create_task(
                        self.audio_track.add_new_bytes(chunk))
                    await asyncio.sleep(0.001)

    def _build_ssml(self, text: str) -> str:
        """Build SSML for AWS Polly with speed and pitch controls"""
        text = text.replace("&", "&amp;").replace(
            "<", "&lt;").replace(">", "&gt;")

        ssml_parts = ["<speak>"]

        if self.speed != 1.0:
            if self.speed <= 0.5:
                rate = "x-slow"
            elif self.speed <= 0.75:
                rate = "slow"
            elif self.speed <= 1.25:
                rate = "medium"
            elif self.speed <= 1.5:
                rate = "fast"
            else:
                rate = "x-fast"

            rate_percent = f"{int(self.speed * 100)}%"
            ssml_parts.append(f'<prosody rate="{rate_percent}">')

        if self.pitch != 0.0:
            pitch_value = f"{int(self.pitch * 100)}%"
            ssml_parts.append(f'<prosody pitch="{pitch_value}">')

        ssml_parts.append(text)

        if self.pitch != 0.0:
            ssml_parts.append("</prosody>")
        if self.speed != 1.0:
            ssml_parts.append("</prosody>")

        ssml_parts.append("</speak>")

        return "".join(ssml_parts)

    async def aclose(self):
        """Close the TTS connection"""
        pass

    async def interrupt(self) -> None:
        """Interrupt the TTS audio stream"""
        if self.audio_track:
            self.audio_track.interrupt()
