from __future__ import annotations

from typing import Any, AsyncIterator, Literal, Optional, Union
import httpx
import os
import openai
import asyncio

from videosdk.agents import TTS, segment_text

OPENAI_TTS_SAMPLE_RATE = 24000
OPENAI_TTS_CHANNELS = 1

DEFAULT_MODEL = "gpt-4o-mini-tts"
DEFAULT_VOICE = "ash"
_RESPONSE_FORMATS = Union[Literal["mp3",
                                  "opus", "aac", "flac", "wav", "pcm"], str]


class OpenAITTS(TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        voice: str = DEFAULT_VOICE,
        speed: float = 1.0,
        instructions: str | None = None,
        base_url: str | None = None,
        response_format: str = "pcm",
    ) -> None:
        """Initialize the OpenAI TTS plugin.

        Args:
            api_key (Optional[str], optional): OpenAI API key. Defaults to None.
            model (str): The model to use for the TTS plugin. Defaults to "gpt-4o-mini-tts".
            voice (str): The voice to use for the TTS plugin. Defaults to "ash".
            speed (float): The speed to use for the TTS plugin. Defaults to 1.0.
            instructions (Optional[str], optional): Additional instructions for the TTS plugin. Defaults to None.
            base_url (Optional[str], optional): Custom base URL for the OpenAI API. Defaults to None.
            response_format (str): The response format to use for the TTS plugin. Defaults to "pcm".
        """
        super().__init__(sample_rate=OPENAI_TTS_SAMPLE_RATE, num_channels=OPENAI_TTS_CHANNELS)

        self.model = model
        self.voice = voice
        self.speed = speed
        self.instructions = instructions
        self.audio_track = None
        self.loop = None
        self.response_format = response_format
        self._first_chunk_sent = False
        self._current_synthesis_task: asyncio.Task | None = None
        self._interrupted = False

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided either through api_key parameter or OPENAI_API_KEY environment variable")

        self._client = openai.AsyncClient(
            max_retries=0,
            api_key=self.api_key,
            base_url=base_url or None,
            http_client=httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=15.0, read=5.0, write=5.0, pool=5.0),
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=50,
                    max_keepalive_connections=50,
                    keepalive_expiry=120,
                ),
            ),
        )

    @staticmethod
    def azure(
        *,
        model: str = DEFAULT_MODEL,
        voice: str = DEFAULT_VOICE,
        speed: float = 1.0,
        instructions: str | None = None,
        azure_endpoint: str | None = None,
        azure_deployment: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        azure_ad_token: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        base_url: str | None = None,
        response_format: str = "pcm",
        timeout: httpx.Timeout | None = None,
    ) -> "OpenAITTS":
        """
        Create a new instance of Azure OpenAI TTS.

        This automatically infers the following arguments from their corresponding environment variables if they are not provided:
        - `api_key` from `AZURE_OPENAI_API_KEY`
        - `organization` from `OPENAI_ORG_ID`
        - `project` from `OPENAI_PROJECT_ID`
        - `azure_ad_token` from `AZURE_OPENAI_AD_TOKEN`
        - `api_version` from `OPENAI_API_VERSION`
        - `azure_endpoint` from `AZURE_OPENAI_ENDPOINT`
        - `azure_deployment` from `AZURE_OPENAI_DEPLOYMENT` (if not provided, uses `model` as deployment name)
        """
        
        azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_deployment = azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        api_version = api_version or os.getenv("OPENAI_API_VERSION")
        api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        azure_ad_token = azure_ad_token or os.getenv("AZURE_OPENAI_AD_TOKEN")
        organization = organization or os.getenv("OPENAI_ORG_ID")
        project = project or os.getenv("OPENAI_PROJECT_ID")
        
        if not azure_deployment:
            azure_deployment = model
        
        if not azure_endpoint:
            raise ValueError("Azure endpoint must be provided either through azure_endpoint parameter or AZURE_OPENAI_ENDPOINT environment variable")
        
        if not api_key and not azure_ad_token:
            raise ValueError("Either API key or Azure AD token must be provided")
        
        azure_client = openai.AsyncAzureOpenAI(
            max_retries=0,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
            api_key=api_key,
            azure_ad_token=azure_ad_token,
            organization=organization,
            project=project,
            base_url=base_url,
            timeout=timeout
            if timeout
            else httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
        )
        
        instance = OpenAITTS(
            model=model,
            voice=voice,
            speed=speed,
            instructions=instructions,
            response_format=response_format,
        )
        instance._client = azure_client
        return instance

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
        Convert text to speech using OpenAI's TTS API and stream to audio track

        Args:
            text: Text to convert to speech
            voice_id: Optional voice override
            **kwargs: Additional provider-specific arguments
        """
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
            self.emit("error", f"TTS synthesis failed: {str(e)}")

    async def _synthesize_segment(self, text: str, voice_id: Optional[str] = None, **kwargs: Any) -> None:
        """Synthesize a single text segment"""
        if not text.strip() or self._interrupted:
            return

        try:
            audio_data = b""
            async with self._client.audio.speech.with_streaming_response.create(
                model=self.model,
                voice=voice_id or self.voice,
                input=text,
                speed=self.speed,
                response_format=self.response_format,
                **({"instructions": self.instructions} if self.instructions else {}),
            ) as response:
                async for chunk in response.iter_bytes():
                    if self._interrupted:
                        break
                    if chunk:
                        audio_data += chunk

            if audio_data and not self._interrupted:
                await self._stream_audio_chunks(audio_data)

        except Exception as e:
            if not self._interrupted:
                self.emit("error", f"Segment synthesis failed: {str(e)}")

    async def _stream_audio_chunks(self, audio_bytes: bytes) -> None:
        """Stream audio data in chunks for smooth playback"""
        chunk_size = int(OPENAI_TTS_SAMPLE_RATE *
                         OPENAI_TTS_CHANNELS * 2 * 20 / 1000)

        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i + chunk_size]

            if len(chunk) < chunk_size and len(chunk) > 0:
                padding_needed = chunk_size - len(chunk)
                chunk += b'\x00' * padding_needed

            if len(chunk) == chunk_size:
                if not self._first_chunk_sent and self._first_audio_callback:
                    self._first_chunk_sent = True
                    await self._first_audio_callback()

                asyncio.create_task(self.audio_track.add_new_bytes(chunk))
                await asyncio.sleep(0.001)

    async def aclose(self) -> None:
        """Cleanup resources"""
        await self._client.close()
        await super().aclose()

    async def interrupt(self) -> None:
        """Interrupt TTS synthesis"""
        self._interrupted = True
        if self._current_synthesis_task:
            self._current_synthesis_task.cancel()
        if self.audio_track:
            self.audio_track.interrupt()
