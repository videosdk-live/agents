from __future__ import annotations

from typing import Any, AsyncIterator, Literal, Optional, Union
import httpx
import os
import openai
import asyncio

from videosdk.agents import TTS

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
        voice: str | dict[str, str] = DEFAULT_VOICE,
        speed: float = 1.0,
        instructions: str | None = None,
        language: str | None = None,
        base_url: str | None = None,
        response_format: str = "pcm",
    ) -> None:
        """Initialize the OpenAI TTS plugin.

        Args:
            api_key (Optional[str], optional): OpenAI API key. Defaults to None.
            model (str): The model to use for the TTS plugin. Defaults to "gpt-4o-mini-tts".
                Built-in options: "gpt-4o-mini-tts" (recommended, supports instructions),
                "tts-1" (low latency), "tts-1-hd" (higher quality).
            voice (str | dict): Built-in voice name (e.g. "marin", "cedar", "ash", "coral")
                or a custom voice reference dict {"id": "voice_xxx"}. Defaults to "ash".
                For best quality with gpt-4o-mini-tts, use "marin" or "cedar".
            speed (float): The speed to use for the TTS plugin. Defaults to 1.0.
            instructions (Optional[str], optional): Natural-language style control
                ("Speak in a cheerful tone", accent hints, etc.). Only honored by
                gpt-4o-mini-tts; ignored by tts-1 / tts-1-hd. Defaults to None.
            language (Optional[str], optional): ISO language hint (e.g. "hi", "mr", "fr").
                Useful for non-English input or with custom voices. Defaults to None.
            base_url (Optional[str], optional): Custom base URL for the OpenAI API. Defaults to None.
            response_format (str): The response format to use for the TTS plugin. Defaults to "pcm".
        """
        super().__init__(sample_rate=OPENAI_TTS_SAMPLE_RATE, num_channels=OPENAI_TTS_CHANNELS)

        self.model = model
        self.voice = voice
        self.speed = speed
        self.instructions = instructions
        self.language = language
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
        voice: str | dict[str, str] = DEFAULT_VOICE,
        speed: float = 1.0,
        instructions: str | None = None,
        language: str | None = None,
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
            language=language,
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
        voice_id: Optional[str | dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Convert text to speech using OpenAI's TTS API and stream to audio track.

        For ``AsyncIterator`` inputs, all chunks are accumulated and posted as a
        single request. The upstream tokenizer / text filter already delivers
        sentence-sized, verbalized segments; client-side re-segmentation here
        would split currency tokens (``$50,000``, ``₹50,00,000``) and
        comma-grouped digits mid-token, and per-segment API calls produce
        discontinuous prosody at chunk boundaries.

        Args:
            text: Text to convert to speech
            voice_id: Optional voice override
            **kwargs: Additional provider-specific arguments
        """
        try:
            if not self.audio_track or not self.loop:
                self.emit("error", "Audio track or event loop not set")
                raise RuntimeError("Audio track or event loop not set")

            self._interrupted = False

            if isinstance(text, AsyncIterator):
                parts: list[str] = []
                async for chunk in text:
                    if self._interrupted:
                        break
                    if chunk and chunk.strip():
                        parts.append(chunk)
                if parts and not self._interrupted:
                    combined_text = "".join(parts)
                    if combined_text.strip():
                        await self._synthesize_segment(combined_text, voice_id, **kwargs)
            else:
                if not self._interrupted:
                    await self._synthesize_segment(text, voice_id, **kwargs)

        except Exception as e:
            self.emit("error", f"TTS synthesis failed: {str(e)}")
            raise

    async def _synthesize_segment(
        self,
        text: str,
        voice_id: Optional[str | dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        """Synthesize a single text segment.

        Streams audio frames to the audio track as they arrive from OpenAI's
        chunked HTTP response. Maintains a leftover buffer between iterations
        so partial bytes don't get silence-padded mid-stream — padding only
        applies to the final frame at end-of-response.
        """
        if not text.strip() or self._interrupted:
            return

        # 20ms frame @ 24kHz, 16-bit, mono = 960 bytes
        frame_size = int(
            OPENAI_TTS_SAMPLE_RATE * OPENAI_TTS_CHANNELS * 2 * 20 / 1000
        )
        leftover = bytearray()

        try:
            async with self._client.audio.speech.with_streaming_response.create(
                model=self.model,
                voice=voice_id or self.voice,
                input=text,
                speed=self.speed,
                response_format=self.response_format,
                **({"instructions": self.instructions} if self.instructions else {}),
                **({"extra_body": {"language": self.language}} if self.language else {}),
            ) as response:
                async for chunk in response.iter_bytes():
                    if self._interrupted:
                        break
                    if not chunk:
                        continue
                    leftover.extend(chunk)

                    # Emit complete 20ms frames as soon as they're available.
                    while len(leftover) >= frame_size and not self._interrupted:
                        frame = bytes(leftover[:frame_size])
                        del leftover[:frame_size]

                        if not self._first_chunk_sent and self._first_audio_callback:
                            self._first_chunk_sent = True
                            await self._first_audio_callback()

                        asyncio.create_task(self.audio_track.add_new_bytes(frame))
                        await asyncio.sleep(0.001)

            # End of stream: zero-pad the final partial frame and emit.
            if leftover and not self._interrupted:
                frame = bytes(leftover) + b"\x00" * (frame_size - len(leftover))
                if not self._first_chunk_sent and self._first_audio_callback:
                    self._first_chunk_sent = True
                    await self._first_audio_callback()
                asyncio.create_task(self.audio_track.add_new_bytes(frame))

        except Exception as e:
            if not self._interrupted:
                self.emit("error", f"Segment synthesis failed: {str(e)}")
                raise

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
