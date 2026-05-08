from __future__ import annotations

from typing import Any, AsyncIterator, Literal, Optional, Union
import os
import httpx
import io
import asyncio

from pydub import AudioSegment

from videosdk.agents import TTS, FlushMarker

SPEECHIFY_SAMPLE_RATE = 24000
SPEECHIFY_CHANNELS = 1
SPEECHIFY_STREAM_ENDPOINT = "https://api.sws.speechify.com/v1/audio/stream"


class SpeechifyTTS(TTS):
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        voice_id: str = "kristy",
        model: Literal[
            "simba-base", "simba-english", "simba-multilingual", "simba-turbo"
        ] = "simba-english",
        language: Optional[str] = None,
        audio_format: Literal["mp3", "ogg", "aac"] = "mp3",
        loudness_normalization: Optional[bool] = None,
        text_normalization: Optional[bool] = None,
        chunked_synthesis: bool = False,
    ) -> None:
        """Initialize the Speechify TTS plugin.

        Args:
            api_key (Optional[str], optional): Speechify API key. Defaults to None.
            voice_id (str): The voice ID to use for the TTS plugin. Defaults to "kristy".
            model (Literal["simba-base", "simba-english", "simba-multilingual", "simba-turbo"]): The model to use for the TTS plugin. Defaults to "simba-english".
            language (Optional[str], optional): The language to use for the TTS plugin. Defaults to None.
            audio_format (Literal["mp3", "ogg", "aac"]): The audio format to use for the TTS plugin. Defaults to "mp3".
            loudness_normalization (Optional[bool]): Forwarded as ``options.loudness_normalization``
                in the request body. ``None`` lets the server apply its default.
            text_normalization (Optional[bool]): Forwarded as ``options.text_normalization``
                in the request body. ``None`` lets the server apply its default.
            chunked_synthesis (bool): When ``True``, dispatches one POST per
                ``FlushMarker`` segment boundary. When ``False`` (default), the
                entire LLM stream is accumulated into a single POST for prosody
                continuity. Set ``True`` only for very long utterances where
                sub-sentence TTFB matters more than cross-sentence prosody.
        """
        super().__init__(
            sample_rate=SPEECHIFY_SAMPLE_RATE, num_channels=SPEECHIFY_CHANNELS
        )

        self.voice_id = voice_id
        self.model = model
        self.language = language
        self.audio_format = audio_format
        self.loudness_normalization = loudness_normalization
        self.text_normalization = text_normalization
        self.chunked_synthesis = chunked_synthesis
        self.audio_track = None
        self.loop = None
        self._first_chunk_sent = False
        self._current_synthesis_task: asyncio.Task | None = None
        self._interrupted = False

        self.api_key = api_key or os.getenv("SPEECHIFY_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Speechify API key required. Provide either:\n"
                "1. api_key parameter, OR\n"
                "2. SPEECHIFY_API_KEY environment variable"
            )

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
        text: AsyncIterator[Union[str, FlushMarker]] | str,
        **kwargs: Any,
    ) -> None:
        try:
            if not self.audio_track or not self.loop:
                self.emit("error", "Audio track or loop not initialized")
                return

            self._interrupted = False

            if isinstance(text, str):
                if not self._interrupted:
                    await self._stream_synthesis(text)
                return

            if self.chunked_synthesis:
                # One POST per FlushMarker boundary.
                buf: list[str] = []
                async for chunk in text:
                    if self._interrupted:
                        break
                    if isinstance(chunk, FlushMarker):
                        if buf:
                            combined = "".join(buf)
                            buf = []
                            if combined.strip():
                                await self._stream_synthesis(combined)
                        continue
                    if chunk and chunk.strip():
                        buf.append(chunk)
                if buf and not self._interrupted:
                    tail = "".join(buf)
                    if tail.strip():
                        await self._stream_synthesis(tail)
                return

            # Default: accumulate full stream into one POST. FlushMarkers dropped.
            parts: list[str] = []
            async for chunk in text:
                if self._interrupted:
                    break
                if isinstance(chunk, FlushMarker):
                    continue
                if chunk and chunk.strip():
                    parts.append(chunk)
            if parts and not self._interrupted:
                combined_text = "".join(parts)
                if combined_text.strip():
                    await self._stream_synthesis(combined_text)

        except Exception as e:
            self.emit("error", f"Speechify TTS synthesis failed: {str(e)}")

    async def _stream_synthesis(self, text: str) -> None:
        """Synthesize text to speech using Speechify stream endpoint"""
        if not text.strip() or self._interrupted:
            return

        headers = {
            "Accept": f"audio/{self.audio_format}",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "input": text,
            "voice_id": self.voice_id,
            "model": self.model,
        }

        if self.language:
            payload["language"] = self.language

        # Server applies its own defaults when omitted.
        options: dict[str, Any] = {}
        if self.loudness_normalization is not None:
            options["loudness_normalization"] = self.loudness_normalization
        if self.text_normalization is not None:
            options["text_normalization"] = self.text_normalization
        if options:
            payload["options"] = options

        for attempt in range(2):
            try:
                async with self._http_client.stream(
                    "POST",
                    SPEECHIFY_STREAM_ENDPOINT,
                    headers=headers,
                    json=payload
                ) as response:
                    response.raise_for_status()

                    audio_data = b""
                    async for chunk in response.aiter_bytes():
                        if self._interrupted:
                            return
                        if chunk:
                            audio_data += chunk

                    if audio_data and not self._interrupted:
                        await self._decode_and_stream(audio_data)
                return

            except (httpx.NetworkError, httpx.ConnectError, httpx.ReadTimeout) as e:
                if attempt == 0 and not self._interrupted:
                    await asyncio.sleep(0.25 * (2 ** attempt))
                    continue
                if not self._interrupted:
                    self.emit("error", f"Speechify TTS network failure: {str(e)}")
                return
            except httpx.HTTPStatusError as e:
                if not self._interrupted:
                    error_msg = f"HTTP error {e.response.status_code}"
                    try:
                        error_data = e.response.json()
                        if isinstance(error_data, dict) and "error" in error_data:
                            error_msg = f"{error_msg}: {error_data['error']}"
                    except Exception:
                        pass
                    self.emit(
                        "error", f"Speechify stream synthesis failed: {error_msg}")
                return
            except Exception as e:
                if not self._interrupted:
                    self.emit("error", f"Stream synthesis failed: {str(e)}")
                return

    async def _decode_and_stream(self, audio_bytes: bytes) -> None:
        """Decode compressed audio to PCM and stream it"""
        if self._interrupted or not audio_bytes:
            return

        try:
            audio = AudioSegment.from_file(
                io.BytesIO(audio_bytes),
                format=self.audio_format
            )

            audio = audio.set_frame_rate(SPEECHIFY_SAMPLE_RATE)
            audio = audio.set_channels(SPEECHIFY_CHANNELS)
            audio = audio.set_sample_width(2)

            pcm_data = audio.raw_data

            chunk_size = int(SPEECHIFY_SAMPLE_RATE *
                             SPEECHIFY_CHANNELS * 2 * 20 / 1000)  # 20ms chunks

            for i in range(0, len(pcm_data), chunk_size):
                if self._interrupted:
                    break

                chunk = pcm_data[i:i + chunk_size]

                if len(chunk) < chunk_size and len(chunk) > 0:
                    padding_needed = chunk_size - len(chunk)
                    chunk += b'\x00' * padding_needed

                if len(chunk) == chunk_size:
                    if not self._first_chunk_sent and self._first_audio_callback:
                        self._first_chunk_sent = True
                        await self._first_audio_callback()

                    asyncio.create_task(
                        self.audio_track.add_new_bytes(chunk))
                    await asyncio.sleep(0.001)

        except Exception as e:
            if not self._interrupted:
                self.emit("error", f"Audio decoding failed: {str(e)}")

    async def aclose(self) -> None:
        if self._http_client:
            await self._http_client.aclose()
        await super().aclose()

    async def interrupt(self) -> None:
        """Interrupt TTS synthesis"""
        self._interrupted = True
        if self._current_synthesis_task and not self._current_synthesis_task.done():
            self._current_synthesis_task.cancel()
        if self.audio_track:
            self.audio_track.interrupt()
