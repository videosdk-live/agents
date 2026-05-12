from __future__ import annotations

from typing import Any, AsyncIterator, Optional, Union
import os
import asyncio
import httpx

from videosdk.agents import TTS, FlushMarker, segment_text

RIME_SAMPLE_RATE = 24000
RIME_CHANNELS = 1
RIME_TTS_ENDPOINT = "https://users.rime.ai/v1/rime-tts"

DEFAULT_MODEL = "mist"
DEFAULT_SPEAKER = "river"
DEFAULT_LANGUAGE = "eng"

# Rime's Arcana long-form model can take well over a minute to render long
# inputs; Mist family runs fast. Per-model read timeout avoids hanging shorter
# Mist requests on Arcana's worst-case while still letting Arcana finish.
MODEL_READ_TIMEOUT_SEC = {
    "arcana": 240.0,
    "mist": 30.0,
    "mistv2": 30.0,
}

KNOWN_SPEAKERS = {
    "mist": ["river", "storm", "brook", "ember", "iris", "pearl"],
    "mistv2": ["river", "storm", "brook", "ember", "iris", "pearl"]
}


class RimeTTS(TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        speaker: str = DEFAULT_SPEAKER,
        model_id: str = DEFAULT_MODEL,
        lang: str = DEFAULT_LANGUAGE,
        sampling_rate: int = RIME_SAMPLE_RATE,
        speed_alpha: float = 1.0,
        reduce_latency: bool = True,
        pause_between_brackets: bool = False,
        phonemize_between_brackets: bool = False,
        inline_speed_alpha: str | None = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        """Initialize the Rime TTS plugin.

        Args:
            api_key (Optional[str], optional): Rime AI API key. Defaults to None.
            speaker (str): The speaker to use for the TTS plugin. Defaults to "river".
            model_id (str): The model ID to use for the TTS plugin. Defaults to "mist".
            lang (str): The language to use for the TTS plugin. Defaults to "eng".
            sampling_rate (int): The sampling rate to use for the TTS plugin. Defaults to 24000.
            speed_alpha (float): The speed alpha to use for the TTS plugin. Defaults to 1.0.
            reduce_latency (bool): Whether to reduce latency for the TTS plugin. Defaults to True.
            pause_between_brackets (bool): Whether to pause between brackets for the TTS plugin. Defaults to False.
            phonemize_between_brackets (bool): Whether to phonemize between brackets for the TTS plugin. Defaults to False.
            inline_speed_alpha (Optional[str], optional): The inline speed alpha to use for the TTS plugin. Defaults to None.
            temperature (Optional[float], optional): Sampling temperature (Arcana model only). Server applies its own
                default when omitted.
            top_p (Optional[float], optional): Nucleus sampling cutoff (Arcana model only).
            repetition_penalty (Optional[float], optional): Repetition penalty (Arcana model only).
            max_tokens (Optional[int], optional): Token limit for output (Arcana model only).
        """
        actual_sample_rate = sampling_rate
        super().__init__(sample_rate=actual_sample_rate, num_channels=RIME_CHANNELS)

        self.speaker = speaker
        self.model_id = model_id
        self.lang = lang
        self.sampling_rate = sampling_rate
        self.speed_alpha = speed_alpha
        self.reduce_latency = reduce_latency
        self.pause_between_brackets = pause_between_brackets
        self.phonemize_between_brackets = phonemize_between_brackets
        self.inline_speed_alpha = inline_speed_alpha
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.max_tokens = max_tokens
        self.audio_track = None
        self.loop = None
        self._first_chunk_sent = False
        self._interrupted = False

        self.api_key = api_key or os.getenv("RIME_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Rime AI API key required. Provide either:\n"
                "1. api_key parameter, OR\n"
                "2. RIME_API_KEY environment variable"
            )

        if model_id in KNOWN_SPEAKERS and speaker not in KNOWN_SPEAKERS[model_id]:
            available = ", ".join(KNOWN_SPEAKERS[model_id])
            print(
                f" Warning: Speaker '{speaker}' may not be available for model '{model_id}'. "
                f"Known speakers: {available}"
            )

        # Read timeout adapts to model: Arcana long-form synthesis can take
        # >60s, while Mist family typically returns within ~5s.
        read_timeout = MODEL_READ_TIMEOUT_SEC.get(model_id, 30.0)
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=15.0, read=read_timeout, write=5.0, pool=5.0
            ),
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
        if not self.audio_track or not self.loop:
            self.emit("error", "Audio track or loop not initialized")
            return

        self._interrupted = False
        try:
            if isinstance(text, str):
                if not self._interrupted:
                    await self._synthesize_audio(text)
            else:
                async for segment in segment_text(text):
                    if self._interrupted:
                        break
                    await self._synthesize_audio(segment)

        except Exception as e:
            self.emit("error", f"Rime TTS synthesis failed: {str(e)}")

    async def _synthesize_audio(self, text: str) -> None:
        """Synthesize text to speech using Rime AI streaming API"""
        if not text or not text.strip() or self._interrupted:
            return
        if len(text) > 500:
            self.emit(
                "error", f"Text exceeds 500 character limit. Got {len(text)} characters.")
            return

        payload: dict[str, Any] = {
            "speaker": self.speaker,
            "text": text,
            "modelId": self.model_id,
            "lang": self.lang,
            "samplingRate": self.sampling_rate,
            "speedAlpha": self.speed_alpha,
            "reduceLatency": self.reduce_latency,
            "pauseBetweenBrackets": self.pause_between_brackets,
            "phonemizeBetweenBrackets": self.phonemize_between_brackets,
        }

        if self.inline_speed_alpha:
            payload["inlineSpeedAlpha"] = self.inline_speed_alpha
        # Arcana-only sampling controls are sent only when the caller set them
        # — letting Rime apply its model defaults on Mist requests, which would
        # otherwise reject the unknown fields.
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.repetition_penalty is not None:
            payload["repetition_penalty"] = self.repetition_penalty
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens

        headers = {
            "Accept": "audio/pcm",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(2):
            try:
                async with self._http_client.stream(
                    "POST", RIME_TTS_ENDPOINT, headers=headers, json=payload
                ) as response:
                    response.raise_for_status()

                    async for chunk in response.aiter_bytes():
                        if self._interrupted:
                            return
                        if chunk:
                            await self._process_audio_chunk(chunk)
                return

            except (httpx.NetworkError, httpx.ConnectError, httpx.ReadTimeout) as e:
                if attempt == 0 and not self._interrupted:
                    await asyncio.sleep(0.25 * (2 ** attempt))
                    continue
                self.emit("error", f"Rime TTS network failure: {str(e)}")
                return
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    self.emit(
                        "error", "Rime TTS authentication failed. Please check your API key.")
                elif e.response.status_code == 400:
                    error_text = e.response.text
                    if "doesn't match list" in error_text:
                        available = ", ".join(
                            KNOWN_SPEAKERS.get(self.model_id, []))
                        self.emit("error", f"Speaker '{self.speaker}' not available for model '{self.model_id}'. "
                                  f"Try one of: {available}")
                    else:
                        self.emit("error", f"Rime TTS bad request: {error_text}")
                else:
                    self.emit(
                        "error", f"Rime TTS HTTP error: {e.response.status_code} - {e.response.text}")
                return
            except Exception as e:
                if not self._interrupted:
                    self.emit("error", f"Rime TTS request failed: {str(e)}")
                return

    async def _process_audio_chunk(self, audio_chunk: bytes) -> None:
        """Process individual audio chunks in real-time for minimal latency"""
        if not audio_chunk or self._interrupted:
            return

        processed_chunk = self._remove_wav_header(audio_chunk)

        if not processed_chunk:
            return

        if not self._first_chunk_sent and self._first_audio_callback:
            self._first_chunk_sent = True
            await self._first_audio_callback()

        if self.audio_track and self.loop:
            asyncio.create_task(
                self.audio_track.add_new_bytes(processed_chunk))

    def _remove_wav_header(self, audio_bytes: bytes) -> bytes:
        """Remove WAV header if present to get raw PCM data"""
        if audio_bytes.startswith(b"RIFF"):
            data_pos = audio_bytes.find(b"data")
            if data_pos != -1:
                return audio_bytes[data_pos + 8:]

        return audio_bytes

    async def aclose(self) -> None:
        """Cleanup HTTP client resources"""
        if self._http_client:
            await self._http_client.aclose()
        await super().aclose()

    async def interrupt(self) -> None:
        """Interrupt the TTS audio stream"""
        self._interrupted = True
        if self.audio_track:
            self.audio_track.interrupt()
