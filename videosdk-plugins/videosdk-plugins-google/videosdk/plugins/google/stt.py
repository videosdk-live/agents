from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Optional, Union, AsyncGenerator
import numpy as np
from videosdk.agents import STT as BaseSTT, STTResponse, SpeechEventType, SpeechData

try:
    from google.cloud.speech_v2 import SpeechAsyncClient, types as speech_types
    from google.api_core.exceptions import DeadlineExceeded, GoogleAPICallError
    from google.auth import default as gauth_default
    from google.auth.exceptions import DefaultCredentialsError
    from google.api_core.client_options import ClientOptions
    try:
        from scipy import signal
        SCIPY_AVAILABLE = True
    except ImportError:
        SCIPY_AVAILABLE = False
    GOOGLE_V2_AVAILABLE = True
except ImportError:
    GOOGLE_V2_AVAILABLE = False

_MAX_SESSION_DURATION = 240  

class GoogleSTT(BaseSTT):
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        languages: Union[str, list[str]] = "en-US",
        model: str = "latest_long",
        sample_rate: int = 16000,
        interim_results: bool = True,
        punctuate: bool = True,
        min_confidence_threshold: float = 0.1,
        location: str = "global",
        **kwargs: Any
    ) -> None:
        super().__init__()
        if not GOOGLE_V2_AVAILABLE:
            raise ImportError("google-cloud-speech is not installed")

        if api_key:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = api_key
        try:
            gauth_default()
        except DefaultCredentialsError:
            raise ValueError("Google credentials are not configured.")

        self.input_sample_rate = 48000
        self.target_sample_rate = sample_rate
        if isinstance(languages, str):
            languages = [languages]
        
        self._config = {
            "languages": languages,
            "model": model,
            "sample_rate": self.target_sample_rate,
            "interim_results": interim_results,
            "punctuate": punctuate,
            "min_confidence_threshold": min_confidence_threshold,
            "location": location,
        }

        self._client: Optional[SpeechAsyncClient] = None
        self._stream: Optional[SpeechStream] = None

    async def _ensure_client(self):
        if self._client:
            return
        
        opts = None
        if self._config["location"] != "global":
            opts = ClientOptions(api_endpoint=f"{self._config['location']}-speech.googleapis.com")
        self._client = SpeechAsyncClient(client_options=opts)

    async def process_audio(self, audio_frames: bytes, **kwargs: Any) -> None:
        if not self._stream:
            await self._start_stream()
        
        if self._stream:
            if SCIPY_AVAILABLE:
                audio_data = np.frombuffer(audio_frames, dtype=np.int16)
                resampled_data = signal.resample(audio_data, int(len(audio_data) * self.target_sample_rate / self.input_sample_rate))
                resampled_bytes = resampled_data.astype(np.int16).tobytes()
                await self._stream.push_audio(resampled_bytes)
            else:
                await self._stream.push_audio(audio_frames)

    async def _start_stream(self):
        await self._ensure_client()
        self._stream = SpeechStream(self._client, self._config, self._transcript_callback)
        await self._stream.start()

    async def aclose(self) -> None:
        if self._stream:
            await self._stream.close()
            self._stream = None
        
        if self._client:
            self._client = None

class SpeechStream:
    def __init__(self, client: SpeechAsyncClient, config: dict, transcript_callback):
        self._client = client
        self._config = config
        self._transcript_callback = transcript_callback
        self._audio_queue = asyncio.Queue()
        self._running = False
        self._stream_task: Optional[asyncio.Task] = None

    async def start(self):
        if self._running:
            return
        self._running = True
        self._stream_task = asyncio.create_task(self._stream_loop())

    async def push_audio(self, audio_frames: bytes):
        if not self._running:
            return
        await self._audio_queue.put(audio_frames)

    async def _audio_generator(self) -> AsyncGenerator[speech_types.StreamingRecognizeRequest, None]:
        _, project_id = gauth_default()
        recognizer = f"projects/{project_id}/locations/{self._config['location']}/recognizers/_"

        streaming_config = speech_types.StreamingRecognitionConfig(
            config=speech_types.RecognitionConfig(
                explicit_decoding_config=speech_types.ExplicitDecodingConfig(
                    encoding='LINEAR16',
                    sample_rate_hertz=self._config["sample_rate"],
                    audio_channel_count=2,
                ),
                language_codes=self._config["languages"],
                model=self._config["model"],
                features=speech_types.RecognitionFeatures(
                    enable_automatic_punctuation=self._config["punctuate"],
                ),
            ),
            streaming_features=speech_types.StreamingRecognitionFeatures(
                interim_results=self._config["interim_results"],
            ),
        )
        yield speech_types.StreamingRecognizeRequest(recognizer=recognizer, streaming_config=streaming_config)

        while self._running:
            try:
                chunk = await asyncio.wait_for(self._audio_queue.get(), timeout=0.1)
                yield speech_types.StreamingRecognizeRequest(audio=chunk)
            except asyncio.TimeoutError:
                continue

    async def _stream_loop(self):
        session_started_at = 0
        while self._running:
            try:
                session_started_at = time.time()
                stream = await self._client.streaming_recognize(requests=self._audio_generator())
                
                async for response in stream:
                    if time.time() - session_started_at > _MAX_SESSION_DURATION:
                        break
                    
                    self._handle_response(response)

            except (DeadlineExceeded, asyncio.TimeoutError):
                pass
            except GoogleAPICallError:
                await asyncio.sleep(2)
            except Exception:
                await asyncio.sleep(2)
            
            while not self._audio_queue.empty():
                self._audio_queue.get_nowait()

    def _handle_response(self, response: speech_types.StreamingRecognizeResponse):
        if not response.results or not response.results[0].alternatives:
            return

        alt = response.results[0].alternatives[0]
        transcript = alt.transcript.strip()
        if not transcript:
            return

        is_final = response.results[0].is_final
        confidence = alt.confidence
        
        if confidence >= self._config["min_confidence_threshold"]:
            if self._transcript_callback:
                event = STTResponse(
                    event_type=SpeechEventType.FINAL if is_final else SpeechEventType.INTERIM,
                    data=SpeechData(
                        text=transcript,
                        confidence=confidence,
                        language=response.results[0].language_code or self._config["languages"][0]
                    )
                )
                asyncio.create_task(self._transcript_callback(event))

    async def close(self):
        self._running = False
        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass