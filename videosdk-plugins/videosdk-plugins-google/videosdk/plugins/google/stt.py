from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Optional, Union, AsyncGenerator
import numpy as np
from videosdk.agents import STT as BaseSTT, STTResponse, SpeechEventType, SpeechData
import logging

logger = logging.getLogger(__name__)

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
            logger.error("Google Cloud Speech V2 is not available")
            raise ImportError("google-cloud-speech is not installed")

        if api_key:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = api_key
        try:
            gauth_default()
        except DefaultCredentialsError as e:
            logger.error("Google credentials are not configured", exc_info=True)
            raise ValueError("Google credentials are not configured.") from e

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
        try:
            opts = None
            if self._config["location"] != "global":
                opts = ClientOptions(api_endpoint=f"{self._config['location']}-speech.googleapis.com")
            self._client = SpeechAsyncClient(client_options=opts)
        except Exception as e:
            logger.error("Failed to create SpeechAsyncClient", exc_info=True)
            raise e

    async def process_audio(self, audio_frames: bytes, **kwargs: Any) -> None:
        try:
            if not self._stream:
                await self._start_stream()
            
            if self._stream:
                if SCIPY_AVAILABLE:
                    try:
                        audio_data = np.frombuffer(audio_frames, dtype=np.int16)
                        resampled_data = signal.resample(audio_data, int(len(audio_data) * self.target_sample_rate / self.input_sample_rate))
                        resampled_bytes = resampled_data.astype(np.int16).tobytes()
                        await self._stream.push_audio(resampled_bytes)
                    except Exception as e:
                        logger.error("Error resampling audio", exc_info=True)
                        self.emit("error", {"message": "Error resampling audio", "error": str(e)})
                else:
                    await self._stream.push_audio(audio_frames)
        except Exception as e:
            logger.error("process_audio failed", exc_info=True)
            if self._stream:
                self.emit("error", {"message": "Failed to process audio", "error": str(e)})

    async def _start_stream(self):
        await self._ensure_client()
        try:
            self._stream = SpeechStream(self._client, self._config, self._transcript_callback)
            await self._stream.start()
        except Exception as e:
            logger.error("Failed to start SpeechStream", exc_info=True)
            raise e

    async def aclose(self) -> None:
        try:
            if self._stream:
                await self._stream.close()
                self._stream = None
            self._client = None
        except Exception as e:
            logger.error("Error during aclose", exc_info=True)


class SpeechStream:
    def __init__(self, client: SpeechAsyncClient, config: dict, transcript_callback):
        self._client = client
        self._config = config
        self._transcript_callback = transcript_callback
        self._audio_queue = asyncio.Queue()
        self._running = False
        self._stream_task: Optional[asyncio.Task] = None
        self.emit = lambda event, payload: logger.warning(f"Emit: {event}, Payload: {payload}")  # mock

    async def start(self):
        if self._running:
            return
        try:
            self._running = True
            self._stream_task = asyncio.create_task(self._stream_loop())
        except Exception as e:
            self.emit("error", {"message": "Failed to start stream loop", "error": str(e)})

    async def push_audio(self, audio_frames: bytes):
        if not self._running:
            logger.warning("Tried to push audio when stream is not running")
            return
        try:
            await self._audio_queue.put(audio_frames)
        except Exception as e:
            self.emit("error", {"message": "Failed to push audio", "error": str(e)})

    async def _audio_generator(self) -> AsyncGenerator[speech_types.StreamingRecognizeRequest, None]:
        try:
            _, project_id = gauth_default()
            recognizer = f"projects/{project_id}/locations/{self._config['location']}/recognizers/_"
        except Exception as e:
            self.emit("error", {"message": "Failed to get project id", "error": str(e)})
            return

        try:
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
            yield speech_types.StreamingRecognizeRequest(
                recognizer=recognizer, streaming_config=streaming_config
            )
        except Exception as e:
            self.emit("error", {"message": "Failed to configure streaming", "error": str(e)})
            return

        while self._running:
            try:
                chunk = await asyncio.wait_for(self._audio_queue.get(), timeout=0.1)
                yield speech_types.StreamingRecognizeRequest(audio=chunk)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.emit("error", {"message": "Audio chunk error", "error": str(e)})

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

            except (DeadlineExceeded, asyncio.TimeoutError) as e:
                self.emit("error", {"message": "Streaming timeout", "error": str(e)})
            except GoogleAPICallError as e:
                self.emit("error", {"message": "Google API call error", "error": str(e)})
                await asyncio.sleep(2)
            except Exception as e:
                self.emit("error", {"message": "Google STT error", "error": str(e)})
                await asyncio.sleep(2)

            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                except Exception as e:
                    logger.warning("Failed to flush audio queue", exc_info=True)

    def _handle_response(self, response: speech_types.StreamingRecognizeResponse):
        try:
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
        except Exception as e:
            self.emit("error", {"message": "Error handling response", "error": str(e)})

    async def close(self):
        self._running = False
        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass