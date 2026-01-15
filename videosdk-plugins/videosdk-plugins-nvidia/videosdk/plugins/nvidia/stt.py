from __future__ import annotations

import asyncio
import os
import logging
import queue
import threading
import time
from typing import Any, Generator

import numpy as np
from videosdk.agents import STT as BaseSTT, STTResponse, SpeechEventType, SpeechData, global_event_emitter

try:
    import riva.client
    from riva.client.proto.riva_audio_pb2 import AudioEncoding
except ImportError:
    riva = None
    AudioEncoding = None

logger = logging.getLogger(__name__)

class NvidiaSTT(BaseSTT):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer",
        server: str = "grpc.nvcf.nvidia.com:443",
        function_id: str = "1598d209-5e27-4d3c-8079-4751568b1081",
        language_code: str = "en-US",
        sample_rate: int = 16000,
        use_ssl: bool = True,
        profanity_filter: bool = False,
        automatic_punctuation: bool = True,
    ) -> None:
        super().__init__()
        
        if riva is None:
            raise ImportError("nvidia-riva-client is not installed.")

        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA API key must be provided")
        self.model = model
        self.server = server
        self.function_id = function_id
        self.language_code = language_code
        self.sample_rate = sample_rate
        self.use_ssl = use_ssl
        self.profanity_filter = profanity_filter
        self.automatic_punctuation = automatic_punctuation
        self.input_sample_rate = 48000
        self._audio_queue = queue.Queue()
        self._buffer = bytearray()
            
        try:
            self._main_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._main_loop = asyncio.get_event_loop()
        
        self._auth = None
        self._asr_service = None
        self._recognition_thread = None
        self._stop_event = threading.Event()

        self._initialize_client()

    def _initialize_client(self):
        self._auth = riva.client.Auth(
            ssl_root_cert=None,
            use_ssl=self.use_ssl,
            uri=self.server,
            metadata_args=[
                ["function-id", self.function_id],
                ["authorization", f"Bearer {self.api_key}"],
            ]
        )
        self._asr_service = riva.client.ASRService(self._auth)

    def _get_config(self):
        return riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                model=self.model,
                encoding=AudioEncoding.LINEAR_PCM,
                language_code=self.language_code,
                max_alternatives=1,
                profanity_filter=self.profanity_filter,
                enable_automatic_punctuation=self.automatic_punctuation,
                sample_rate_hertz=self.sample_rate,
                audio_channel_count=1,
                enable_word_time_offsets=True,
            ),
            interim_results=True,
        )

    async def process_audio(self, audio_frames: bytes, **kwargs: Any) -> None:
        """Receive audio from VideoSDK, convert to Mono/16kHz, and put into queue."""
        if self._recognition_thread is None:
            self._start_recognition_thread()

        try:
            audio_np = np.frombuffer(audio_frames, dtype=np.int16)
            if audio_np.size % 2 == 0:
                audio_np = audio_np.reshape(-1, 2).mean(axis=1).astype(np.int16)
            if self.input_sample_rate == 48000 and self.sample_rate == 16000:
                audio_np = audio_np[::3]
            
            processed_bytes = audio_np.tobytes()
            if processed_bytes:
                self._audio_queue.put(processed_bytes)
                
        except Exception as e:
            self.emit("error", f"Error processing audio for Nvidia STT: {str(e)}")
            logger.error(f"Error processing audio for Nvidia STT: {e}")

    def _start_recognition_thread(self):
        self._stop_event.clear()
        self._recognition_thread = threading.Thread(
            target=self._recognition_worker,
            name="nvidia-stt-worker",
            daemon=True
        )
        self._recognition_thread.start()

    def _audio_generator(self) -> Generator[bytes, None, None]:
        """Generator that yields audio chunks from the queue."""
        while not self._stop_event.is_set():
            try:
                chunk = self._audio_queue.get(timeout=0.5)
                if chunk is None:
                    break
                yield chunk
            except queue.Empty:
                continue

    def _recognition_worker(self):
        """Blocking worker running in a separate thread."""
        logger.info("Nvidia STT: Worker thread started.")
        config = self._get_config()
        
        try:
            responses = self._asr_service.streaming_response_generator(
                audio_chunks=self._audio_generator(),
                streaming_config=config
            )
            
            for response in responses:
                if self._stop_event.is_set():
                    break
                self._handle_response(response)
                
        except Exception as e:
            self.emit("error", f"Error in Nvidia STT worker: {str(e)}")
            logger.error(f"Nvidia STT Error: {e}")
        finally:
            logger.info("Nvidia STT: Worker thread stopped.")

    def _handle_response(self, response):
        """Process raw Riva response and dispatch to main loop."""
        if not response.results:
            return

        for result in response.results:
            if not result.alternatives:
                continue

            transcript = result.alternatives[0].transcript
            is_final = result.is_final

            if transcript:
                self._main_loop.call_soon_threadsafe(
                    lambda t=transcript, f=is_final, r=result: asyncio.create_task(
                        self._dispatch_response(t, f, r)
                    )
                )

    async def _dispatch_response(self, text: str, is_final: bool, raw_result: Any):
        event_type = SpeechEventType.FINAL if is_final else SpeechEventType.INTERIM
        
        if event_type == SpeechEventType.INTERIM:
            global_event_emitter.emit("speech_started")
        elif event_type == SpeechEventType.FINAL:
            global_event_emitter.emit("speech_stopped")

        response = STTResponse(
            event_type=event_type,
            data=SpeechData(
                text=text,
                confidence=raw_result.alternatives[0].confidence,
                language=self.language_code,
                start_time=0.0,
                end_time=0.0,
            ),
            metadata={
                "model": "nvidia-riva",
                "stability": getattr(raw_result, "stability", 0.0),
            }
        )
        
        if self._transcript_callback:
            await self._transcript_callback(response)

    async def aclose(self) -> None:
        """Cleanup resources"""
        self._stop_event.set()
        self._audio_queue.put(None)
        
        if self._recognition_thread:
            self._recognition_thread = None
            
        await super().aclose()