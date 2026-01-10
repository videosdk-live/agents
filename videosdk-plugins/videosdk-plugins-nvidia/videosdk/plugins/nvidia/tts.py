from __future__ import annotations

import asyncio
import os
import logging
import inspect
from typing import Any, AsyncIterator

from videosdk.agents import TTS

try:
    import riva.client
    from riva.client.proto.riva_audio_pb2 import AudioEncoding
except ImportError:
    riva = None
    AudioEncoding = None

logger = logging.getLogger(__name__)

RIVA_SAMPLE_RATE = 24000 
RIVA_CHANNELS = 1

class NvidiaTTS(TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        server: str = "grpc.nvcf.nvidia.com:443",
        function_id: str = "877104f7-e885-42b9-8de8-f6e4c6303969", 
        voice_name: str = "Magpie-Multilingual.EN-US.Aria",
        language_code: str = "en-US",
        sample_rate: int = RIVA_SAMPLE_RATE,
        use_ssl: bool = True,
    ) -> None:
        super().__init__(sample_rate=sample_rate, num_channels=RIVA_CHANNELS)

        if riva is None:
             raise ImportError("nvidia-riva-client is not installed.")

        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA API key must be provided")

        self.server = server
        self.function_id = function_id
        self.voice_name = voice_name
        self.language_code = language_code
        self.use_ssl = use_ssl        
        self._service = None
        self._main_loop = asyncio.get_event_loop()
        self._interrupted = False

        self._initialize_client()

    def _initialize_client(self):
        auth = riva.client.Auth(
            ssl_root_cert=None, 
            use_ssl=self.use_ssl, 
            uri=self.server, 
            metadata_args=[
                ["function-id", self.function_id],
                ["authorization", f"Bearer {self.api_key}"],
            ]
        )
        self._service = riva.client.SpeechSynthesisService(auth)

    async def synthesize(
        self, 
        text: AsyncIterator[str] | str, 
        **kwargs: Any,
    ) -> None:
        """Synthesize text to speech using NVIDIA Riva."""
        try:
            if not self.audio_track:
                self.emit("error", "Audio track not set")
                return

            self._interrupted = False

            input_text = ""
            if inspect.isasyncgen(text):
                async for chunk in text:
                    if self._interrupted:
                        break
                    input_text += chunk
            else:
                input_text = text

            if not input_text.strip() or self._interrupted:
                return

            await asyncio.to_thread(self._worker_synthesize, input_text)

        except Exception as e:
            self.emit("error", f"TTS synthesis failed: {str(e)}")
            logger.error(f"Nvidia TTS Error: {e}")

    def _worker_synthesize(self, text: str):
        """Blocking worker that streams audio back to the main loop."""
        try:
            logger.info(f"Nvidia TTS: Requesting '{text[:20]}...' at {self.sample_rate}Hz")
            
            responses = self._service.synthesize_online(
                text,
                voice_name=self.voice_name,
                language_code=self.language_code,
                sample_rate_hz=self.sample_rate, 
                encoding=AudioEncoding.LINEAR_PCM 
            )
            
            first_chunk = True
            
            for resp in responses:
                if self._interrupted:
                    break
                    
                audio_data = resp.audio
                if audio_data:
                    self._main_loop.call_soon_threadsafe(
                        self._dispatch_audio, audio_data, first_chunk
                    )
                    first_chunk = False
                    
        except Exception as e:
            self.emit("error", f"TTS synthesis failed: {str(e)}")
            logger.error(f"Nvidia TTS Worker Error: {e}")

    def _dispatch_audio(self, audio_data: bytes, is_first: bool):
        """Executed on main loop."""
        if self._interrupted:
            return

        if is_first and self._first_audio_callback:
            asyncio.create_task(self._first_audio_callback())

        if self.audio_track:
            asyncio.create_task(self.audio_track.add_new_bytes(audio_data))

    async def interrupt(self) -> None:
        """Interrupt current synthesis and audio playback."""
        self._interrupted = True
        if self.audio_track:
            self.audio_track.interrupt()

    async def aclose(self) -> None:
        await super().aclose()