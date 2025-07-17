from __future__ import annotations

import asyncio
import os
from typing import Any, Optional
import numpy as np
from videosdk.agents import STT as BaseSTT, STTResponse, SpeechData, SpeechEventType, global_event_emitter
from bodhi import BodhiClient, TranscriptionConfig, TranscriptionResponse, LiveTranscriptionEvents

try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class NavanaSTT(BaseSTT):
    """
    VideoSDK Agent Framework STT plugin for Navana's Bodhi API.
    
    This plugin uses the official 'bodhi-sdk' and implements best practices for audio handling,
    including robust stereo-to-mono conversion and event model adaptation.
    """
    
    def __init__(
        self,
        *,
        customer_id: str | None = None,
        api_key: str | None = None,
        model: str = "en-general-v2-8khz",
        language: str = "en",
        input_sample_rate: int = 48000,
    ) -> None:
        super().__init__()

        if not SCIPY_AVAILABLE:
            raise ImportError("The 'scipy' library is not installed. Please install it with 'pip install scipy' to use the NavanaSTT plugin for audio resampling.")

        self.customer_id = customer_id or os.getenv("NAVANA_CUSTOMER_ID")
        self.api_key = api_key or os.getenv("NAVANA_API_KEY")
        
        if not self.api_key or not self.customer_id:
            raise ValueError(
                "Navana API key and Customer ID must be provided either through parameters or "
                "NAVANA_API_KEY/NAVANA_CUSTOMER_ID environment variables."
            )
        
        self.model = model
        self.language = language
        self.input_sample_rate = input_sample_rate
        self.target_sample_rate = 8000
        
        self.client = BodhiClient(api_key=self.api_key, customer_id=self.customer_id)
        self._connection_started = False
        self._last_transcript_text = ""
        
        self._register_event_handlers()

    def _register_event_handlers(self):
        """Registers handlers for the Bodhi client's transcription events."""
        self.client.on(LiveTranscriptionEvents.Transcript, self._on_transcript)
        self.client.on(LiveTranscriptionEvents.UtteranceEnd, self._on_utterance_end)
        self.client.on(LiveTranscriptionEvents.SpeechStarted, self._on_speech_started)
        self.client.on(LiveTranscriptionEvents.Error, self._on_error)
        self.client.on(LiveTranscriptionEvents.Close, self._on_close)

    async def _on_transcript(self, response: TranscriptionResponse):
        """Handles interim results, updating the latest transcript buffer."""
        if response.text and self._transcript_callback:
            self._last_transcript_text = response.text
            event = STTResponse(
                event_type=SpeechEventType.INTERIM,
                data=SpeechData(text=response.text, language=self.language, confidence=1.0)
            )
            await self._transcript_callback(event)

    async def _on_utterance_end(self, response: dict):
        """On utterance end, promotes the last known transcript to FINAL."""
        if self._last_transcript_text and self._transcript_callback:
            final_text = self._last_transcript_text
            self._last_transcript_text = ""
            event = STTResponse(
                event_type=SpeechEventType.FINAL,
                data=SpeechData(text=final_text, language=self.language, confidence=1.0)
            )
            await self._transcript_callback(event)

    async def _on_speech_started(self, response: TranscriptionResponse):
        global_event_emitter.emit("speech_started")

    async def _on_error(self, e: Exception):
        error_message = f"Navana SDK Error: {str(e)}"
        print(error_message)
        self.emit("error", error_message)

    async def _on_close(self):
        print("Navana SDK connection closed.")
        self._connection_started = False
        
    async def process_audio(
        self,
        audio_frames: bytes,
        language: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Processes audio by converting stereo to mono, resampling, and sending to the STT service.
        """
        try:
            if not self._connection_started:
                config = TranscriptionConfig(
                    model=self.model,
                    sample_rate=self.target_sample_rate
                )
                await self.client.start_connection(config=config)
                self._connection_started = True
            
            raw_audio_data = np.frombuffer(audio_frames, dtype=np.int16)
            stereo_audio = raw_audio_data.reshape(-1, 2)
            mono_audio_float = stereo_audio.astype(np.float32).mean(axis=1)
            resampled_data = signal.resample(
                mono_audio_float, 
                int(len(mono_audio_float) * self.target_sample_rate / self.input_sample_rate)
            )
            
            audio_bytes = resampled_data.astype(np.int16).tobytes()
            
            await self.client.send_audio_stream(audio_bytes)
            
        except Exception as e:
            error_message = f"Audio processing error: {str(e)}"
            print(error_message)
            self.emit("error", error_message)
            self._connection_started = False
            if self.client._live_client and not self.client._live_client.is_closed:
                await self.client.close_connection()


    async def aclose(self) -> None:
        """Cleans up resources by closing the SDK connection."""
        if self._connection_started:
            await self.client.close_connection()