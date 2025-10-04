from __future__ import annotations

import asyncio
import os
import time
import threading
from dataclasses import dataclass
from typing import Any, Optional, List

import azure.cognitiveservices.speech as speechsdk

from videosdk.agents import (
    STT as BaseSTT,
    STTResponse,
    SpeechEventType,
    SpeechData,
    global_event_emitter,
)

import logging

logger = logging.getLogger(__name__)

try:
    from scipy import signal
    import numpy as np

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class AzureSTTConfig:
    """Configuration for Azure STT"""

    speech_key: str
    speech_region: str
    language: str = "en-US"
    sample_rate: int = 16000
    enable_phrase_list: bool = False
    phrase_list: Optional[List[str]] = None


class AzureSTT(BaseSTT):
    def __init__(
        self,
        *,
        speech_key: Optional[str] = None,
        speech_region: Optional[str] = None,
        language: str = "en-US",
        sample_rate: int = 16000,
        enable_phrase_list: bool = False,
        phrase_list: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Azure STT plugin.

        Args:
            speech_key (Optional[str]): Azure Speech API key. Uses AZURE_SPEECH_KEY environment variable if not provided.
            speech_region (Optional[str]): Azure Speech region. Uses AZURE_SPEECH_REGION environment variable if not provided.
            language (str): The language to use for the STT plugin. Defaults to "en-US".
            sample_rate (int): Sample rate to use for the STT plugin. Defaults to 16000.
            enable_phrase_list (bool): Whether to enable phrase list for better recognition. Defaults to False.
            phrase_list (Optional[List[str]]): List of phrases to boost recognition. Defaults to None.
        """
        super().__init__()

        if not SCIPY_AVAILABLE:
            raise ImportError(
                "scipy and numpy are required for Azure STT. Please install with 'pip install scipy numpy'"
            )

        self.speech_key = speech_key or os.getenv("AZURE_SPEECH_KEY")
        self.speech_region = speech_region or os.getenv("AZURE_SPEECH_REGION")

        if not self.speech_key or not self.speech_region:
            raise ValueError(
                "Azure Speech key and region must be provided either through parameters or "
                "AZURE_SPEECH_KEY and AZURE_SPEECH_REGION environment variables"
            )

        self.config = AzureSTTConfig(
            speech_key=self.speech_key,
            speech_region=self.speech_region,
            language=language,
            sample_rate=sample_rate,
            enable_phrase_list=enable_phrase_list,
            phrase_list=phrase_list,
        )

        self.input_sample_rate = 48000
        self.target_sample_rate = sample_rate

        self._speech_processor: Optional[speechsdk.SpeechRecognizer] = None
        self._audio_stream: Optional[speechsdk.audio.PushAudioInputStream] = None
        self._is_speaking = False
        self._last_speech_time = 0.0

        self._loop = asyncio.get_running_loop()
        self._event_queue = asyncio.Queue()
        self._processing_task: Optional[asyncio.Task] = None

    async def process_audio(
        self, audio_frames: bytes, language: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Process audio frames and send to Azure Speech Service"""
        try:
            if not self._speech_processor:
                await self._setup_speech_processor(language)

            if self._audio_stream and SCIPY_AVAILABLE:
                audio_data = np.frombuffer(audio_frames, dtype=np.int16)

                if len(audio_data) > 0:
                    stereo_data = audio_data.reshape(-1, 2)
                    mono_data = stereo_data.mean(axis=1)

                    resampled_data = signal.resample(
                        mono_data,
                        int(
                            len(mono_data)
                            * self.target_sample_rate
                            / self.input_sample_rate
                        ),
                    )
                    resampled_bytes = resampled_data.astype(np.int16).tobytes()
                    self._audio_stream.write(resampled_bytes)

        except Exception as e:
            logger.error(f"Error in process_audio: {str(e)}")
            self.emit("error", str(e))
            await self._cleanup_speech_processor()

    async def _setup_speech_processor(self, language: Optional[str] = None) -> None:
        """Setup Azure speech processor"""
        try:
            self._processing_task = self._loop.create_task(self._process_events())

            speech_config = speechsdk.SpeechConfig(
                subscription=self.config.speech_key, region=self.config.speech_region
            )
            speech_config.speech_recognition_language = language or self.config.language

            stream_format = speechsdk.audio.AudioStreamFormat(
                samples_per_second=self.config.sample_rate,
                bits_per_sample=16,
                channels=1,
            )
            self._audio_stream = speechsdk.audio.PushAudioInputStream(
                stream_format=stream_format
            )

            audio_config = speechsdk.audio.AudioConfig(stream=self._audio_stream)

            self._speech_processor = speechsdk.SpeechRecognizer(
                speech_config=speech_config, audio_config=audio_config
            )

            if self.config.enable_phrase_list and self.config.phrase_list:
                phrase_list_grammar = speechsdk.PhraseListGrammar.from_recognizer(
                    self._speech_processor
                )
                for phrase in self.config.phrase_list:
                    phrase_list_grammar.addPhrase(phrase)

            self._speech_processor.recognized.connect(self._on_final_transcript)
            self._speech_processor.recognizing.connect(self._on_interim_transcript)
            self._speech_processor.speech_start_detected.connect(self._on_user_started_speaking)
            self._speech_processor.speech_end_detected.connect(self._on_user_stopped_speaking)
            self._speech_processor.canceled.connect(self._on_speech_processing_error)

            self._speech_processor.start_continuous_recognition()
            logger.info("Azure STT speech processor started")

        except Exception as e:
            logger.error(f"Failed to setup speech processor: {str(e)}")
            raise

    def _on_final_transcript(self, evt: speechsdk.SpeechRecognitionEventArgs) -> None:
        """Handle final recognition results"""
        text = evt.result.text.strip()
        if not text:
            return

        if self._transcript_callback:
            response = STTResponse(
                event_type=SpeechEventType.FINAL,
                data=SpeechData(
                    text=text, language=self.config.language, confidence=1.0
                ),
                metadata={"provider": "azure", "result_reason": str(evt.result.reason)},
            )
            self._event_queue.put_nowait(response)

    def _on_interim_transcript(self, evt: speechsdk.SpeechRecognitionEventArgs) -> None:
        """Handle interim recognition results"""
        text = evt.result.text.strip()
        if not text:
            return

        if self._transcript_callback:
            response = STTResponse(
                event_type=SpeechEventType.INTERIM,
                data=SpeechData(
                    text=text, language=self.config.language, confidence=0.5
                ),
                metadata={"provider": "azure", "result_reason": str(evt.result.reason)},
            )
            self._event_queue.put_nowait(response)

    def _on_user_started_speaking(self, evt: speechsdk.SpeechRecognitionEventArgs) -> None:
        """Handle speech start detection"""
        if self._is_speaking:
            return

        self._is_speaking = True
        current_time = time.time()

        if self._last_speech_time == 0.0:
            self._last_speech_time = current_time
        else:
            if current_time - self._last_speech_time < 1.0:
                global_event_emitter.emit("speech_started")

        self._last_speech_time = current_time

    def _on_user_stopped_speaking(self, evt: speechsdk.SpeechRecognitionEventArgs) -> None:
        """Handle speech end detection"""
        if not self._is_speaking:
            return

        self._is_speaking = False
        global_event_emitter.emit("speech_stopped")

    def _on_speech_processing_error(self, evt: speechsdk.SpeechRecognitionCanceledEventArgs) -> None:
        """Handle speech processing errors and cancellations"""
        if evt.cancellation_details.reason == speechsdk.CancellationReason.Error:
            error_msg = f"Speech recognition canceled due to error: {evt.cancellation_details.error_details}"
            logger.error(error_msg)
            self.emit("error", error_msg)

    async def _process_events(self) -> None:
        """Process STT events from the queue"""
        while True:
            try:
                response = await self._event_queue.get()
                if self._transcript_callback:
                    await self._transcript_callback(response)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error processing STT event: %s", str(e), exc_info=True)

    async def _cleanup_speech_processor(self) -> None:
        """Cleanup speech processor resources"""
        try:
            if self._speech_processor:
                self._speech_processor.stop_continuous_recognition()
                self._speech_processor = None

            if self._audio_stream:
                self._audio_stream.close()
                self._audio_stream = None

        except Exception as e:
            logger.error(f"Error during speech processor cleanup: {str(e)}")

    async def aclose(self) -> None:
        """Cleanup resources"""
        if self._processing_task:
            self._processing_task.cancel()
            await asyncio.gather(self._processing_task, return_exceptions=True)

        await self._cleanup_speech_processor()
        logger.info("Azure STT closed")
        await super().aclose()
