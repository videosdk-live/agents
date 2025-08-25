from __future__ import annotations

import numpy as np
from typing import Any, Literal
import time
import asyncio
from scipy import signal
from .onnx_runtime import VadModelWrapper, SAMPLE_RATES
from videosdk.agents.vad import VAD as BaseVAD, VADResponse, VADEventType, VADData


class SileroVAD(BaseVAD):
    """Silero Voice Activity Detection implementation using ONNX runtime"""

    def __init__(
        self,
        input_sample_rate: int = 48000,
        model_sample_rate: Literal[8000, 16000] = 16000,
        threshold: float = 0.30,
        min_speech_duration: float = 0.1,
        min_silence_duration: float = 0.75,
        force_cpu: bool = True,
    ) -> None:

        if model_sample_rate not in SAMPLE_RATES:
            self.emit(
                "error", f"Invalid model sample rate {model_sample_rate}: must be one of {SAMPLE_RATES}")
            raise ValueError(
                f"Model sample rate {model_sample_rate} not supported. Must be one of {SAMPLE_RATES}")

        super().__init__(
            sample_rate=model_sample_rate,
            threshold=threshold,
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration
        )

        self._input_sample_rate = input_sample_rate
        self._model_sample_rate = model_sample_rate
        self._needs_resampling = input_sample_rate != model_sample_rate

        try:
            self._session = VadModelWrapper.create_inference_session(force_cpu)
            self._model = VadModelWrapper(
                session=self._session, rate=model_sample_rate)
        except Exception as e:
            self.emit("error", f"Failed to initialize VAD model: {str(e)}")
            raise

        self._exp_filter = 0.0

        self._speech_threshold_duration = 0.0
        self._silence_threshold_duration = 0.0

        self._pub_speaking = False
        self._pub_speech_duration = 0.0
        self._pub_silence_duration = 0.0
        self._pub_timestamp = 0.0

        self._remaining_input_fraction = 0.0

        self._input_accumulator = np.array([], dtype=np.int16)
        self._inference_accumulator = np.array([], dtype=np.float32)

        self._frame_count = 0
        self._inference_count = 0

        self._consecutive_low_confidence_count = 0
        self._error_emission_threshold = 10

    async def process_audio(self, audio_frames: bytes, **kwargs: Any) -> None:
        try:
            input_frame_data = np.frombuffer(audio_frames, dtype=np.int16)

            self._input_accumulator = np.concatenate(
                [self._input_accumulator, input_frame_data])

            if self._needs_resampling:
                input_float = input_frame_data.astype(np.float32) / 32768.0
                target_length = int(
                    len(input_float) * self._model_sample_rate / self._input_sample_rate)
                if target_length > 0:
                    resampled_float = signal.resample(
                        input_float, target_length)
                    self._inference_accumulator = np.concatenate([
                        self._inference_accumulator,
                        resampled_float.astype(np.float32)
                    ])
            else:
                input_float = input_frame_data.astype(np.float32) / 32768.0
                self._inference_accumulator = np.concatenate(
                    [self._inference_accumulator, input_float])

            while len(self._inference_accumulator) >= self._model.frame_size:
                inference_window = self._inference_accumulator[:self._model.frame_size]

                try:
                    raw_prob = self._model.process(inference_window)
                except Exception as e:
                    self.emit("error", f"VAD inference error: {e}")
                    raw_prob = 0.0

                alpha = 0.40
                self._exp_filter = alpha * raw_prob + \
                    (1 - alpha) * self._exp_filter

                window_duration = self._model.frame_size / self._model_sample_rate
                self._pub_timestamp += window_duration

                resampling_ratio = self._input_sample_rate / self._model_sample_rate

                _copy = self._model.frame_size * \
                    resampling_ratio + self._remaining_input_fraction
                _int_copy = int(_copy)

                self._remaining_input_fraction = _copy - _int_copy

                if len(self._input_accumulator) >= _int_copy:
                    self._input_accumulator = self._input_accumulator[_int_copy:]

                if self._pub_speaking:
                    self._pub_speech_duration += window_duration
                else:
                    self._pub_silence_duration += window_duration

                if self._exp_filter >= self._threshold:
                    self._speech_threshold_duration += window_duration
                    self._silence_threshold_duration = 0.0

                    if not self._pub_speaking:
                        if self._speech_threshold_duration >= self._min_speech_duration:
                            self._pub_speaking = True
                            self._pub_silence_duration = 0.0
                            self._pub_speech_duration = self._speech_threshold_duration

                            self._send_speech_event(
                                VADEventType.START_OF_SPEECH)
                else:
                    self._silence_threshold_duration += window_duration
                    self._speech_threshold_duration = 0.0

                    if not self._pub_speaking:
                        pass

                    if self._pub_speaking and self._silence_threshold_duration >= self._min_silence_duration:
                        self._pub_speaking = False
                        self._pub_speech_duration = 0.0
                        self._pub_silence_duration = self._silence_threshold_duration

                        self._send_speech_event(VADEventType.END_OF_SPEECH)
                        self._reset_model_state()
                        pass

                self._inference_accumulator = self._inference_accumulator[self._model.frame_size:]

        except Exception as e:
            self.emit("error", f"VAD audio processing failed: {str(e)}")

    def _send_speech_event(self, event_type: VADEventType) -> None:
        response = VADResponse(
            event_type=event_type,
            data=VADData(
                is_speech=event_type == VADEventType.START_OF_SPEECH,
                confidence=self._exp_filter,
                timestamp=self._pub_timestamp,
                speech_duration=self._pub_speech_duration,
                silence_duration=self._pub_silence_duration
            )
        )
        if self._vad_callback:
            asyncio.create_task(self._vad_callback(response))

    def _reset_model_state(self) -> None:
        """Reset model internal state when errors occur"""
        try:
            self._model._hidden_state = np.zeros((2, 1, 128), dtype=np.float32)
            self._model._prev_context = np.zeros(
                (1, self._model.history_len), dtype=np.float32)

            self._exp_filter = 0.0
            self._speech_threshold_duration = 0.0
            self._silence_threshold_duration = 0.0
        except Exception as e:
            self.emit("error", f"Failed to reset VAD model state: {e}")

    async def aclose(self) -> None:
        """Cleanup resources"""
        try:
            self._input_accumulator = np.array([], dtype=np.int16)
            self._inference_accumulator = np.array([], dtype=np.float32)
        except Exception as e:
            self.emit("error", f"Error during VAD cleanup: {str(e)}")
