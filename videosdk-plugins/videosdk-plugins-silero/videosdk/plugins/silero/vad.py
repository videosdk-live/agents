from __future__ import annotations

import numpy as np
from typing import Any, Literal
import time
import asyncio
from scipy import signal
from .onnx_runtime import VadModelWrapper, SAMPLE_RATES
from videosdk.agents.vad import VAD as BaseVAD, VADResponse, VADEventType, VADData
import logging

logger = logging.getLogger(__name__)

INFERENCE_DELAY_TOLERANCE = 0.2


class SileroVAD(BaseVAD):
    """Silero Voice Activity Detection implementation using ONNX runtime.
    
    This implementation buffers audio, runs it through the Silero model,
    applies exponential smoothing to the probabilities to accurately detect the start and end of speech.
    """

    def __init__(
        self,
        input_sample_rate: int = 48000,
        model_sample_rate: Literal[8000, 16000] = 16000,
        threshold: float = 0.5,
        start_threshold: float = 0.5,
        end_threshold: float = 0.25,
        min_speech_duration: float = 0.1,
        min_silence_duration: float = 0.5,
        padding_duration: float = 0.5,
        max_buffered_speech: float = 60.0,
        force_cpu: bool = True,
    ) -> None:
        if model_sample_rate not in SAMPLE_RATES:
            self.emit("error", f"Invalid model sample rate {model_sample_rate}: must be one of {SAMPLE_RATES}")
            raise ValueError(f"Model sample rate {model_sample_rate} not supported. Must be one of {SAMPLE_RATES}")

        super().__init__(
            sample_rate=model_sample_rate,
            threshold=threshold,
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration,
        )

        # Config properties
        self._start_thresh = start_threshold
        self._stop_thresh = end_threshold
        self._padding_sec = padding_duration
        self._max_buffer_sec = max_buffered_speech

        self._in_rate = input_sample_rate
        self._mod_rate = model_sample_rate
        self._requires_resample = input_sample_rate != model_sample_rate

        try:
            self._onnx_sess = VadModelWrapper.create_inference_session(force_cpu)
            self._silero = VadModelWrapper(session=self._onnx_sess, rate=model_sample_rate)
        except Exception as e:
            self.emit("error", f"Failed to init VAD model: {e}")
            raise

        # Smoothing
        self._smooth_factor = 0.35
        self._smoothed_prob = 0.0

        # State tracking
        self._speech_run_time = 0.0
        self._silence_run_time = 0.0

        self._is_active = False
        self._active_speech_time = 0.0
        self._active_silence_time = 0.0
        self._total_time = 0.0

        self._fract_offset = 0.0

        # Buffers
        self._raw_queue = np.array([], dtype=np.int16)
        self._model_queue = np.array([], dtype=np.float32)

        # Padding / Speech
        self._pad_frames = int(self._padding_sec * self._in_rate)
        buffer_size = int(self._max_buffer_sec * self._in_rate) + self._pad_frames
        self._audio_capture = np.empty(buffer_size, dtype=np.int16)
        self._capture_ptr = 0
        self._buffer_full = False

        self._lag_time = 0.0

    def _smooth_probability(self, val: float) -> float:
        self._smoothed_prob = (self._smooth_factor * val) + ((1 - self._smooth_factor) * self._smoothed_prob)
        return self._smoothed_prob

    def _flush_capture_buffer(self) -> None:
        if self._capture_ptr <= self._pad_frames:
            return

        retained = self._audio_capture[self._capture_ptr - self._pad_frames : self._capture_ptr].copy()
        self._buffer_full = False
        self._audio_capture[: self._pad_frames] = retained
        self._capture_ptr = self._pad_frames

    async def process_audio(self, audio_frames: bytes, **kwargs: Any) -> None:
        try:
            incoming = np.frombuffer(audio_frames, dtype=np.int16)

            self._raw_queue = np.concatenate([self._raw_queue, incoming])

            if self._requires_resample:
                normalized = incoming.astype(np.float32) / 32768.0
                target_len = int(len(normalized) * self._mod_rate / self._in_rate)
                if target_len > 0:
                    resampled = signal.resample(normalized, target_len)
                    self._model_queue = np.concatenate([self._model_queue, resampled.astype(np.float32)])
            else:
                normalized = incoming.astype(np.float32) / 32768.0
                self._model_queue = np.concatenate([self._model_queue, normalized])

            frame_size = self._silero.frame_size

            while len(self._model_queue) >= frame_size:
                t0 = time.perf_counter()
                chunk = self._model_queue[:frame_size]

                try:
                    p_raw = self._silero.process(chunk)
                except Exception as err:
                    self.emit("error", f"VAD error: {err}")
                    p_raw = 0.0

                p = self._smooth_probability(p_raw)
                step_time = frame_size / self._mod_rate
                self._total_time += step_time

                ratio = self._in_rate / self._mod_rate
                samples_needed = (frame_size * ratio) + self._fract_offset
                consume_count = int(samples_needed)
                self._fract_offset = samples_needed - consume_count

                space_left = len(self._audio_capture) - self._capture_ptr
                copy_amt = min(consume_count, space_left)

                if copy_amt > 0 and len(self._raw_queue) >= consume_count:
                    self._audio_capture[self._capture_ptr : self._capture_ptr + copy_amt] = self._raw_queue[:copy_amt]
                    self._capture_ptr += copy_amt
                elif not self._buffer_full:
                    self._buffer_full = True
                    logger.warning("VAD buffer full, dropping new samples")

                exec_time = time.perf_counter() - t0
                self._lag_time = max(0.0, self._lag_time + exec_time - step_time)
                if exec_time > INFERENCE_DELAY_TOLERANCE:
                    logger.warning(f"VAD slow: delay {self._lag_time:.3f}s")

                if self._is_active:
                    self._active_speech_time += step_time
                else:
                    self._active_silence_time += step_time

                if p >= self._start_thresh or (self._is_active and p > self._stop_thresh):
                    self._speech_run_time += step_time
                    self._silence_run_time = 0.0

                    if not self._is_active:
                        if self._speech_run_time >= self._min_speech_duration:
                            self._is_active = True
                            self._active_silence_time = 0.0
                            self._active_speech_time = self._speech_run_time
                            self._dispatch_event(VADEventType.START_OF_SPEECH)
                else:
                    self._silence_run_time += step_time
                    self._speech_run_time = 0.0

                    if not self._is_active:
                        self._flush_capture_buffer()

                    if self._is_active and self._silence_run_time >= self._min_silence_duration:
                        self._is_active = False
                        self._active_silence_time = self._silence_run_time
                        self._dispatch_event(VADEventType.END_OF_SPEECH)
                        self._active_speech_time = 0.0
                        self._flush_capture_buffer()

                if len(self._raw_queue) >= consume_count:
                    self._raw_queue = self._raw_queue[consume_count:]
                else:
                    self._raw_queue = np.array([], dtype=np.int16)

                self._model_queue = self._model_queue[frame_size:]

        except Exception as e:
            self.emit("error", f"VAD processing failed: {e}")

    def _dispatch_event(self, event_type: VADEventType) -> None:
        dur = self._active_speech_time
        if event_type == VADEventType.END_OF_SPEECH:
            dur = max(0.0, self._active_speech_time - self._silence_run_time)

        evt = VADResponse(
            event_type=event_type,
            data=VADData(
                is_speech=event_type == VADEventType.START_OF_SPEECH,
                confidence=self._smoothed_prob,
                timestamp=self._total_time,
                speech_duration=dur,
                silence_duration=self._active_silence_time,
            ),
        )
        if self._vad_callback:
            asyncio.create_task(self._vad_callback(evt))

    async def aclose(self) -> None:
        try:
            logger.info("SileroVAD garbage collection completed")

            self._raw_queue = np.array([], dtype=np.int16)
            self._model_queue = np.array([], dtype=np.float32)
            
            if hasattr(self, "_silero") and self._silero is not None:
                try:
                    if hasattr(self._silero, "_hidden_state"):
                        self._silero._hidden_state = None
                    if hasattr(self._silero, "_prev_context"):
                        self._silero._prev_context = None
                    if hasattr(self._silero, "_model_session"):
                        self._silero._model_session = None
                    self._silero = None
                except Exception as e:
                    logger.error(f"Error closing model: {e}")

            if hasattr(self, "_onnx_sess") and self._onnx_sess is not None:
                try:
                    self._onnx_sess = None
                except Exception as e:
                    logger.error(f"Error closing session: {e}")

            try:
                import gc
                gc.collect()
            except Exception as e:
                logger.error(f"GC error: {e}")

            await super().aclose()
        except Exception as e:
            self.emit("error", f"VAD close error: {e}")
