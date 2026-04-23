from __future__ import annotations

import asyncio
import collections
import concurrent.futures
import time
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

import numpy as np
from scipy import signal

from .onnx_runtime import SAMPLE_RATES, VadModelWrapper
from videosdk.agents.vad import VAD as BaseVAD, VADData, VADEventType, VADResponse

import logging

logger = logging.getLogger(__name__)

SLOW_INFERENCE_THRESHOLD = 0.2

class _VADPhase(Enum):
    IDLE = 0
    PENDING = 1
    ACTIVE = 2
    TRAILING = 3

class _EMAFilter:
    __slots__ = ("_alpha", "_value")

    def __init__(self, alpha: float) -> None:
        self._alpha = alpha
        self._value = 0.0

    def apply(self, sample: float) -> float:
        self._value = self._alpha * sample + (1.0 - self._alpha) * self._value
        return self._value

    def reset(self) -> None:
        self._value = 0.0


class _MovingAverageFilter:
    __slots__ = ("_buf",)

    def __init__(self, window: int) -> None:
        self._buf: collections.deque[float] = collections.deque(maxlen=max(window, 1))

    def apply(self, sample: float) -> float:
        self._buf.append(sample)
        return sum(self._buf) / len(self._buf)

    def reset(self) -> None:
        self._buf.clear()


class _PassthroughFilter:
    __slots__ = ()

    @staticmethod
    def apply(sample: float) -> float:
        return sample

    @staticmethod
    def reset() -> None:
        pass

class SileroVAD(BaseVAD):
    """Silero Voice Activity Detection with advanced streaming features.

    All new parameters default to values that reproduce v1 behaviour.
    """

    def __init__(
        self,
        input_sample_rate: int = 48000,
        model_sample_rate: Literal[8000, 16000] = 16000,
        threshold: float = 0.5,
        start_threshold: float = 0.4,
        end_threshold: float = 0.25,
        min_speech_duration: float = 0.3,
        min_silence_duration: float = 0.4,
        padding_duration: float = 0.5,
        max_buffered_speech: float = 60.0,
        force_cpu: bool = True,
        onnx_model_path: str | Path | None = None,
        max_speech_duration: float | None = None,
        min_silence_at_split: float = 0.098,
        energy_filter_enabled: bool = False,
        energy_silence_threshold: float = 0.001,
        smoothing_strategy: Literal["ema", "moving_average", "none"] = "ema",
        smoothing_factor: float = 0.35,
        smoothing_window: int = 5,
        min_volume: float = 0.0,
        probability_history_size: int = 0,
        offload_inference: bool = False,
    ) -> None:
        if model_sample_rate not in SAMPLE_RATES:
            raise ValueError(
                f"Model sample rate {model_sample_rate} not supported. "
                f"Must be one of {SAMPLE_RATES}"
            )

        super().__init__(
            sample_rate=model_sample_rate,
            threshold=threshold,
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration,
        )

        self._start_thresh = start_threshold
        self._stop_thresh = end_threshold
        self._min_volume = min_volume
        self._padding_sec = padding_duration
        self._max_buffer_sec = max_buffered_speech

        self._in_rate = input_sample_rate
        self._mod_rate = model_sample_rate
        self._requires_resample = input_sample_rate != model_sample_rate

        try:
            self._onnx_sess = VadModelWrapper.create_inference_session(
                force_cpu, onnx_file_path=onnx_model_path
            )
            self._silero = VadModelWrapper(session=self._onnx_sess, rate=model_sample_rate)
        except Exception as e:
            self.emit("error", f"Failed to init VAD model: {e}")
            raise

        if smoothing_strategy == "ema":
            self._smoother: _EMAFilter | _MovingAverageFilter | _PassthroughFilter = _EMAFilter(smoothing_factor)
        elif smoothing_strategy == "moving_average":
            self._smoother = _MovingAverageFilter(smoothing_window)
        else:
            self._smoother = _PassthroughFilter()

        self._phase = _VADPhase.IDLE
        self._speech_run_time = 0.0
        self._silence_run_time = 0.0
        self._active_speech_time = 0.0
        self._active_silence_time = 0.0
        self._total_time = 0.0
        self._total_samples = 0

        # Max speech splitting
        self._max_speech_duration = max_speech_duration
        self._min_silence_at_split = min_silence_at_split
        self._split_candidates: list[tuple[float, float]] = []
        self._speech_start_time = 0.0
        self._temp_silence_start: float | None = None

        # Energy pre-filter
        self._energy_filter_enabled = energy_filter_enabled
        self._energy_silence_threshold = energy_silence_threshold
        self._noise_floor = 0.0

        # Inference offloading
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        if offload_inference:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="vad-inference"
            )

        # Per-frame values
        self._last_raw_prob = 0.0
        self._last_smoothed_prob = 0.0
        self._last_energy = 0.0
        self._last_inference_ms = 0.0
        self._inference_skipped = False

        # Audio queues
        self._fract_offset = 0.0
        self._raw_queue = np.array([], dtype=np.int16)
        self._model_queue = np.array([], dtype=np.float32)

        # Capture buffer
        self._pad_frames = int(self._padding_sec * self._in_rate)
        buffer_size = int(self._max_buffer_sec * self._in_rate) + self._pad_frames
        self._audio_capture = np.empty(buffer_size, dtype=np.int16)
        self._capture_ptr = 0
        self._buffer_full = False

        self._lag_time = 0.0

        # Probability ring buffer
        self._prob_ring: collections.deque[tuple[float, float, float, float]] | None = None
        if probability_history_size > 0:
            self._prob_ring = collections.deque(maxlen=probability_history_size)

        # Optional callbacks
        self._inference_callback: Callable[[VADResponse], Awaitable[None]] | None = None
        self._metrics_callback: Callable[[dict], None] | None = None
        self._inference_count = 0


    def on_inference(self, callback: Callable[[VADResponse], Awaitable[None] | None]) -> None:
        """Register callback for FRAME_PROCESSED events (~31/sec at 16 kHz).

        The callback is invoked directly (not via asyncio.create_task)
        to avoid flooding the event loop with ~31 micro-tasks per second.
        If the callback is a coroutine function it will still be awaited
        inline within ``process_audio``.
        """
        self._inference_callback = callback

    def on_metrics(self, callback: Callable[[dict], None]) -> None:
        """Register lightweight sync callback for per-frame metrics."""
        self._metrics_callback = callback

    @property
    def probability_history(self) -> list[tuple[float, float, float, float]]:
        """Recent inference history as (timestamp, raw, smoothed, energy) tuples."""
        if self._prob_ring is None:
            return []
        return list(self._prob_ring)

    @staticmethod
    def _compute_energy(chunk: np.ndarray) -> float:
        return float(np.sqrt(np.mean(chunk * chunk)))

    def _flush_capture_buffer(self, reset_model: bool = False) -> None:
        """Retain only the padding portion of the capture buffer.

        Args:
            reset_model: When True, also reset the ONNX LSTM state and
                smoother.  Should only be set after a *confirmed*
                END_OF_SPEECH — never during the PENDING phase, because
                resetting the warm model/smoother state mid-speech
                prevents the probability from ever building up enough
                to cross the min_speech_duration threshold.
        """
        self._buffer_full = False
        if self._capture_ptr <= self._pad_frames:
            if reset_model:
                self._silero.reset_state()
                self._smoother.reset()
            return
        retained = self._audio_capture[
            self._capture_ptr - self._pad_frames : self._capture_ptr
        ].copy()
        self._audio_capture[: self._pad_frames] = retained
        self._capture_ptr = self._pad_frames
        if reset_model:
            self._silero.reset_state()
            self._smoother.reset()

    def _copy_capture_audio(self) -> bytes | None:
        if self._capture_ptr <= 0:
            return None
        return self._audio_capture[: self._capture_ptr].tobytes()

    def _split_capture_at_offset(self, keep_from_samples: int) -> None:
        if keep_from_samples <= 0 or keep_from_samples >= self._capture_ptr:
            self._flush_capture_buffer()
            return
        remaining = self._capture_ptr - keep_from_samples
        self._audio_capture[:remaining] = self._audio_capture[
            keep_from_samples : self._capture_ptr
        ]
        self._capture_ptr = remaining
        self._buffer_full = False

    def _dispatch_vad_event(self, event_type: VADEventType) -> None:
        dur = self._active_speech_time
        if event_type == VADEventType.END_OF_SPEECH:
            dur = max(0.0, self._active_speech_time - self._silence_run_time)

        audio = self._copy_capture_audio()

        evt = VADResponse(
            event_type=event_type,
            data=VADData(
                is_speech=event_type == VADEventType.START_OF_SPEECH,
                confidence=self._last_smoothed_prob,
                timestamp=self._total_time,
                speech_duration=dur,
                silence_duration=self._active_silence_time,
                audio_frames=audio,
                raw_probability=self._last_raw_prob,
                inference_duration_ms=self._last_inference_ms,
                energy=self._last_energy,
                samples_index=self._total_samples,
            ),
        )
        callback = self._vad_callback
        if callback:
            task = asyncio.create_task(callback(evt))
            task.add_done_callback(self._on_task_done)

    def _dispatch_frame_event(self) -> None:
        """Dispatch FRAME_PROCESSED directly (no asyncio.create_task).

        Called ~31 times/sec — creating a task per frame would flood
        the event loop.  Instead we update the callback synchronously
        since the handler only sets a few scalar values.
        """
        cb = self._inference_callback
        if cb is None:
            return

        evt = VADResponse(
            event_type=VADEventType.FRAME_PROCESSED,
            data=VADData(
                is_speech=self._phase in (_VADPhase.ACTIVE, _VADPhase.TRAILING),
                confidence=self._last_smoothed_prob,
                timestamp=self._total_time,
                speech_duration=self._active_speech_time,
                silence_duration=self._active_silence_time,
                raw_probability=self._last_raw_prob,
                inference_duration_ms=self._last_inference_ms,
                energy=self._last_energy,
                samples_index=self._total_samples,
            ),
        )
        try:
            result = cb(evt)
            if result is not None and asyncio.iscoroutine(result):
                task = asyncio.create_task(result)
                task.add_done_callback(self._on_task_done)
        except Exception as e:
            logger.error(f"FRAME_PROCESSED callback error: {e}")

    def _dispatch_metrics(self) -> None:
        cb = self._metrics_callback
        if cb is None:
            return
        self._inference_count += 1
        cb({
            "timestamp": self._total_time,
            "probability": self._last_raw_prob,
            "smoothed_probability": self._last_smoothed_prob,
            "energy": self._last_energy,
            "inference_duration_ms": self._last_inference_ms,
            "state": self._phase.name,
            "speaking": self._phase in (_VADPhase.ACTIVE, _VADPhase.TRAILING),
            "inference_skipped": self._inference_skipped,
            "inference_count": self._inference_count,
        })

    @staticmethod
    def _on_task_done(task: asyncio.Task) -> None:
        if not task.cancelled() and task.exception():
            logger.error(
                f"VAD callback failed: {task.exception()}",
                exc_info=task.exception(),
            )

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _transition(self, is_speech_frame: bool, step_time: float) -> None:
        phase = self._phase

        if is_speech_frame:
            self._speech_run_time += step_time
            self._silence_run_time = 0.0

            if self._temp_silence_start is not None:
                sil_dur = self._total_time - self._temp_silence_start
                if sil_dur >= self._min_silence_at_split:
                    self._split_candidates.append((self._temp_silence_start, sil_dur))
                self._temp_silence_start = None

            if phase == _VADPhase.IDLE:
                self._phase = _VADPhase.PENDING
            elif phase == _VADPhase.PENDING:
                if self._speech_run_time >= self._min_speech_duration:
                    self._phase = _VADPhase.ACTIVE
                    self._active_silence_time = 0.0
                    self._active_speech_time = self._speech_run_time
                    self._speech_start_time = self._total_time - self._speech_run_time
                    self._split_candidates.clear()
                    self._dispatch_vad_event(VADEventType.START_OF_SPEECH)
                    logger.info("[VAD] START_OF_SPEECH")
            elif phase == _VADPhase.TRAILING:
                self._phase = _VADPhase.ACTIVE
        else:
            self._silence_run_time += step_time
            self._speech_run_time = 0.0

            if phase == _VADPhase.PENDING:
                self._phase = _VADPhase.IDLE
                self._flush_capture_buffer(reset_model=False)
            elif phase == _VADPhase.IDLE:
                self._flush_capture_buffer(reset_model=False)
            elif phase == _VADPhase.ACTIVE:
                self._phase = _VADPhase.TRAILING
                if self._temp_silence_start is None:
                    self._temp_silence_start = self._total_time
            elif phase == _VADPhase.TRAILING:
                if self._temp_silence_start is None:
                    self._temp_silence_start = self._total_time
                if self._silence_run_time >= self._min_silence_duration:
                    # Confirmed end of speech — safe to reset everything.
                    self._phase = _VADPhase.IDLE
                    self._active_silence_time = self._silence_run_time
                    self._dispatch_vad_event(VADEventType.END_OF_SPEECH)
                    logger.info("[VAD] END_OF_SPEECH")
                    self._active_speech_time = 0.0
                    self._temp_silence_start = None
                    self._split_candidates.clear()
                    self._flush_capture_buffer(reset_model=True)

        if self._phase in (_VADPhase.ACTIVE, _VADPhase.TRAILING):
            self._active_speech_time += step_time
        elif self._phase == _VADPhase.IDLE:
            self._active_silence_time += step_time

        # Max speech splitting
        if (
            self._max_speech_duration is not None
            and self._phase in (_VADPhase.ACTIVE, _VADPhase.TRAILING)
        ):
            speech_len = self._total_time - self._speech_start_time
            if speech_len >= self._max_speech_duration:
                self._handle_max_speech_split()

    def _handle_max_speech_split(self) -> None:
        if self._split_candidates:
            best_ts, best_dur = max(self._split_candidates, key=lambda x: x[1])
            logger.info(f"[VAD] Max speech split at silence t={best_ts:.3f}s dur={best_dur:.3f}s")
            self._dispatch_vad_event(VADEventType.END_OF_SPEECH)
            elapsed_since_split = self._total_time - (best_ts + best_dur)
            keep_samples = int(elapsed_since_split * self._in_rate)
            self._split_capture_at_offset(max(0, self._capture_ptr - keep_samples))
        else:
            logger.info("[VAD] Max speech split (hard, no silence candidate)")
            self._dispatch_vad_event(VADEventType.END_OF_SPEECH)
            self._flush_capture_buffer(reset_model=False)

        self._speech_start_time = self._total_time
        self._split_candidates.clear()
        self._temp_silence_start = None
        self._active_speech_time = 0.0
        self._phase = _VADPhase.ACTIVE
        self._dispatch_vad_event(VADEventType.START_OF_SPEECH)
        logger.info("[VAD] START_OF_SPEECH (continuation after split)")

    async def process_audio(self, audio_frames: bytes, **kwargs: Any) -> None:
        try:
            if not audio_frames:
                return
            incoming = np.frombuffer(audio_frames, dtype=np.int16)
            if len(incoming) == 0:
                return
            self._raw_queue = np.concatenate([self._raw_queue, incoming])

            if self._requires_resample:
                normalized = incoming.astype(np.float32) / 32768.0
                target_len = int(len(normalized) * self._mod_rate / self._in_rate)
                if target_len > 0:
                    resampled = signal.resample(normalized, target_len)
                    self._model_queue = np.concatenate(
                        [self._model_queue, resampled.astype(np.float32)]
                    )
            else:
                normalized = incoming.astype(np.float32) / 32768.0
                self._model_queue = np.concatenate([self._model_queue, normalized])

            frame_size = self._silero.frame_size

            while len(self._model_queue) >= frame_size:
                t0 = time.perf_counter()
                chunk = self._model_queue[:frame_size]

                # Energy
                energy = self._compute_energy(chunk)
                self._last_energy = energy

                # Energy pre-filter
                if self._energy_filter_enabled and self._phase == _VADPhase.IDLE:
                    if energy > 0:
                        self._noise_floor = 0.995 * self._noise_floor + 0.005 * energy
                    effective_threshold = max(
                        self._energy_silence_threshold, self._noise_floor * 3.0
                    )
                    if energy < effective_threshold:
                        p_raw = 0.0
                        self._inference_skipped = True
                    else:
                        p_raw = await self._run_inference(chunk)
                        self._inference_skipped = False
                else:
                    p_raw = await self._run_inference(chunk)
                    self._inference_skipped = False

                inference_time = time.perf_counter() - t0
                self._last_inference_ms = inference_time * 1000.0
                self._last_raw_prob = p_raw

                # Smoothing
                p_smoothed = self._smoother.apply(p_raw)
                self._last_smoothed_prob = p_smoothed

                # Timing
                step_time = frame_size / self._mod_rate
                self._total_time += step_time
                self._total_samples += frame_size

                # Consume matching raw-rate samples
                ratio = self._in_rate / self._mod_rate
                samples_needed = (frame_size * ratio) + self._fract_offset
                consume_count = int(samples_needed)
                self._fract_offset = samples_needed - consume_count
                if self._fract_offset > ratio:
                    self._fract_offset = 0.0

                space_left = len(self._audio_capture) - self._capture_ptr
                copy_amt = min(consume_count, space_left)

                if copy_amt > 0 and len(self._raw_queue) >= consume_count:
                    self._audio_capture[
                        self._capture_ptr : self._capture_ptr + copy_amt
                    ] = self._raw_queue[:copy_amt]
                    self._capture_ptr += copy_amt
                elif copy_amt == 0 and not self._buffer_full:
                    self._buffer_full = True
                    logger.warning("VAD buffer full, dropping new samples")

                # Lag tracking
                self._lag_time = max(0.0, self._lag_time + inference_time - step_time)
                if inference_time > SLOW_INFERENCE_THRESHOLD:
                    logger.warning(f"VAD slow: delay {self._lag_time:.3f}s")

                # Speech decision
                is_speech_frame = p_smoothed >= self._start_thresh or (
                    self._phase in (_VADPhase.ACTIVE, _VADPhase.TRAILING, _VADPhase.PENDING)
                    and p_smoothed > self._stop_thresh
                )
                if self._min_volume > 0.0:
                    is_speech_frame = is_speech_frame and energy >= self._min_volume

                # State machine
                self._transition(is_speech_frame, step_time)

                # Per-frame dispatches
                self._dispatch_frame_event()
                self._dispatch_metrics()

                # Ring buffer
                if self._prob_ring is not None:
                    self._prob_ring.append((self._total_time, p_raw, p_smoothed, energy))

                # Advance queues
                if len(self._raw_queue) >= consume_count:
                    self._raw_queue = self._raw_queue[consume_count:]
                else:
                    self._raw_queue = np.array([], dtype=np.int16)
                self._model_queue = self._model_queue[frame_size:]

        except Exception as e:
            logger.error(f"VAD processing failed: {e}", exc_info=True)
            self.emit("error", f"VAD processing failed: {e}")

    async def _run_inference(self, chunk: np.ndarray) -> float:
        if self._executor is not None:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._executor, self._silero.process, chunk)
        return self._silero.process(chunk)

    async def flush(self) -> None:
        self._raw_queue = np.array([], dtype=np.int16)
        self._model_queue = np.array([], dtype=np.float32)
        self._fract_offset = 0.0
        if self._silero:
            self._silero.reset_state()
        self._smoother.reset()
        self._phase = _VADPhase.IDLE
        self._speech_run_time = 0.0
        self._silence_run_time = 0.0
        self._split_candidates.clear()
        self._temp_silence_start = None

    async def aclose(self) -> None:
        try:
            if self._executor is not None:
                self._executor.shutdown(wait=False)
                self._executor = None

            self._raw_queue = np.array([], dtype=np.int16)
            self._model_queue = np.array([], dtype=np.float32)

            if self._prob_ring is not None:
                self._prob_ring.clear()
            self._split_candidates.clear()

            if hasattr(self, "_silero") and self._silero is not None:
                try:
                    self._silero._hidden_state = None
                    self._silero._prev_context = None
                    self._silero._input_buffer = None
                    self._silero._model_session = None
                    self._silero = None
                except Exception as e:
                    logger.error(f"Error closing model: {e}")

            if hasattr(self, "_onnx_sess") and self._onnx_sess is not None:
                self._onnx_sess = None

            self._inference_callback = None
            self._metrics_callback = None

            await super().aclose()
        except Exception as e:
            self.emit("error", f"VAD close error: {e}")