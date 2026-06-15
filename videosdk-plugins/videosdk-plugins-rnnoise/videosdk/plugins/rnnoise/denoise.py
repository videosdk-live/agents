import asyncio
import logging
import time
from math import gcd
from typing import Any

import numpy as np
from scipy.signal import resample_poly

from videosdk.agents.denoise import Denoise
from .rnnoise import RNN

logger = logging.getLogger(__name__)

SLOW_DENOISE_THRESHOLD_MS = 20.0


class RNNoise(Denoise):
    def __init__(self):
        """Initialize the RNNoise denoise plugin.
        """
        super().__init__()
        self.rnnoise = RNN()
        self._target_sample_rate = 48000
        self._frame_duration_ms = 20
        self._rnnoise_frame_size = 480

    async def denoise(self, audio_frames: bytes, **kwargs: Any) -> bytes:
        if not audio_frames:
            return b""
        return await asyncio.to_thread(self._denoise_sync, audio_frames)

    def _denoise_sync(self, audio_frames: bytes) -> bytes:
        t0 = time.perf_counter()

        audio_np = np.frombuffer(audio_frames, dtype=np.int16)
        num_samples = len(audio_np)
        if num_samples == 0:
            return b""
        original_sample_rate = int(
            num_samples * 1000 / self._frame_duration_ms)

        if original_sample_rate != self._target_sample_rate:
            audio_float = audio_np.astype(np.float32) / 32768.0
            up, down = self._resample_factors(
                original_sample_rate, self._target_sample_rate)
            resampled_audio_float = resample_poly(audio_float, up, down)
            resampled_audio_np = self._to_int16(resampled_audio_float)
        else:
            resampled_audio_np = audio_np

        num_rnnoise_frames = len(
            resampled_audio_np) // self._rnnoise_frame_size
        denoised_chunks = []

        for i in range(num_rnnoise_frames):
            start = i * self._rnnoise_frame_size
            end = start + self._rnnoise_frame_size
            chunk = resampled_audio_np[start:end]

            if len(chunk) != self._rnnoise_frame_size:
                continue

            chunk_bytes = chunk.tobytes()
            _vod_prob, denoised_chunk_bytes = self.rnnoise.process_frame(
                chunk_bytes)
            denoised_chunk_np = np.frombuffer(
                denoised_chunk_bytes, dtype=np.int16)
            denoised_chunks.append(denoised_chunk_np)

        if not denoised_chunks:
            return b""

        denoised_audio_np = np.concatenate(denoised_chunks)

        if original_sample_rate != self._target_sample_rate:    
            denoised_float = denoised_audio_np.astype(np.float32) / 32768.0
            up, down = self._resample_factors(
                original_sample_rate, self._target_sample_rate)

            original_format_float = resample_poly(denoised_float, down, up)
            final_audio_np = self._to_int16(original_format_float)
            
        else:
            final_audio_np = denoised_audio_np

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if elapsed_ms > SLOW_DENOISE_THRESHOLD_MS:
            logger.warning(
                "RNNoise slow: denoise pass took %.1f ms (>%.0f ms budget) "
                "for %d samples @ %d Hz",
                elapsed_ms, SLOW_DENOISE_THRESHOLD_MS,
                num_samples, original_sample_rate,
            )

        return final_audio_np.tobytes()

    @staticmethod
    def _resample_factors(orig_rate: int, target_rate: int) -> tuple[int, int]:
        """Integer (up, down) polyphase factors. For 24k<->48k this is (2, 1)."""
        g = gcd(orig_rate, target_rate)
        return target_rate // g, orig_rate // g

    @staticmethod
    def _to_int16(audio_float: np.ndarray) -> np.ndarray:
        """Clip a float32 [-1, 1] signal and convert to int16 PCM."""
        clipped = np.clip(audio_float, -1.0, 1.0)
        return (clipped * 32767.0).astype(np.int16)

    async def aclose(self) -> None:
        self.rnnoise.destroy()
        await super().aclose()
