from typing import Any
from videosdk.agents.denoise import Denoise
from .rnnoise import RNN
import numpy as np
import resampy

class RNNoise(Denoise):
    def __init__(self):
        super().__init__()
        self.rnnoise = RNN()
        self._target_sample_rate = 48000
        self._frame_duration_ms = 20 
        self._rnnoise_frame_size = 480

    async def denoise(self, audio_frames: bytes, **kwargs: Any) -> bytes:
        if not audio_frames:
            return b""

        audio_np = np.frombuffer(audio_frames, dtype=np.int16)
        num_samples = len(audio_np)
        original_sample_rate = int(num_samples * 1000 / self._frame_duration_ms)

        if original_sample_rate != self._target_sample_rate:
            audio_float = audio_np.astype(np.float32) / 32767.0
            resampled_audio_float = resampy.resample(audio_float, sr_orig=original_sample_rate, sr_new=self._target_sample_rate)
            resampled_audio_np = (resampled_audio_float * 32767.0).astype(np.int16)
        else:
            resampled_audio_np = audio_np

        num_rnnoise_frames = len(resampled_audio_np) // self._rnnoise_frame_size
        denoised_chunks = []

        for i in range(num_rnnoise_frames):
            start = i * self._rnnoise_frame_size
            end = start + self._rnnoise_frame_size
            chunk = resampled_audio_np[start:end]
            
            if len(chunk) != self._rnnoise_frame_size:
                continue

            chunk_bytes = chunk.tobytes()
            _vod_prob, denoised_chunk_bytes = self.rnnoise.process_frame(chunk_bytes)
            denoised_chunk_np = np.frombuffer(denoised_chunk_bytes, dtype=np.int16)
            denoised_chunks.append(denoised_chunk_np)

        if not denoised_chunks:
            return b""

        denoised_audio_np = np.concatenate(denoised_chunks)

        if original_sample_rate != self._target_sample_rate:
            denoised_float = denoised_audio_np.astype(np.float32) / 32767.0
            original_format_float = resampy.resample(denoised_float, sr_orig=self._target_sample_rate, sr_new=original_sample_rate)
            final_audio_np = (original_format_float * 32767.0).astype(np.int16)
        else:
            final_audio_np = denoised_audio_np

        return final_audio_np.tobytes()

    async def aclose(self) -> None:
        self.rnnoise.destroy()
        await super().aclose()