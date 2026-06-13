from __future__ import annotations

import numpy as np


def resample_fft(x: np.ndarray, num: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n_in = x.shape[0]
    if n_in == 0 or num <= 0:
        return np.zeros(max(num, 0), dtype=np.float64)
    if num == n_in:
        return x.copy()

    X = np.fft.rfft(x)
    new_bins = num // 2 + 1
    Y = np.zeros(new_bins, dtype=complex)

    n = min(num, n_in)
    nyq = n // 2 + 1
    Y[:nyq] = X[:nyq]

    if n % 2 == 0:
        if num < n_in:
            Y[n // 2] *= 2.0
        elif num > n_in:
            Y[n // 2] *= 0.5

    y = np.fft.irfft(Y, num)
    y *= float(num) / float(n_in)
    return y


def target_length(n_in: int, in_rate: int, out_rate: int) -> int:
    if in_rate <= 0:
        return 0
    return int(n_in * out_rate / in_rate)


def resample_mono_f32(
    samples: np.ndarray,
    in_rate: int,
    out_rate: int,
) -> np.ndarray:
    if samples.size == 0 or in_rate == out_rate:
        return np.ascontiguousarray(samples, dtype=np.float32)

    out_len = target_length(samples.size, in_rate, out_rate)
    if out_len <= 0:
        return np.zeros(0, dtype=np.float32)

    src_positions = np.arange(out_len, dtype=np.float64) * (in_rate / out_rate)
    src_index = np.arange(samples.size, dtype=np.float64)
    resampled = np.interp(src_positions, src_index, samples.astype(np.float64))
    return resampled.astype(np.float32)
