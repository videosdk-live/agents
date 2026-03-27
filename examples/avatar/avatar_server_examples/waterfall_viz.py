"""
Waterfall Spectrogram Visualizer

Scrolls FFT bands upward over time, painting each row with a heat-map color.

Dependencies: numpy, opencv-python-headless (cv2)
"""

import time

import cv2
import numpy as np

class WaterfallVisualizer:
    def __init__(self, sample_rate: int = 24000, n_fft: int = 512, freq_bands: int = 64):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.freq_bands = freq_bands
        self.prev_fft = np.zeros(freq_bands)
        self.smooth = 0.25
        self.noise_gate = 0.02
        self.start_time = time.time()
        self._waterfall: np.ndarray | None = None  

    def _compute_fft(self, audio_samples: np.ndarray):
        audio = audio_samples.astype(np.float32) / 32767.0
        audio = audio.mean(axis=1)
        if len(audio) < self.n_fft:
            return 0.0, np.zeros(self.freq_bands)

        fft_vals = np.abs(np.fft.rfft(audio[: self.n_fft] * np.hanning(self.n_fft)))
        volume = float(np.clip(np.sqrt(np.mean(np.square(fft_vals))) * 2.0, 0, 1))
        fft_vals = np.clip((20 * np.log10(fft_vals + 1e-10) + 80) / 80, 0, 1)
        band_vals = np.array([b.mean() for b in np.array_split(fft_vals, self.freq_bands)])
        self.prev_fft = self.prev_fft * (1 - self.smooth) + band_vals * self.smooth

        if volume < self.noise_gate:
            return 0.0, np.zeros_like(self.prev_fft)
        return volume, self.prev_fft

    def draw(self, canvas: np.ndarray, audio_samples: np.ndarray, fps=None) -> None:
        h, w = canvas.shape[:2]

        if self._waterfall is None or self._waterfall.shape[:2] != (h, w):
            self._waterfall = np.zeros((h, w, 3), dtype=np.uint8)

        _, fft_vals = self._compute_fft(audio_samples)

        new_row = np.zeros((1, w, 3), dtype=np.uint8)
        band_w = w / self.freq_bands
        for i, v in enumerate(fft_vals):
            x0, x1 = int(i * band_w), int((i + 1) * band_w)
            intensity = int(v * 255)
            color_px = cv2.applyColorMap(np.array([[intensity]], dtype=np.uint8), cv2.COLORMAP_INFERNO)[0, 0]
            new_row[0, x0:x1] = color_px

        self._waterfall[:-1] = self._waterfall[1:]
        self._waterfall[-1] = new_row

        canvas[:, :, :3] = self._waterfall
        canvas[:, :, 3] = 255

        elapsed = time.time() - self.start_time
        label = f"{elapsed:.1f}s @ {fps:.1f} fps" if fps is not None else f"{elapsed:.1f}s"
        cv2.putText(canvas, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
