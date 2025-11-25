import numpy as np
import cv2
import time


class CircularGlowVisualizer:
    def __init__(self, sample_rate=24000, n_fft=512, freq_bands=64):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.freq_bands = freq_bands
        self.prev_fft = np.zeros(freq_bands)
        self.smooth = 0.25
        self.noise_gate = 0.02
        self.start_time = time.time()

    def _compute_fft(self, audio_samples: np.ndarray):
        audio = audio_samples.astype(np.float32) / 32767.0
        audio = audio.mean(axis=1)

        if len(audio) < self.n_fft:
            return 0.0, np.zeros(self.freq_bands)

        window = np.hanning(self.n_fft)
        fft_vals = np.abs(np.fft.rfft(audio[:self.n_fft] * window))

        volume = float(np.sqrt(np.mean(np.square(fft_vals))))
        volume = np.clip(volume * 2.0, 0, 1)

        fft_vals = 20 * np.log10(fft_vals + 1e-10)
        fft_vals = np.clip((fft_vals + 80) / 80, 0, 1)

        bands = np.array_split(fft_vals, self.freq_bands)
        band_vals = np.array([b.mean() for b in bands])

        self.prev_fft = self.prev_fft * (1 - self.smooth) + band_vals * self.smooth

        if volume < self.noise_gate:
            return 0.0, np.zeros_like(self.prev_fft)

        return volume, self.prev_fft

    def draw(self, canvas: np.ndarray, audio_samples: np.ndarray, fps=None):
        h, w = canvas.shape[:2]
        cx, cy = w // 2, h // 2
        radius = min(w, h) // 4

        volume, fft_vals = self._compute_fft(audio_samples)

        pulse = int(radius * (0.1 + volume * 0.4))
        cv2.circle(canvas, (cx, cy), radius + pulse, (0, 200, 255, 255), 8)

        angles = np.linspace(0, 2 * np.pi, self.freq_bands)
        max_len = radius * 0.9

        for i, a in enumerate(angles):
            intensity = fft_vals[i]
            if intensity <= 0:
                continue

            line_len = int(max_len * intensity)

            x1 = int(cx + np.cos(a) * (radius - 20))
            y1 = int(cy + np.sin(a) * (radius - 20))
            x2 = int(cx + np.cos(a) * (radius - 20 + line_len))
            y2 = int(cy + np.sin(a) * (radius - 20 + line_len))

            cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 180, 255), 3)

        if fps is not None:
            txt = f"{time.time() - self.start_time:.1f}s @ {fps:.1f} fps"
        else:
            txt = f"{time.time() - self.start_time:.1f}s"

        cv2.putText(canvas, txt, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)