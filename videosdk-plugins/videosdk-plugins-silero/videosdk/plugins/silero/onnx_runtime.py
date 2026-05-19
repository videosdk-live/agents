import os
import logging
import urllib.request
import numpy as np
import onnxruntime
from pathlib import Path

logger = logging.getLogger(__name__)

_MODEL_DOWNLOAD_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
)
_CACHE_DIR = Path.home() / ".cache" / "videosdk" / "silero"

SAMPLE_RATES = [8000, 16000]


class VadModelWrapper:
    """Wraps an ONNX Runtime session for Silero VAD inference.

    Uses pre-allocated numpy buffers so that each ``process()`` call
    performs zero heap allocations — context and audio are written
    in-place into a fixed buffer before being fed to the model.
    """

    def __init__(self, *, session: onnxruntime.InferenceSession, rate: int) -> None:
        if rate not in SAMPLE_RATES:
            raise ValueError(f"Rate {rate} not supported; use 8000 or 16000")

        self._model_session = session
        self._audio_rate = rate
        self._frame_size = 256 if rate == 8000 else 512
        self._history_len = 32 if rate == 8000 else 64

        self._hidden_state = np.zeros((2, 1, 128), dtype=np.float32)
        self._prev_context = np.zeros((1, self._history_len), dtype=np.float32)
        self._input_buffer = np.zeros(
            (1, self._history_len + self._frame_size), dtype=np.float32
        )
        self._sample_rate_nd = np.array([rate], dtype=np.int64)

    def reset_state(self) -> None:
        """Reset hidden state and context to initial values.

        Call when audio continuity is broken (e.g., after buffer flush)
        to prevent the model from processing discontinuous audio.
        """
        self._hidden_state[:] = 0.0
        self._prev_context[:] = 0.0
        self._input_buffer[:] = 0.0

    @property
    def frame_size(self) -> int:
        return self._frame_size

    @property
    def history_len(self) -> int:
        return self._history_len

    def process(self, input_audio: np.ndarray) -> float:
        """Run a single inference on a chunk of audio.

        The input is written into a pre-allocated buffer alongside the
        previous context so that no ``np.concatenate`` allocation occurs.
        """
        if input_audio.ndim == 1:
            input_audio = input_audio.reshape(1, -1)
        if input_audio.ndim > 2:
            raise ValueError(f"Too many dimensions for input audio chunk {input_audio.ndim}")

        if self._audio_rate / input_audio.shape[1] > 31.25:
            raise ValueError("Input audio chunk is too short")

        num_samples = self._frame_size
        if input_audio.shape[-1] != num_samples:
            raise ValueError(f"Provided number of samples is {input_audio.shape[-1]} (Supported values: 256 for 8000 sample rate, 512 for 16000)")

        self._input_buffer[:, :self._history_len] = self._prev_context
        self._input_buffer[:, self._history_len:] = input_audio

        inputs = {
            "input": self._input_buffer,
            "state": self._hidden_state,
            "sr": self._sample_rate_nd,
        }

        prob, state = self._model_session.run(None, inputs)

        self._hidden_state = state
        self._prev_context[:] = self._input_buffer[:, -self._history_len:]

        return prob.item()

    @staticmethod
    def create_inference_session(
        use_cpu_only: bool,
        onnx_file_path: str | Path | None = None,
    ) -> onnxruntime.InferenceSession:
        """Create an optimised ONNX Runtime InferenceSession.

        Improvements over v1:
        * Thread spinning disabled — no busy-wait when idle.
        * Sequential execution mode — deterministic ordering.
        * Custom model path support — bring your own fine-tuned model.

        Resolution order: onnx_file_path -> cached download (~/.cache/videosdk/silero).
        """
        session_opts = onnxruntime.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1
        session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        session_opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        session_opts.add_session_config_entry("session.inter_op.allow_spinning", "0")

        providers = (
            ["CPUExecutionProvider"]
            if use_cpu_only and "CPUExecutionProvider" in onnxruntime.get_available_providers()
            else None
        )

        if onnx_file_path is not None:
            onnx_file_path = Path(onnx_file_path)
            if not onnx_file_path.exists():
                raise FileNotFoundError(f"Silero VAD model file not found: {onnx_file_path}")
            return onnxruntime.InferenceSession(
                str(onnx_file_path), sess_options=session_opts, providers=providers
            )

        cached = str(_CACHE_DIR / "silero_vad.onnx")
        if not os.path.exists(cached):
            os.makedirs(str(_CACHE_DIR), exist_ok=True)
            urllib.request.urlretrieve(_MODEL_DOWNLOAD_URL, cached)
            logger.info("Silero VAD model downloaded")

        return onnxruntime.InferenceSession(cached, sess_options=session_opts, providers=providers)