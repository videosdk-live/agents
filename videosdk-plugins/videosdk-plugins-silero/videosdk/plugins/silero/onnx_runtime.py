import os
import logging
import threading
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

_session_cache: dict[tuple, onnxruntime.InferenceSession] = {}
_session_cache_lock = threading.Lock()

def _ensure_model_downloaded() -> str:
    """Download silero_vad.onnx into the local cache if not present.

    Returns the absolute path to the cached model file.
    """
    cached = _CACHE_DIR / "silero_vad.onnx"
    if cached.exists():
        return str(cached)
    os.makedirs(str(_CACHE_DIR), exist_ok=True)
    urllib.request.urlretrieve(_MODEL_DOWNLOAD_URL, str(cached))
    logger.info("Silero VAD model downloaded")
    return str(cached)


def pre_download_model() -> None:
    """Pre-download the Silero VAD ONNX model into the local cache.

    Mirrors the turn-detector plugin's `pre_download_model()`. Call this
    at module level (before `WorkerJob.start`) so spawned worker
    processes never pay the network download on first job.
    """
    _ensure_model_downloaded()


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
        """Create or reuse an optimised ONNX Runtime InferenceSession.

        Sessions are cached at module level keyed by (use_cpu_only,
        resolved_path), so repeated SileroVAD() constructions in the
        same process share a single underlying session.

        Resolution order: onnx_file_path -> cached download (~/.cache/videosdk/silero).
        """
        if onnx_file_path is not None:
            resolved_path = Path(onnx_file_path).resolve()
            if not resolved_path.exists():
                raise FileNotFoundError(f"Silero VAD model file not found: {resolved_path}")
            resolved_path_str = str(resolved_path)
        else:
            resolved_path_str = _ensure_model_downloaded()

        cache_key = (bool(use_cpu_only), resolved_path_str)

        cached_session = _session_cache.get(cache_key)
        if cached_session is not None:
            return cached_session

        with _session_cache_lock:
            cached_session = _session_cache.get(cache_key)
            if cached_session is not None:
                return cached_session

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

            session = onnxruntime.InferenceSession(
                resolved_path_str, sess_options=session_opts, providers=providers
            )
            _session_cache[cache_key] = session
            return session