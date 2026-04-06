import importlib.resources
import atexit
import os
import logging
import urllib.request
import numpy as np
import onnxruntime
from contextlib import ExitStack
from pathlib import Path

_resource_files = ExitStack()
atexit.register(_resource_files.close)

logger = logging.getLogger(__name__)

_MODEL_DOWNLOAD_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
)
_CACHE_DIR = Path.home() / ".cache" / "videosdk" / "silero"

SAMPLE_RATES = [8000, 16000]


class VadModelWrapper:
    def __init__(self, *, session: onnxruntime.InferenceSession, rate: int) -> None:
        if rate not in SAMPLE_RATES:
            raise ValueError(f"Rate {rate} not supported; use 8000 or 16000")
        
        self._model_session = session
        self._audio_rate = rate
        self._frame_size = 256 if rate == 8000 else 512
        self._history_len = 32 if rate == 8000 else 64
        
        self._hidden_state = np.zeros((2, 1, 128), dtype=np.float32)
        self._prev_context = np.zeros((1, self._history_len), dtype=np.float32)

    def reset_state(self) -> None:
        """Reset hidden state and context to initial values.

        Call when audio continuity is broken (e.g., after buffer flush)
        to prevent the model from processing discontinuous audio.
        """
        self._hidden_state = np.zeros((2, 1, 128), dtype=np.float32)
        self._prev_context = np.zeros((1, self._history_len), dtype=np.float32)

    @property
    def frame_size(self) -> int:
        return self._frame_size
    
    @property
    def history_len(self) -> int:
        return self._history_len

    def process(self, input_audio: np.ndarray) -> float:
        if input_audio.ndim == 1:
            input_audio = input_audio.reshape(1, -1)
        if input_audio.ndim > 2:
            raise ValueError(f"Too many dimensions for input audio chunk {input_audio.ndim}")
        
        if self._audio_rate / input_audio.shape[1] > 31.25:
            raise ValueError("Input audio chunk is too short")
        
        num_samples = 512 if self._audio_rate == 16000 else 256
        if input_audio.shape[-1] != num_samples:
            raise ValueError(f"Provided number of samples is {input_audio.shape[-1]} (Supported values: 256 for 8000 sample rate, 512 for 16000)")
        
        buffer = np.concatenate((self._prev_context, input_audio.reshape(1, -1)), axis=1)
        
        inputs = {
            "input": buffer.astype(np.float32),
            "state": self._hidden_state,
            "sr": np.array([self._audio_rate], dtype=np.int64),
        }
        
        prob, state = self._model_session.run(None, inputs)
        
        self._hidden_state = state
        self._prev_context = buffer[:, -self._history_len:]
        
        return prob.item()
    
    @staticmethod
    def create_inference_session(use_cpu_only: bool) -> onnxruntime.InferenceSession:
        session_opts = onnxruntime.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1
        session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL

        providers = (
            ["CPUExecutionProvider"]
            if use_cpu_only and "CPUExecutionProvider" in onnxruntime.get_available_providers()
            else None
        )

        try:
            model_path = importlib.resources.files("videosdk.plugins.silero.model") / "silero_vad.onnx"
            resolved = _resource_files.enter_context(importlib.resources.as_file(model_path))
            return onnxruntime.InferenceSession(str(resolved), sess_options=session_opts, providers=providers)
        except Exception as e:
            logger.warning(f"Bundled model failed: {e}, downloading...")

        cached = str(_CACHE_DIR / "silero_vad.onnx")
        if not os.path.exists(cached):
            os.makedirs(str(_CACHE_DIR), exist_ok=True)
            urllib.request.urlretrieve(_MODEL_DOWNLOAD_URL, cached)
            logger.info("Silero VAD model downloaded")

        return onnxruntime.InferenceSession(cached, sess_options=session_opts, providers=providers)
