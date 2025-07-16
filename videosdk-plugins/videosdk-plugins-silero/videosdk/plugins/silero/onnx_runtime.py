import atexit
import importlib.resources
from contextlib import ExitStack

import numpy as np
import onnxruntime  

_resource_files = ExitStack()
atexit.register(_resource_files.close)

SUPPORTED_SAMPLE_RATES = [8000, 16000]

def create_onnx_session(use_cpu_only: bool) -> onnxruntime.InferenceSession:
    model_path = importlib.resources.files("videosdk.plugins.silero.model") / "silero_vad.onnx"
    with importlib.resources.as_file(model_path) as temp_path:
        session_opts = onnxruntime.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1
        session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        
        providers = ["CPUExecutionProvider"] if use_cpu_only and "CPUExecutionProvider" in onnxruntime.get_available_providers() else None
        
        return onnxruntime.InferenceSession(
            str(temp_path),
            sess_options=session_opts,
            providers=providers
        )

class VadModelWrapper:
    def __init__(self, *, session: onnxruntime.InferenceSession, rate: int) -> None:
        if rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(f"Rate {rate} not supported; use 8000 or 16000")
        
        self._model_session = session
        self._audio_rate = rate
        self._frame_size = 256 if rate == 8000 else 512
        self._history_len = 32 if rate == 8000 else 64
        
        self._hidden_state = np.zeros((2, 1, 128), dtype=np.float32)
        self._prev_context = np.zeros((1, self._history_len), dtype=np.float32)
        
    @property
    def frame_size(self) -> int:
        return self._frame_size
    
    @property
    def history_len(self) -> int:
        return self._history_len

    def process(self, input_audio: np.ndarray) -> float:
        buffer = np.concatenate((self._prev_context, input_audio.reshape(1, -1)), axis=1)
        
        inputs = {
            "input": buffer.astype(np.float32),
            "state": self._hidden_state,
            "sr": np.array([self._audio_rate], dtype=np.int64),
        }
        
        prob, _ = self._model_session.run(None, inputs)
        
        self._prev_context = buffer[:, -self._history_len:]
        
        return prob.item()
