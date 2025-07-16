import atexit
import importlib.resources
from contextlib import ExitStack

import numpy as np
import onnxruntime  

_resource_files = ExitStack()
atexit.register(_resource_files.close)


SUPPORTED_SAMPLE_RATES = [8000, 16000]

def inference_session(force_cpu: bool) -> onnxruntime.InferenceSession:
    res = importlib.resources.files("videosdk.plugins.silero.model") / "silero_vad.onnx"
    ctx = importlib.resources.as_file(res)
    path = str(_resource_files.enter_context(ctx))

    opts = onnxruntime.SessionOptions()
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL

    if force_cpu and "CPUExecutionProvider" in onnxruntime.get_available_providers():
        session = onnxruntime.InferenceSession(
            path, providers=["CPUExecutionProvider"], sess_options=opts
        )
    else:
        session = onnxruntime.InferenceSession(path, sess_options=opts)

    return session

class SileroOnnx:
    def __init__(self, *, onnx_session: onnxruntime.InferenceSession, sample_rate: int) -> None:
        self._session = onnx_session
        self._sample_rate = sample_rate

        if sample_rate == 16000:
            self._window_size = 512
            self._context_size = 64
        elif sample_rate == 8000:
            self._window_size = 256
            self._context_size = 32
        else:
            raise ValueError("Supported sample rates are 8000 or 16000")

        self._recurrent_state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self._context_size), dtype=np.float32)
        self._input_buffer = np.zeros(
            (1, self._context_size + self._window_size), dtype=np.float32
        )

    @property
    def window_size_samples(self) -> int:
        return self._window_size
    
    @property
    def context_size(self) -> int:
        return self._context_size

    def __call__(self, audio_frame: np.ndarray) -> float:
        self._input_buffer[:, :self._context_size] = self._context
        self._input_buffer[:, self._context_size:] = audio_frame

        model_inputs = {
            "input": self._input_buffer,
            "state": self._recurrent_state,
            "sr": np.array(self._sample_rate, dtype=np.int64),
        }
        output_prob, _ = self._session.run(None, model_inputs)

        self._context = self._input_buffer[:, -self._context_size:]

        return output_prob.item()
