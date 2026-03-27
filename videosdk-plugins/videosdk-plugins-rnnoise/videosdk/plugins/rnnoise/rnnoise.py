import ctypes
import numpy
import os
import platform

script_dir = os.path.dirname(os.path.abspath(__file__))

sys_name = platform.system()
if sys_name == "Darwin":
    lib_name = "librnnoise.dylib"
elif sys_name == "Linux":
    lib_name = "librnnoise.so"
elif sys_name == "Windows":
    lib_name = "rnnoise.dll"
else:
    raise OSError(f"Unsupported OS: {sys_name}")

lib_path = os.path.join(script_dir, "files", lib_name)

try:
    lib = ctypes.cdll.LoadLibrary(lib_path)
except OSError as e:
    raise OSError(
        f"Error loading rnnoise library at {lib_path}. "
        f"It may be corrupted or incompatible with your platform. "
        f"Original error: {e}"
    ) from e

lib.rnnoise_create.argtypes = [ctypes.c_void_p]
lib.rnnoise_create.restype = ctypes.c_void_p
lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]

lib.rnnoise_process_frame.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
lib.rnnoise_process_frame.restype = ctypes.c_float

class RNN(object):
    def __init__(self):
        self.obj = lib.rnnoise_create(None)
        
        if not self.obj:
             raise RuntimeError("RNNoise library failed to create internal state (returned NULL).")

    def process_frame(self, inbuf):
        if len(inbuf) != 960:
            return (0.0, inbuf)

        indata = numpy.frombuffer(inbuf, dtype=numpy.int16).astype(numpy.float32)

        if not indata.flags['C_CONTIGUOUS']:
            indata = numpy.ascontiguousarray(indata)

        outdata = numpy.zeros(480, dtype=numpy.float32)

        in_ptr = indata.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        out_ptr = outdata.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        try:
            prob = lib.rnnoise_process_frame(self.obj, out_ptr, in_ptr)
        except Exception:
            return (0.0, inbuf)

        return (prob, outdata.astype(numpy.int16).tobytes())

    def destroy(self):
        if self.obj:
            lib.rnnoise_destroy(self.obj)
            self.obj = None