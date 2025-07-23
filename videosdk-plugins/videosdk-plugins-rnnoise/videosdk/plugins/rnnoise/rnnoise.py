import ctypes,numpy,os
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

lib.rnnoise_process_frame.argtypes = [ctypes.c_void_p,ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float)]
lib.rnnoise_process_frame.restype = ctypes.c_float
lib.rnnoise_create.restype = ctypes.c_void_p
lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]

class RNN(object):
	def __init__(self):
		self.obj = lib.rnnoise_create()
	def process_frame(self,inbuf):
		outbuf = numpy.ndarray((480,), 'h', inbuf).astype(ctypes.c_float)
		outbuf_ptr = outbuf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
		VodProb =  lib.rnnoise_process_frame(self.obj,outbuf_ptr,outbuf_ptr)
		return (VodProb,outbuf.astype(ctypes.c_short).tobytes())

	def destroy(self):
		lib.rnnoise_destroy(self.obj)