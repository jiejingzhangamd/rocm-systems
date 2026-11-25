"""Run under rocprofv3 after compiling register_lifetime_probe.hip as a DSO."""
import _ctypes
import ctypes
import sys

import torch

x = torch.ones(1, device="cuda")
lib = ctypes.CDLL(sys.argv[1], mode=ctypes.RTLD_LOCAL)
launch = lib.launch_register_lifetime_probe
launch.argtypes = [ctypes.c_void_p]
launch(ctypes.c_void_p(x.data_ptr()))
torch.cuda.synchronize()
assert x.item() == 2
del launch
_ctypes.dlclose(lib._handle)
print("DSO_UNLOADED_AND_GPU_ORACLE_PASSED", flush=True)
