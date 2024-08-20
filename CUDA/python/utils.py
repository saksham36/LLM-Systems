# python/utils.py

import ctypes
import os

# Load the shared library from the lib directory
def load_add_library():
    lib_path = os.path.join(os.path.dirname(__file__), '../lib/libadd.so')
    return ctypes.CDLL(lib_path)

# Configure argument and return types for the add_arrays function
def configure_add_function(lib):
    lib.add_arrays.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int
    ]
    lib.add_arrays.restype = None
