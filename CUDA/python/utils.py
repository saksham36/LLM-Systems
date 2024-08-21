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

def load_tiled_matmul_library():
    lib_path = os.path.join(os.path.dirname(__file__), '../lib/libtiled_matmul.so')
    return ctypes.CDLL(lib_path)

def configure_tiled_matmul_function(lib):
    lib.tiled_matrix_multiply.argtypes = [
        ctypes.c_void_p,  # A
        ctypes.c_void_p,  # B
        ctypes.c_void_p,  # D
        ctypes.c_int,  # m
        ctypes.c_int,  # n
        ctypes.c_int,  # k
    ]
    lib.tiled_matrix_multiply.restype = None  # void
