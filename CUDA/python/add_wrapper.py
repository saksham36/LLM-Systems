# python/add_wrapper.py

import numpy as np
import ctypes
from utils import load_add_library, configure_add_function

def add_arrays(a, b):
    n = len(a)
    c = np.zeros(n, dtype=np.int32)

    a = (ctypes.c_int * n)(*a)
    b = (ctypes.c_int * n)(*b)
    c = (ctypes.c_int * n)(*c)

    lib = load_add_library()
    configure_add_function(lib)
    lib.add_arrays(a, b, c, n)

    return list(c)

if __name__ == "__main__":
    a = [1, 2, 3, 4]
    b = [5, 6, 7, 8]

    result = add_arrays(a, b)
    print("Result:", result)
