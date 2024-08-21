# python/add_wrapper.py

import torch
import numpy as np
import ctypes
from utils import load_tiled_matmul_library, configure_tiled_matmul_function

def multiply_matrices(a, b, d):

    if a.device != b.device:
        raise ValueError("Tensors must be on the same device.")
    
    # Ensure the tensors are on the GPU
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("Tensors must be on a CUDA device (GPU).")

    # Check the matrix dimensions
    m, n = a.shape
    n_b, k = b.shape
    assert n == n_b, "Inner matrix dimensions must agree."

    lib = load_tiled_matmul_library()
    configure_tiled_matmul_function(lib)
    lib.tiled_matrix_multiply(
        ctypes.c_void_p(a.data_ptr()),
        ctypes.c_void_p(b.data_ptr()),
        ctypes.c_void_p(d.data_ptr()),
        ctypes.c_int(m),
        ctypes.c_int(n),
        ctypes.c_int(k)
    )

    return 

if __name__ == "__main__":
    m = 10
    k = 5
    n = 2

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    a = torch.arange(m*n, dtype=torch.float32, device=device).view(m,n).contiguous()
    b = torch.arange(n*k, dtype=torch.float32, device=device).view(n,k).contiguous()
    d = torch.empty((m,k), dtype=torch.float32, device=device)
    multiply_matrices(a, b, d)
    print("Result:", d)
