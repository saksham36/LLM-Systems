import torch
from utils.benchmark import run_benchmark
from functions.softmax import naive_softmax,triton_softmax

triton_func = triton_softmax  
torch_func = torch.softmax
torch_jit_func = naive_softmax
torch_compile_func = torch.compile(torch_jit_func)

# Run the benchmark with the functions and custom plot name
run_benchmark(triton_func=triton_func, torch_func=torch_func, torch_jit_func=torch_jit_func, torch_compile_func=torch_compile_func, plot_name="benchmark-softmax")