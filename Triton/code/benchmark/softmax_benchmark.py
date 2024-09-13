import torch
from utils.benchmark import run_benchmark
from functions.softmax import naive_softmax,triton_softmax

triton_func = triton_softmax  
torch_func = torch.softmax
torch_jit_func = naive_softmax

# Run the benchmark with the functions and custom plot name
run_benchmark(triton_func=triton_func, torch_func=torch_func, torch_jit_func=naive_softmax, plot_name="benchmark-softmax")