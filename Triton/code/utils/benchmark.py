import os
import torch
import triton
from pathlib import Path

def run_benchmark(triton_func, torch_func, torch_jit_func, torch_compile_func, plot_name="softmax-performance", save_path='code/benchmark'):
    save_path = os.path.join(save_path,plot_name)
    try:
        os.makedirs(save_path)
    except Exception as e:
        print(f"New folder not created. {e}")
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['N'],  # argument names to use as an x-axis for the plot
            x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name`
            line_arg='provider',  # argument name whose value corresponds to a different line in the plot
            line_vals=['triton', 'torch-native', 'torch-jit', 'torch-compile'],  # possible values for `line_arg`
            line_names=["Triton", "Torch (native)", "Torch (jit)", "Torch (compile)"],  # label name for the lines
            styles=[('blue', '-'), ('green', '-'), ('red', '--'), ('gold', '--')],  # line styles
            ylabel="GB/s",  # label name for the y-axis
            plot_name=plot_name,  # dynamic plot name
            args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
        )
    )

    def benchmark(M, N, provider):
        x = torch.randn(M, N, device='cuda', dtype=torch.float32)
        quantiles = [0.5, 0.2, 0.8]
        
        if provider == 'torch-native':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_func(x , axis=-1), quantiles=quantiles)
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_func(x), quantiles=quantiles)
        if provider == 'torch-jit':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_jit_func(x), quantiles=quantiles)
        
        if provider == 'torch-compile':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_compile_func(x), quantiles=quantiles)
        
        gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
        return gbps(ms), gbps(max_ms), gbps(min_ms)
    benchmark.run(show_plots=True, print_data=True, save_path=save_path)

