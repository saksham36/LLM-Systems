def run_benchmark(triton_func, torch_func, torch_jit_func, plot_name="softmax-performance"):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['N'],  # argument names to use as an x-axis for the plot
            x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name`
            line_arg='provider',  # argument name whose value corresponds to a different line in the plot
            line_vals=['triton', 'torch-native', 'torch-jit'],  # possible values for `line_arg`
            line_names=["Triton", "Torch (native)", "Torch (jit)"],  # label name for the lines
            styles=[('blue', '-'), ('green', '-'), ('green', '--')],  # line styles
            ylabel="GB/s",  # label name for the y-axis
            plot_name=plot_name,  # dynamic plot name
            args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
        )
    )
    def benchmark(M, N, provider):
        x = torch.randn(M, N, device='cuda', dtype=torch.float32)
        quantiles = [0.5, 0.2, 0.8]
        
        if provider == 'torch-native':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_func(x), quantiles=quantiles)
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_func(x), quantiles=quantiles)
        if provider == 'torch-jit':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_jit_func(x), quantiles=quantiles)
        
        gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
        return gbps(ms), gbps(max_ms), gbps(min_ms)

