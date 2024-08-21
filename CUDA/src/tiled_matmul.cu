// gemm.cu

#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel function to multiply 2 matrices. D = A*B
__global__ void tiled_matmul(float *A, float *B, float *D, int m, int n, int k, int tile_width, int tile_height)
{
    __shared__ float A_ds[tile_height][tile_width];
    __shared__ float B_ds[tile_height][tile_width];

    // Set them to register for faster access
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * tile_height + ty;
    int col = bx * tile_width + tx;

    float d = 0;
    for (int pass = 0; pass < pass < (k + tile_width - 1) / tile_width; pass++)
    {
        if ((row < m) && (pass * tile_width + tx) < k)
            A_ds[ty][tx] = A[row * k + pass * tile_width + tx];
        else
            A_ds[ty][tx] = 0.0f;
        if ((col < n) && (pass * tile_height + ty) < k)
            B_ds[ty][tx] = B[(pass * tile_height + ty) * tile_width + col];
        else
            B_ds[ty][tx] = 0.0f;
        __syncthreads();

        for (int w = 0; w < tile_width; w++)
        {
            d += A_ds[ty][w] * B_ds[w][tx];
        }
        __syncthreads();
    }
    if (row < m && col < n)
        D[row * n + col] = d;
}

extern "C" void tiled_matrix_multiply(float *A, float *B, float *D, int m, int n, int k)
{
    float *d_A, *d_B, *d_D;
    size_t size = sizeof(float);

    // Allocate memory on the device (GPU)
    cudaMalloc((void **)&d_A, m * n * size);
    cudaMalloc((void **)&d_B, n * k * size);
    cudaMalloc((void **)&d_D, m * k * size);

    // Copy input arrays from host memory to device memory
    cudaMemcpy(d_A, A, m * n * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * k * size, cudaMemcpyHostToDevice);

    // Launch add() kernel on the GPU with one thread for each element
    int BLOCK_SIZE = 32;
    dim3 threads_per_block = (BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks_per_grid((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

    tiled_gemm<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, m, n, k, BLOCK_SIZE, BLOCK_SIZE);

    // Copy result array from device memory to host memory
    cudaMemcpy(d, d_D, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_D);
}
