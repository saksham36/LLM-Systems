// gemm.cu

#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel function to multiply 2 matrices. D = A*B
#define TILE_WIDTH 32
#define TILE_HEIGHT 32
__global__ void tiled_matmul(float *A, float *B, float *D, const int m, const int n, const int k)
{
    __shared__ float A_ds[TILE_HEIGHT][TILE_WIDTH];
    __shared__ float B_ds[TILE_HEIGHT][TILE_WIDTH];

    // Set them to register for faster access
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_HEIGHT + ty;
    int col = bx * TILE_WIDTH + tx;

    float d = 0;
    for (int pass = 0; pass < pass < (k + TILE_WIDTH - 1) / TILE_WIDTH; pass++)
    {
        if ((row < m) && (pass * TILE_WIDTH + tx) < k)
            A_ds[ty][tx] = A[row * k + pass * TILE_WIDTH + tx];
        else
            A_ds[ty][tx] = 0.0f;
        if ((col < n) && (pass * TILE_HEIGHT + ty) < k)
            B_ds[ty][tx] = B[(pass * TILE_HEIGHT + ty) * TILE_WIDTH + col];
        else
            B_ds[ty][tx] = 0.0f;
        __syncthreads();

        for (int w = 0; w < TILE_WIDTH; w++)
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
    dim3 threads_per_block(TILE_HEIGHT, TILE_WIDTH);
    dim3 blocks_per_grid((k + TILE_HEIGHT - 1) / TILE_HEIGHT, (m + TILE_WIDTH - 1) / TILE_WIDTH);

    tiled_matmul<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_D, m, n, k);
    // Copy result array from device memory to host memory
    cudaMemcpy(D, d_D, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_D);
}
