// include/tiled_matmul.h

#ifndef TILED_MATMUL_H
#define TILED_MATMUL_H

#ifdef __cplusplus
extern "C"
{
#endif

    void tiled_matrix_multiply(float *A, float *B, float *D, int m, int n, int k);

#ifdef __cplusplus
}
#endif

#endif // TILED_MATMUL_H
