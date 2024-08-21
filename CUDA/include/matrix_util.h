#ifndef MATRIX_UTIL_H
#define MATRIX_UTIL_H

#include <cmath> // For std::fabs

// Function to initialize a matrix with values
void initialize_matrix(float *matrix, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            matrix[i * cols + j] = static_cast<float>((i + 1) * (j + 1));
        }
    }
}

// Function to check if two matrices are approximately equal
bool matrices_are_approx_equal(const float *mat1, const float *mat2, int rows, int cols, float tol = 1e-5)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            if (std::fabs(mat1[i * cols + j] - mat2[i * cols + j]) > tol)
            {
                return false; // The matrices are not approximately equal
            }
        }
    }
    return true; // The matrices are approximately equal
}

#endif // MATRIX_UTIL_H
