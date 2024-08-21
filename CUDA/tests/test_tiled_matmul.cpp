// tests/test_tiled_matmul.cpp

#include <iostream>
#include <vector>
#include <cassert>
#include <iomanip>

#include "../include/matrix_util.h"
#include "../include/tiled_matmul.h"

void test_matmul()
{
    int m = 4, n = 8, k = 2; // Matrix dimensions
    // Allocate memory for matrices
    std::vector<float> A(m * k);
    std::vector<float> B(k * n);
    std::vector<float> D(m * n, 0); // Result matrix initialized to 0
    std::vector<float> expected_D = {
        5, 10, 15, 20, 25, 30, 35, 40,
        10, 20, 30, 40, 50, 60, 70, 80,
        15, 30, 45, 60, 75, 90, 105, 120,
        20, 40, 60, 80, 100, 120, 140, 160};
    ; // Expected result matrix

    initialize_matrix(A.data(), m, k);
    initialize_matrix(B.data(), k, n);

    tiled_matrix_multiply(A.data(), B.data(), D.data(), m, n, k);

    assert(matrices_are_approx_equal(D.data(), expected_D.data(), m, n) && "Matrix multiplication result does not match the expected output");

    std::cout << "test_add passed!" << std::endl;
}

int main()
{
    test_matmul();
    return 0;
}
