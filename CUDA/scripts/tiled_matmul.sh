#!/bin/bash


# Compile the CUDA code into a shared library
echo "Compiling CUDA code..."
nvcc -Xcompiler -fPIC -shared -o lib/libtiled_matmul.so src/tiled_matmul.cu

# Check if the compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful!"
else
    echo "Compilation failed!"
    exit 1
fi

# Step 3: Update LD_LIBRARY_PATH to include the 'lib' directory
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/lib

# Step 4: Compile and run the C++ unit test
echo "Compiling and running C++ unit tests..."
g++ -o tests/test_tiled_matmul tests/test_tiled_matmul.cpp -L./lib -ltiled_matmul -I./include -lcuda

if [ $? -eq 0 ]; then
    ./tests/test_add
    if [ $? -eq 0 ]; then
        echo "C++ unit tests passed!"
    else
        echo "C++ unit tests failed!"
        exit 1
    fi
else
    echo "C++ unit test compilation failed!"
    exit 1
fi

# Run the Python script
# echo "Running Python script..."
# python3 python/tiled_matmul_wrapper.py
