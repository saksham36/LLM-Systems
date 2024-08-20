#!/bin/bash

# Create directories if they don't exist
mkdir -p lib

# Compile the CUDA code into a shared library
echo "Compiling CUDA code..."
nvcc -Xcompiler -fPIC -shared -o lib/libadd.so src/add.cu

# Check if the compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful!"
else
    echo "Compilation failed!"
    exit 1
fi

# Run the Python script
echo "Running Python script..."
python3 python/add_wrapper.py
