#!/bin/bash

# Define the path to the code folder
code_folder="code"
benchmark_folder="$code_folder/benchmark"

# Check if the benchmark folder exists
if [ ! -d "$benchmark_folder" ]; then
    echo "Folder $benchmark_folder not found."
    exit 1
fi

# Set PYTHONPATH to include the code folder
export PYTHONPATH=$(pwd)/$code_folder

# Loop through all Python files in the benchmark folder and execute them
for script in "$benchmark_folder"/*.py; do
    echo "Running $script..."
    python3 "$script"
    if [ $? -ne 0 ]; then
        echo "Error: $script encountered an issue."
        exit 1
    fi
    echo "$script completed."
done

echo "All scripts executed."
