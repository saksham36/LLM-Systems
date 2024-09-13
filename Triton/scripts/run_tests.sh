#!/bin/bash

# Find all Python test files in the test folder and run them
python3 -m unittest discover -s tests -p "test_*.py" -v