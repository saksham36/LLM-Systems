# Extend the base image
# FROM rocm/pytorch:rocm6.1_ubuntu22.04_py3.10_pytorch_2.4
FROM nvcr.io/nvidia/pytorch:24.07-py3
# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set environment variables if needed
ENV INSTALL_ROOT=${INSTALL_ROOT}

# Keep the entrypoint as in docker-compose.yaml
ENTRYPOINT ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]
