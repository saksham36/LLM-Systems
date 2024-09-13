#!/bin/bash

# Safely execute this bash script
# e exit on first failure
# x all executed commands are printed to the terminal
# u unset variables are errors
# a export all variables to the environment
# E any trap on ERR is inherited by shell functions
# -o pipefail | produces a failure code if any stage fails
set -Eeuoxa pipefail

# Set the name for the Docker container
CONTAINER_NAME=test

LOCAL_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

SERVER_HOSTNAME="$(uname -n)"

INSTALL_ROOT="${INSTALL_ROOT:-/app}"

DOCKERFILE_NAME="docker-compose.yaml"

SERVER_HOSTNAME=${SERVER_HOSTNAME} docker compose -f $LOCAL_DIRECTORY/$DOCKERFILE_NAME up -d

# Wait a moment to ensure containers are up
sleep 2

# Get the name of the first running container (modify as needed)
CONTAINER_NAME=$(docker ps --format '{{.Names}}' | head -n 1)

# Exec into the container if found
if [ -n "$CONTAINER_NAME" ]; then
    echo "Executing into container: $CONTAINER_NAME"
    docker exec -it "$CONTAINER_NAME" /bin/bash
else
    echo "No running container found."
fi