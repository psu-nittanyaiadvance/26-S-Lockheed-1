#!/usr/bin/env bash
# Convenience script to run z_splatting interactively via Docker
#
# Usage:
#   DATASET_PATH="/path/to/data" bash docker_run.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -z "${DATASET_PATH:-}" ]; then
    read -p "Please enter the path to the dataset you wish to mount: " DATASET_PATH
    if [ -z "$DATASET_PATH" ]; then
        echo "ERROR: Path cannot be empty."
        exit 1
    fi
fi

echo "Ensuring the base Docker container is running..."
export DATASET_PATH
cd "$REPO_ROOT"
# The docker/run.sh will start the container and attach, but if we want to just run our script:
# We will invoke run.sh to make sure it's up, then exec run.py.

CONTAINER_NAME="lockheed1_$(whoami)"

# Start container if not running without attaching to its bash
if ! docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
    # We call run.sh but run.sh automatically attaches. We'll just run it in the background briefly.
    echo "Container not running. Please start the container first via: bash docker/run.sh from the repo root."
    echo "After it's running, open a new terminal and run: docker exec -it $CONTAINER_NAME python3 /workspace/z_splatting/run.py"
    exit 1
fi

echo "Running Z-Splat interactive wrapper inside Docker container $CONTAINER_NAME..."
docker exec -it "$CONTAINER_NAME" python3 /workspace/z_splatting/run.py
