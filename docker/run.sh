#!/usr/bin/env bash
# Start, restart, or attach to the 26-S-Lockheed-1 dev container.
#
# Usage (from repo root or docker/):
#   bash docker/run.sh            — create/start/attach
#   bash docker/run.sh restart    — force-recreate container
#
# Volume mounts:
#   /workspace            ← repo root (live edits without rebuild)
#   /data                 ← "/media/priyanshu/2TB SSD" (datasets)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE_TAG="$(whoami)_lockheed1"
CONTAINER_NAME="lockheed1_$(whoami)"

# Allow callers to override the dataset root via env var.
# Example:  DATASET_PATH="/mnt/my_drive/data" bash docker/run.sh
DATASET_PATH="${DATASET_PATH:-}"

if [ -z "$DATASET_PATH" ]; then
    echo "ERROR: DATASET_PATH is not set."
    echo "  Export the path to your dataset directory before running, e.g.:"
    echo '    export DATASET_PATH="/path/to/your/data"'
    echo '    bash docker/run.sh'
    exit 1
fi

DOCKER_OPTIONS=(
    -it
    --gpus all
    --runtime=nvidia
    -e NVIDIA_DRIVER_CAPABILITIES=all
    -e DISPLAY="${DISPLAY:-}"
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    -v /tmp/.X11-unix:/tmp/.X11-unix
    -v "$HOME/.Xauthority:/home/$(whoami)/.Xauthority"
    -v "$REPO_ROOT:/workspace"
    --name "$CONTAINER_NAME"
    --net=host
    --shm-size=32G
    -u "$(id -u):$(id -g)"
)

# Mount the 2TB SSD dataset drive if it exists
if [ -d "$DATASET_PATH" ]; then
    DOCKER_OPTIONS+=(-v "$DATASET_PATH:/data")
    echo "Mounting dataset: $DATASET_PATH → /data"
else
    echo "WARNING: Dataset path not found: $DATASET_PATH"
    echo "         Start training scripts with /data will fail."
fi

if [ "${1:-}" == "restart" ]; then
    echo "Restarting container: $CONTAINER_NAME"
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    docker run "${DOCKER_OPTIONS[@]}" "$IMAGE_TAG:latest"

elif [ ! "$(docker ps -q -f name="$CONTAINER_NAME")" ]; then
    # Not running
    if [ "$(docker ps -aq -f name="$CONTAINER_NAME")" ]; then
        # Exists but stopped — restart it
        echo "Resuming stopped container: $CONTAINER_NAME"
        docker start "$CONTAINER_NAME"
        docker exec -it "$CONTAINER_NAME" /bin/bash
    else
        # Doesn't exist — create fresh
        echo "Creating container: $CONTAINER_NAME"
        docker run "${DOCKER_OPTIONS[@]}" "$IMAGE_TAG:latest"
    fi
else
    # Already running — attach
    echo "Attaching to running container: $CONTAINER_NAME"
    docker exec -it "$CONTAINER_NAME" /bin/bash
fi
