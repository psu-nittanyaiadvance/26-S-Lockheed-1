#!/usr/bin/env bash
# Build the unified 26-S-Lockheed-1 Docker image.
#
# Usage (from repo root or docker/):
#   bash docker/build.sh
#
# Passes host user UID/GID so files written inside the container are owned
# by you, not root.  GPU arch 8.6 = RTX 3080.  Change TORCH_CUDA_ARCH if
# your GPU differs (https://developer.nvidia.com/cuda-gpus).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE_TAG="$(whoami)_lockheed1"
TORCH_CUDA_ARCH="${TORCH_CUDA_ARCH:-8.6}"
MAX_JOBS="${MAX_JOBS:-4}"

echo "=== Building $IMAGE_TAG ==="
echo "    TORCH_CUDA_ARCH = $TORCH_CUDA_ARCH"
echo "    MAX_JOBS        = $MAX_JOBS"
echo "    Repo root       = $REPO_ROOT"
echo ""

docker build \
    -t "$IMAGE_TAG:latest" \
    -f "$REPO_ROOT/docker/Dockerfile" \
    --build-arg USER_NAME="$(whoami)" \
    --build-arg USER_ID="$(id -u)" \
    --build-arg GROUP_ID="$(id -g)" \
    --build-arg TORCH_CUDA_ARCH="$TORCH_CUDA_ARCH" \
    --build-arg MAX_JOBS="$MAX_JOBS" \
    "$REPO_ROOT"

echo ""
echo "=== Build complete: $IMAGE_TAG:latest ==="
echo "    Run with:  bash docker/run.sh"
