# =============================================================================
# 26-S-Lockheed-1 — Unified Docker image
# =============================================================================
# Covers both sub-projects in this repo:
#   • SonarSplat   (sonar_splat/)                   — gsplat CUDA extension
#   • Z-Splat v2   (z_splatting/)  — diff-gaussian-rasterization
#                                                      + simple-knn CUDA extensions
#
# Build:  bash docker/build.sh              (from repo root)
# Run:    bash docker/run.sh                (from repo root)
# =============================================================================

# ---------- Base image -------------------------------------------------------
# Using 12.4.1-devel: includes nvcc + CUDA headers needed to compile extensions.
# cu124 PyTorch wheels are compatible with this runtime.
ARG CUDA_VERSION=12.4.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

# ---------- Build-time arguments ---------------------------------------------
# USER_NAME / USER_ID / GROUP_ID: match host user so mounted files are
# owned by you, not root.  build.sh passes $(id -u) automatically.
ARG USER_NAME=user
ARG USER_ID=1000
ARG GROUP_ID=1000
# GPU compute capability for CUDA extension compilation.
# RTX 3080 = 8.6.  Find yours: https://developer.nvidia.com/cuda-gpus
ARG TORCH_CUDA_ARCH="8.6"
# Limit parallel compile jobs to avoid OOM on machines with <16 GB RAM.
# Increase to 8+ if you have more RAM.
ARG MAX_JOBS=4

# ---------- Environment ------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# CUDA paths (nvcc must be on PATH for extension compilation)
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# nerfacc does JIT kernel compilation at first import — it needs CUDA headers.
# In Docker (devel image) they live here:
ENV CPLUS_INCLUDE_PATH=/usr/local/cuda/include:${CPLUS_INCLUDE_PATH}

# GPU arch list baked into every CUDA extension compiled below
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH}
# Controls parallel compile jobs for ninja (sonar_splat/setup.py respects this)
ENV MAX_JOBS=${MAX_JOBS}

# Required by sonar_simple_trainer.py — prevents CUDA allocator OOM on large scenes.
# Leave set; removing it can silently crash long training runs.
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------- System packages --------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Compilers and build tools
    build-essential \
    cmake \
    ninja-build \
    # Python 3.10 (matches the sonarsplat conda env)
    python3.10 \
    python3.10-dev \
    python3-pip \
    # Utilities
    git \
    wget \
    curl \
    vim \
    sudo \
    # OpenGL + graphics (Open3D headless rendering, libGL for OpenCV)
    libgl1-mesa-dev \
    libglib2.0-0 \
    libglew-dev \
    libglfw3-dev \
    libxxf86vm-dev \
    # Linear algebra
    libeigen3-dev \
    libboost-filesystem-dev \
    libboost-system-dev \
    # Image / video (imageio[ffmpeg])
    ffmpeg \
  && rm -rf /var/lib/apt/lists/*

# ---------- Python 3.10 as default -------------------------------------------
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    python3 -m pip install --upgrade "pip<25" setuptools wheel ninja

# ---------- PyTorch (cu124 — exact match to sonarsplat conda env) -----------
# cu124 wheels work against both CUDA 12.4.x and 12.6.x runtimes.
RUN python3 -m pip install \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# ---------- Pure Python dependencies -----------------------------------------
# Installed before CUDA extensions so they're available as build deps.
COPY docker/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# ---------- CUDA-linked dependencies (need --no-build-isolation) -------------
# These compile against the already-installed torch, so they can't use
# pip's default build isolation (which would pull a fresh CPU torch).

# nerfacc: used by SonarSplat for volumetric rendering; compiles CUDA kernels
# that are JIT-cached at first import (see CPLUS_INCLUDE_PATH above).
RUN python3 -m pip install --no-build-isolation --no-cache-dir \
    git+https://github.com/nerfstudio-project/nerfacc

# fused-ssim: fast CUDA SSIM implementation used by SonarSplat's loss
RUN python3 -m pip install --no-build-isolation --no-cache-dir \
    git+https://github.com/rahul-goel/fused-ssim@1272e21a282342e89537159e4bad508b19b34157

# nerfview: web-based NeRF/3DGS viewer (pure Python, but needs nerfacc installed first)
RUN python3 -m pip install --no-cache-dir \
    git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787

# ---------- Copy repository --------------------------------------------------
# .dockerignore at repo root filters out logs, outputs, __pycache__, .git, etc.
WORKDIR /workspace
COPY . .

# ---------- SonarSplat: gsplat CUDA extension --------------------------------
# Two-phase build (matches SETUP_LINUX.md exactly):
#   Phase 1: CPU-only install registers the Python API without CUDA.
#   Phase 2: Full reinstall compiles the CUDA kernels against phase-1 API.
# Skipping phase 1 can cause JIT registration errors on first import.
RUN cd /workspace/sonar_splat && \
    BUILD_NO_CUDA=1 python3 -m pip install --no-build-isolation -e . && \
    python3 -m pip install --no-build-isolation -e .

# ---------- Z-Splat: diff-gaussian-rasterization ----------------------------
# Note: rasterizer_impl.h already has `#include <cstdint>` applied (CUDA 12+ fix).
# The rasterizer returns only (color, radii) — z_density is computed by
# compute_z_density_diff() in train_v2.py, not here.
RUN cd /workspace/z_splatting/submodules/diff-gaussian-rasterization && \
    python3 -m pip install --no-build-isolation -e .

# ---------- Z-Splat: simple-knn ----------------------------------------------
RUN cd /workspace/z_splatting/submodules/simple-knn && \
    python3 -m pip install --no-build-isolation -e .

# ---------- Non-root user (matches host UID for volume file permissions) -----
# build.sh passes --build-arg USER_ID=$(id -u) so the container user's UID
# matches the host user — files created in mounted volumes are owned by you.
RUN groupadd -f -g ${GROUP_ID} ${USER_NAME} && \
    useradd -m -l -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USER_NAME} && \
    usermod -aG video ${USER_NAME} && \
    echo "${USER_NAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && \
    chown -R ${USER_NAME}:${GROUP_ID} /workspace

USER ${USER_NAME}
WORKDIR /workspace

ENTRYPOINT ["/bin/bash"]
