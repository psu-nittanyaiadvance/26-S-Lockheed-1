# SonarSplat Windows Setup Guide

System tested on: Windows 11 Pro, RTX 3080 (12GB), NVIDIA Driver 591.59, VS 2019 Build Tools.

## Prerequisites

- **Conda** (Miniconda or Anaconda)
- **Visual Studio 2019 Build Tools** with C++ workload (specifically `vcvars64.bat` at `C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat`)
- **NVIDIA GPU** with CUDA support (RTX 3080 = compute capability 8.6)

## Step 1: Create Conda Environment

```bash
conda create -n sonarsplat python=3.10 -y
conda activate sonarsplat
```

## Step 2: Install PyTorch with CUDA 12.4

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

or 

pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

## Step 3: Install CUDA nvcc (must match PyTorch's CUDA 12.4)

```bash
conda install -c "nvidia/label/cuda-12.4.1" cuda-nvcc -y
```

**Important:** Do NOT install `cuda-toolkit` from the default nvidia channel -- it may install CUDA 13.x which mismatches PyTorch's CUDA 12.4.

## Step 4: Install gsplat (without CUDA first)

First install build dependencies and gsplat in CPU-only mode:

```bash
pip install ninja setuptools wheel
```

Set env variable to skip CUDA compilation initially:
```bash
conda env config vars set BUILD_NO_CUDA=1
conda deactivate && conda activate sonarsplat
```

Install gsplat in editable mode:
```bash
pip install --no-build-isolation -e .
```

Then unset the flag:
```bash
conda env config vars unset BUILD_NO_CUDA
conda deactivate && conda activate sonarsplat
```

## Step 5: Build gsplat with CUDA Extensions

CUDA compilation on Windows requires VS2019 developer environment. Run `build_gsplat.bat` from a standard Command Prompt (NOT from conda/bash):

```
build_gsplat.bat
```

This batch file:
1. Activates the conda env
2. Sets up VS2019 developer tools (vcvars64.bat)
3. Sets CUDA_HOME to the conda env root
4. Sets TORCH_CUDA_ARCH_LIST=8.6 (adjust for your GPU)
5. Runs `pip install --no-build-isolation -e .`

**Critical:** `CUDA_HOME` must point to the conda environment root (e.g., `C:\Users\omviz\.conda\envs\sonarsplat`), NOT `Library/`. Conda places CUDA headers in `<env>/include/` and binaries in `<env>/bin/`.

## Step 6: Build fused-ssim

Clone fused-ssim source:
```bash
cd D:\Lockheed2026\sonar_splat
git clone https://github.com/jparismorgan/fused-ssim.git fused-ssim-src
```

Run `build_fused_ssim.bat` from a standard Command Prompt:
```
build_fused_ssim.bat
```

## Step 7: Install Remaining Dependencies

```bash
conda activate sonarsplat
pip install -r examples/requirements.txt
pip install nerfacc
```

Note: `fused_ssim` from examples/requirements.txt may fail via pip (needs CUDA build) -- that's fine since we already built it in Step 6.

## Step 8: Fix Code Issues

**Comment out unused pycolmap import** in `sonar/dataset/dataloader.py`, line 11:
```python
# from pycolmap import SceneManager  # unused import
```

> **Note:** If you are on the `zplat` branch, the `fused_ssim` and `nerfacc` import fixes are already committed — no manual patching needed.

## Step 9: Initialize GLM Submodule

```bash
cd D:\Lockheed2026\sonar_splat
git submodule update --init --recursive
```

This pulls the GLM math library needed by CUDA kernels into `gsplat/cuda/csrc/third_party/glm/`.

## Step 10: Switch to the zplat branch

```bash
git checkout zplat
```

## Step 11: Run Training

Run from a standard Command Prompt (so the `.bat` environment is active):

```
run_training.bat
```

Or manually:
```bash
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python examples/sonar_simple_trainer.py zsplat ^
    --data_dir C:\path\to\your\dataset ^
    --result_dir C:\path\to\your\results ^
    --train
```

For a quick 2000-step sanity check:
```bash
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python examples/sonar_simple_trainer.py zsplat ^
    --data_dir C:\path\to\your\dataset ^
    --result_dir C:\path\to\your\results ^
    --max_steps 2000 ^
    --train
```

Results (PSNR, SSIM, LPIPS) are saved to `result_dir\stats\`. Rendered test images are saved to `result_dir\test\sonar_images\`.

## Batch Files Reference

Three `.bat` files were created in the project root for Windows CUDA compilation:

| File | Purpose |
|------|---------|
| `build_gsplat.bat` | Build gsplat with CUDA extensions |
| `build_fused_ssim.bat` | Build fused-ssim CUDA extension |
| `run_training.bat` | Run training with VS2019/CUDA environment |

All three share the same environment setup pattern:
```bat
@echo off
call conda activate sonarsplat
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set DISTUTILS_USE_SDK=1
set MSSdk=1
set CUDA_HOME=C:\Users\omviz\.conda\envs\sonarsplat
set TORCH_CUDA_ARCH_LIST=8.6
set PATH=C:\Users\omviz\.conda\envs\sonarsplat\bin;%PATH%
```

## Troubleshooting

- **"CUDA_HOME not set"**: Make sure CUDA_HOME points to the conda env root, not Library/
- **CUDA version mismatch**: nvcc version must match PyTorch's CUDA version (12.4)
- **cl.exe not found**: Must run builds from .bat files that call vcvars64.bat first
- **`[WinError 2]` during build**: Usually means CUDA_HOME is wrong or nvcc isn't on PATH
- **`skip_frames` division by zero**: Use `--skip_frames 1` not `--skip_frames 0`
