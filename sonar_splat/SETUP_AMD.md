# SonarSplat AMD GPU Setup Guide (ROCm)

AMD GPUs are supported on Linux via ROCm. PyTorch's ROCm build mirrors the CUDA API, so `torch.cuda.*` calls work as-is. **AMD ROCm is Linux-only** — no Windows or Mac support.

Tested with: ROCm 6.x, RX 7900 XTX / MI series GPUs.

## Prerequisites

- Linux (Ubuntu 22.04 or later recommended)
- AMD GPU with ROCm support (check compatibility: [rocm.docs.amd.com](https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html))
- Conda (Miniconda or Anaconda)
- ROCm installed system-wide (see below)

## Step 1: Install ROCm (system-level)

```bash
sudo apt update
wget https://repo.radeon.com/amdgpu-install/6.1.3/ubuntu/jammy/amdgpu-install_6.1.60103-1_all.deb
sudo apt install ./amdgpu-install_6.1.60103-1_all.deb
sudo amdgpu-install --usecase=rocm
sudo usermod -aG render,video $USER
reboot
```

Verify after reboot:
```bash
rocminfo | grep "gfx"
```

## Step 2: Create Conda Environment

```bash
conda create -n sonarsplat python=3.10 -y
conda activate sonarsplat
```

## Step 3: Install PyTorch with ROCm

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1
```

Verify:
```bash
python -c "import torch; print(torch.cuda.is_available())"  # should print True
```

> PyTorch ROCm maps all `torch.cuda.*` calls to ROCm/HIP — no code changes needed.

## Step 4: Install gsplat with ROCm

```bash
pip install ninja setuptools wheel
export ROCM_HOME=/opt/rocm
export CUDA_HOME=$ROCM_HOME
export PATH=$ROCM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_HOME/lib:$LD_LIBRARY_PATH
# Set your GPU architecture (run `rocminfo | grep gfx` to find yours)
export PYTORCH_ROCM_ARCH="gfx1100"  # RX 7900 XTX; use gfx90a for MI250, gfx942 for MI300
pip install --no-build-isolation -e .
```

## Step 5: Build fused-ssim

```bash
export ROCM_HOME=/opt/rocm
export CUDA_HOME=$ROCM_HOME
export PATH=$ROCM_HOME/bin:$PATH
cd fused-ssim-src && pip install --no-build-isolation -e . && cd ..
```

If fused-ssim fails to build, that's fine — the code falls back to `torchmetrics` automatically.

## Step 6: Install Remaining Dependencies

```bash
pip install -r examples/requirements.txt
pip install torchmetrics
```

## Step 7: Fix Code Issues

Comment out the unused pycolmap import in `sonar/dataset/dataloader.py`, line 11:
```python
# from pycolmap import SceneManager
```

## Step 8: Initialize GLM Submodule

```bash
git submodule update --init --recursive
```

## Step 9: Switch to the zplat branch

```bash
git checkout zplat
```

## Step 10: Run Training

```bash
export ROCM_HOME=/opt/rocm
export CUDA_HOME=$ROCM_HOME
export PATH=$ROCM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_HOME/lib:$LD_LIBRARY_PATH

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python examples/sonar_simple_trainer.py zsplat \
    --data_dir /path/to/your/dataset \
    --result_dir /path/to/your/results \
    --train
```

For a quick 2000-step sanity check:
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python examples/sonar_simple_trainer.py zsplat \
    --data_dir /path/to/your/dataset \
    --result_dir /path/to/your/results \
    --max_steps 2000 \
    --train
```

## Persist Environment Variables

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh << 'EOF'
export ROCM_HOME=/opt/rocm
export CUDA_HOME=$ROCM_HOME
export PATH=$ROCM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_HOME/lib:$LD_LIBRARY_PATH
EOF
```

## Troubleshooting

- **`torch.cuda.is_available()` returns False**: ROCm not installed or GPU not in `render` group — check `rocminfo`
- **Kernel compile error**: Make sure `PYTORCH_ROCM_ARCH` matches your GPU (`rocminfo | grep gfx`)
- **`hipcc` not found**: Add `/opt/rocm/bin` to PATH
- **Out of memory**: AMD cards report memory differently — try reducing `--init_num_pts` to 20000
