Here's the Linux version:

---

# SonarSplat Linux Setup Guide

## Prerequisites

- Conda (Miniconda or Anaconda)
- GCC (7–12 recommended)
- NVIDIA GPU with CUDA support

## Step 1: Create Conda Environment

```bash
conda create -n sonarsplat python=3.10 -y
conda activate sonarsplat
```

## Step 2: Install PyTorch with CUDA 12.4

```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

## Step 3: Install CUDA nvcc

```bash
conda install -c "nvidia/label/cuda-12.4.1" cuda-nvcc -y
```

**Important:** Do NOT install `cuda-toolkit` from the default nvidia channel — it may pull CUDA 13.x which mismatches PyTorch's CUDA 12.4.

## Step 4: Install gsplat (CPU-only first)

```bash
pip install ninja setuptools wheel
```

```bash
conda env config vars set BUILD_NO_CUDA=1
conda deactivate && conda activate sonarsplat
pip install --no-build-isolation -e .
conda env config vars unset BUILD_NO_CUDA
conda deactivate && conda activate sonarsplat
```

## Step 5: Build gsplat with CUDA Extensions

Set the required environment variables and build:

```bash
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="8.6"  # adjust for your GPU
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

pip install --no-build-isolation -e .
```

Find your GPU's compute capability at [developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus) if you're unsure what to set for `TORCH_CUDA_ARCH_LIST`.

## Step 6: Build fused-ssim

```bash
cd /path/to/sonar_splat
git clone https://github.com/jparismorgan/fused-ssim.git fused-ssim-src

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

cd fused-ssim-src && pip install --no-build-isolation -e . && cd ..
```

## Step 7: Install Remaining Dependencies

```bash
pip install -r examples/requirements.txt
pip install nerfacc
```

`fused_ssim` from requirements.txt may fail via pip — that's fine since you already built it above.

## Step 8: Fix Code Issues

**Comment out unused pycolmap import** in `sonar/dataset/dataloader.py`, line 11:
```python
# from pycolmap import SceneManager
```

**Fix opacity_pred_for_loss_img bug** in `examples/sonar_simple_trainer.py` around line 817: move `opacity_pred_for_loss_img` and `opacity_gt_img` assignments before the `if step >= cfg.opacity_supervision_start_step` block.

## Step 9: Initialize GLM Submodule

```bash
git submodule update --init --recursive
```

## Step 10: Run Training

```bash
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

python examples/sonar_simple_trainer.py prune_only \
    --batch_size 1 \
    --camera_model ortho \
    --data_dir data/sonarsplat_dataset/concrete_piling_3D \
    --result_dir results/test_run \
    --data_factor 1 \
    --disable_viewer \
    --init_type predefined \
    --init_num_pts 100000 \
    --init_opa 0.9 \
    --init_scale 0.01 \
    --max_steps 100 \
    --near_plane -10 \
    --far_plane 10 \
    --test_every 8 \
    --train \
    --render_eval \
    --sh_degree 3 \
    --tb_every 50 \
    --tb_save_image \
    --skip_frames 1 \
    --start_from_frame 0 \
    --end_at_frame 10000
```

**Use `--skip_frames 1`, not 0** — 0 causes a division-by-zero error.

## Optional: Persist Environment Variables

To avoid setting exports every session, add them to your conda env activation:

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export CUDA_HOME=$CONDA_PREFIX' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

These fire automatically whenever you `conda activate sonarsplat`.

## Troubleshooting

- **"CUDA_HOME not set"**: Run `export CUDA_HOME=$CONDA_PREFIX`
- **nvcc not found**: Make sure `$CUDA_HOME/bin` is on your PATH
- **CUDA version mismatch**: `nvcc --version` should report 12.4
- **GCC too new**: CUDA 12.4 tops out at GCC 12. If you're on GCC 13+, run `conda install -c conda-forge gcc=12 gxx=12`
- **`skip_frames` division by zero**: Use `--skip_frames 1`