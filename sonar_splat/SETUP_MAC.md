# SonarSplat Mac Setup Guide

Mac is supported for training and inference using CPU or the MPS (Metal) backend on Apple Silicon. Training will be significantly slower than on an NVIDIA GPU — use `--max_steps 500` for quick tests.

## Prerequisites

- Conda (Miniconda or Anaconda)
- Mac OS 12 Monterey or later
- Apple Silicon (M1/M2/M3/M4) recommended for MPS acceleration; Intel Macs run CPU-only

## Step 1: Create Conda Environment

```bash
conda create -n sonarsplat python=3.10 -y
conda activate sonarsplat
```

## Step 2: Install PyTorch

```bash
pip install torch torchvision torchaudio
```

Apple Silicon will automatically use the MPS backend. Intel Macs will use CPU.

## Step 3: Install gsplat (CPU/MPS mode, no CUDA)

```bash
pip install ninja setuptools wheel
conda env config vars set BUILD_NO_CUDA=1
conda deactivate && conda activate sonarsplat
pip install --no-build-isolation -e .
conda env config vars unset BUILD_NO_CUDA
conda deactivate && conda activate sonarsplat
```

## Step 4: Install dependencies

```bash
pip install -r examples/requirements.txt
pip install torchmetrics
```

`fused_ssim` requires CUDA to build — the code will automatically fall back to `torchmetrics` on Mac.

## Step 5: Fix Code Issues

Comment out the unused pycolmap import in `sonar/dataset/dataloader.py`, line 11:
```python
# from pycolmap import SceneManager
```

## Step 6: Switch to the zplat branch

```bash
git checkout zplat
git submodule update --init --recursive
```

## Step 7: Run Training

The trainer will automatically detect and use MPS (Apple Silicon) or CPU:

```bash
python examples/sonar_simple_trainer.py zsplat \
    --data_dir /path/to/your/dataset \
    --result_dir /path/to/your/results \
    --train
```

For a quick test:
```bash
python examples/sonar_simple_trainer.py zsplat \
    --data_dir /path/to/your/dataset \
    --result_dir /path/to/your/results \
    --max_steps 500 \
    --init_num_pts 5000 \
    --train
```

Results (PSNR, SSIM, LPIPS) are saved to `result_dir/stats/`. Rendered test images are saved to `result_dir/test/sonar_images/`.

## Troubleshooting

- **Slow training**: Expected without a CUDA GPU. Use fewer steps and Gaussians for testing.
- **MPS fallback errors**: If an operation isn't supported on MPS, set `PYTORCH_ENABLE_MPS_FALLBACK=1` before running:
  ```bash
  PYTORCH_ENABLE_MPS_FALLBACK=1 python examples/sonar_simple_trainer.py zsplat ...
  ```
- **Out of memory on MPS**: Reduce `--init_num_pts` (e.g. `--init_num_pts 5000`)
