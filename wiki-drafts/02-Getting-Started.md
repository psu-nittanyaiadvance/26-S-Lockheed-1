# Getting Started

This project has three separate environments because the two reconstruction frameworks (SonarSplat and WaterSplatting) have incompatible Python and CUDA version requirements. The data analysis and preprocessing pipeline is independent of both.

## System Requirements Overview

|  | Data / Preprocessing Pipeline | SonarSplat | WaterSplatting |
|--|-------------------------------|------------|----------------|
| **Python** | 3.9+ | 3.10 (strict) | 3.8 (strict) |
| **OS** | Any (macOS, Linux, Windows) | Linux recommended | Linux required |
| **GPU** | Optional (CUDA or Apple MPS) | NVIDIA required | NVIDIA required |
| **CUDA** | N/A | 12.4 | 11.8 |
| **VRAM** | N/A | 8 GB minimum | 8 GB minimum |
| **RAM** | 4 GB | 16 GB | 16 GB |

SonarSplat and WaterSplatting **cannot** share a conda environment.

---

## 1. Data / Preprocessing Pipeline

This environment runs the preprocessing scripts (`preprocessing/` folder) and the optional EDA/download tools (`eda_and_dataset_downloads/`).

### Clone the repository

```bash
git clone <repo-url>
cd 26-S-Lockheed-1-clean
```

### Create virtual environment and install

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### Key packages installed

| Package | Purpose |
|---------|---------|
| `torch` | Deep learning framework (CLIP inference, SAM) |
| `torchvision` | Image transforms and pretrained models |
| `transformers` | CLIP model from Hugging Face |
| `segment-anything` | SAM model for artifact segmentation |
| `opencv-python` | Computer vision operations (Sobel edges, connected components) |
| `Pillow` | Image loading and manipulation |
| `numpy` | Numerical arrays |
| `scipy` | Signal processing (peak detection, Gaussian filtering) |
| `matplotlib` | Plotting and visualization |
| `tqdm` | Progress bars |

The SAM model checkpoint (`sam_vit_h_4b8939.pth`, ~2.5 GB) must be downloaded separately. See the [[Preprocessing Pipeline]] page for details.

### Verify

```bash
# Core packages load correctly
python -c "import torch, cv2, numpy; print('Core OK')"

# Device detection
python -c "import torch; print('CUDA' if torch.cuda.is_available() else 'MPS' if hasattr(torch.backends,'mps') and torch.backends.mps.is_available() else 'CPU')"
```

---

## 2. SonarSplat (Gaussian Splatting for Sonar)

SonarSplat requires a CUDA-capable NVIDIA GPU and has its own conda environment. This is only needed if you are doing 3D reconstruction from sonar data.

### Create conda environment

```bash
conda create -n sonarsplat python=3.10 -y
conda activate sonarsplat
```

### Install PyTorch with CUDA 12.4

```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124
```

### Install CUDA nvcc

```bash
conda install -c "nvidia/label/cuda-12.4.1" cuda-nvcc -y
```

Do **not** install `cuda-toolkit` from the default nvidia channel — it may pull CUDA 13.x which mismatches PyTorch's CUDA 12.4.

### Build gsplat with CUDA extensions

```bash
cd sonar_splat
pip install ninja setuptools wheel

export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="8.6"   # adjust for your GPU — see table below
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

pip install --no-build-isolation -e .
```

**GPU compute capability values:**

| GPU | Compute Capability |
|-----|-------------------|
| RTX 2070/2080 (Turing) | 7.5 |
| A100 (Ampere) | 8.0 |
| RTX 3070/3080/3090 (Ampere) | 8.6 |
| RTX 4070/4080/4090 (Ada) | 8.9 |
| H100 (Hopper) | 9.0 |

Find yours at [developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus).

### Build fused-ssim

```bash
cd /path/to/sonar_splat
git clone https://github.com/jparismorgan/fused-ssim.git fused-ssim-src
cd fused-ssim-src && pip install --no-build-isolation -e . && cd ..
```

### Install remaining dependencies

```bash
pip install -r examples/requirements.txt
pip install nerfacc
```

### Initialize submodules

```bash
git submodule update --init --recursive
```

### Fix known code issues

Comment out the unused pycolmap import in `sonar/dataset/dataloader.py`, line 11:

```python
# from pycolmap import SceneManager
```

### Persist environment variables (optional)

To avoid setting exports every session:

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export CUDA_HOME=$CONDA_PREFIX' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

### Verify

```bash
conda activate sonarsplat
nvcc --version                        # Should show CUDA 12.4
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import gsplat; print('gsplat OK')"
```

Full platform-specific instructions: `sonar_splat/SETUP_LINUX.md` and `sonar_splat/SETUP_WINDOWS.md`.

---

## 3. WaterSplatting (Gaussian Splatting for Optical RGB)

WaterSplatting requires a separate conda environment from SonarSplat due to different PyTorch/CUDA versions. This is only needed if you are doing 3D reconstruction from optical camera footage.

### Create conda environment

```bash
conda create --name water_splatting -y python=3.8
conda activate water_splatting
python -m pip install --upgrade pip
```

### Install PyTorch with CUDA 11.8 (must be first)

```bash
pip uninstall torch torchvision functorch tinycudann
python3 -m pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118
```

### Install CUDA toolkit and GCC 11

```bash
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
conda install -y -c conda-forge gcc=11 gxx=11
```

GCC 11 is required for CUDA kernel compilation. GCC 12+ will cause build failures.

### Set environment variables

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export LDFLAGS="-L$CONDA_PREFIX/lib -L/usr/lib/x86_64-linux-gnu"
```

### Install tiny-cuda-nn

```bash
python3 -m pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

This compiles CUDA code and may take 5-10 minutes.

### Install Nerfstudio

```bash
pip install nerfstudio==1.1.4
ns-install-cli
```

### Install WaterSplatting

```bash
# From the repo root
python3 -m pip install --no-use-pep517 -e .
```

### Verify

```bash
conda activate water_splatting

gcc --version | head -1              # Should show 11.x
nvcc --version | tail -1             # Should show 11.8

python -c "
import torch
print(f'PyTorch CUDA: {torch.version.cuda}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
"
```

Expected output: GCC 11.x, CUDA 11.8, PyTorch CUDA 11.8, CUDA available True.
