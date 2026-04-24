# 26-S-Lockheed-1

**Physics-informed 3D Gaussian Splatting for underwater imaging sonar.**

A unified end-to-end pipeline for underwater 3D scene reconstruction, combining optical and sonar imagery via two complementary Gaussian splatting models developed for the Lockheed Martin NAISS project.

| Model | Trainer | Input | Loss |
|-------|---------|-------|------|
| **SonarSplat v2** | `sonar_splat/examples/sonar_simple_trainer_v2.py` | Sonar-only PKL | Gamma NLL + beam pattern |
| **Z-Splat v2** | `z_splatting/train_v2.py` | RGB + sonar depth | Gamma NLL + L_camera |

Both models add the same physics layer on top of their upstream baselines: per-Gaussian reflectivity `r_tilde`, ULA beam-pattern weighting, elevation constraint loss, and a reflectivity spatial regularizer.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Repository Layout](#repository-layout)
- [Datasets](#datasets)
- [Benchmark Results](#benchmark-results)
- [Docker (Recommended)](#docker-recommended)
- [Local Conda Environment](#local-conda-environment)
- [Pipeline Components](#pipeline-components)
- [Wiki](#wiki)
- [Dependencies](#dependencies)
- [Acknowledgments](#acknowledgments)

---

## Quick Start

```bash
conda activate sonarsplat

# ── SonarSplat v2 ── any dataset, any duration ──────────────────────────────
cd sonar_splat
bash scripts/run_v2.sh monohansett_3D /path/to/results 30000

# ── Z-Splat v2 ── sonar-only PKL dataset (one-time conversion, then train) ──
cd "Download Datasets"
python convert_to_zsplat.py --data_dir /path/to/sonarsplat_dataset/monohansett_3D

cd ../z_splatting
bash scripts/run_aoneus_v2.sh /path/to/results 30000

# ── Z-Splat v2 ── via Docker ─────────────────────────────────────────────────
export DATASET_PATH=/path/to/datasets
bash docker/run.sh
```

Both trainers accept extra flags to override any hyperparameter:

```bash
# L1 baseline ablation (sonar loss disabled)
bash scripts/run_v2.sh monohansett_3D /tmp/test 5000 --z_loss_weight 0.0

# Sunboat (Oculus M1200d sonar hardware)
bash scripts/run_sunboat_v2.sh ~/datasets/sunboat_zsplat /path/to/sunboat_pkl /tmp/results 30000 \
    --n_array_elements 256 --element_spacing 0.000625 --center_frequency 1200000.0
```

All results land in a timestamped subdirectory; logs go to `logs/`.

---

## Repository Layout

```
26-S-Lockheed-1/
├── sonar_splat/                             # SonarSplat v2 (gsplat-based)
│   ├── examples/
│   │   ├── sonar_simple_trainer_v2.py       # ← main entry point
│   │   └── sonar_simple_trainer.py          # upstream baseline (reference)
│   ├── scripts/
│   │   ├── run_v2.sh                        # universal launcher (any dataset / steps)
│   │   ├── run_3D_monohansett_v2.sh         # Monohansett dataset launcher
│   │   ├── run_aoneus_sonar_v2.sh           # AONeuS sonar launcher
│   │   ├── run_infra_360_1_v2.sh            # infra_360_1 launcher
│   │   ├── run_sunboat_v2.sh                # Sunboat (Oculus M1200d) launcher
│   │   ├── evaluate_imgs.py                 # PSNR / SSIM / LPIPS evaluation
│   │   ├── compute_pcd_metrics_ply.py       # Chamfer distance vs. GT mesh
│   │   └── mesh_gaussian.py                 # Gaussian → mesh export
│   ├── sonar/
│   │   ├── dataset/dataloader.py            # SonarSensorDataset — PKL loading
│   │   ├── convert_to_cartesian.py          # polar → Cartesian coordinate conversion
│   │   ├── utils.py                         # visualization & geometry utilities
│   │   └── img_metrics.py                   # NIQE, TV, ICV metrics
│   ├── gsplat/                              # modified gsplat with sonar rasterization
│   │   └── cuda/
│   │       ├── _torch_impl_sonar.py         # sonar-specific rasterization kernel
│   │       ├── _torch_impl.py               # standard 3DGS rasterization
│   │       ├── _torch_impl_2dgs.py          # 2DGS rasterization
│   │       └── _wrapper.py                  # CUDA bindings
│   ├── scene/gaussian_model.py              # Gaussian representation
│   ├── docker/
│   │   ├── container.Dockerfile             # SonarSplat-specific Docker image
│   │   ├── build.sh                         # build SonarSplat image
│   │   ├── entrypoint.sh                    # container entrypoint
│   │   └── run.sh                           # create / resume SonarSplat container
│   └── tests/                               # unit tests (rasterization, compression)
│
├── z_splatting/                             # Z-Splat v2 (3DGS-based)
│   ├── train_v2.py                          # ← main entry point
│   ├── train.py                             # upstream baseline (reference)
│   ├── convert.py                           # scene / dataset conversion utility
│   ├── full_eval.py                         # full evaluation pipeline
│   ├── render.py                            # novel view synthesis renderer
│   ├── metrics.py                           # PSNR / SSIM / LPIPS evaluation
│   ├── rl_loss_controller.py                # RL-based adaptive loss weighting
│   ├── run.py                               # interactive runner (fallback / local dev)
│   ├── gaussian_renderer/__init__.py        # patched: handles 2- or 4-output rasterizer
│   ├── scene/
│   │   ├── gaussian_model_v2.py             # GaussianModelV2 — per-Gaussian reflectivity
│   │   ├── gaussian_model.py                # base 3DGS Gaussian model
│   │   ├── dataset_readers.py               # COLMAP loading + depth histogram extraction
│   │   ├── cameras.py                       # camera data structures
│   │   └── colmap_loader.py                 # COLMAP binary/text parser
│   ├── scripts/
│   │   └── run_aoneus_v2.sh                 # AONeuS launcher (steps + passthrough args)
│   ├── arguments/                           # training argument definitions
│   ├── utils/                               # shared utility functions
│   └── submodules/
│       ├── diff-gaussian-rasterization      # CUDA rasterizer (3DGS)
│       └── simple-knn                       # k-NN acceleration
│
├── Download Datasets/                       # dataset preparation utilities
│   ├── convert_to_zsplat.py                 # universal PKL → COLMAP converter
│   ├── create_valid_z_splat_scene.py        # AONeuS → COLMAP converter
│   ├── convert_sunboat.py                   # Sunboat PKL → SonarSplat PKL converter
│   └── convert_monohansett_zsplat.py        # Monohansett-specific converter (legacy)
│
├── preprocessing/                           # image preprocessing
│   ├── artifact_edge_detection.py           # artifact removal via edge detection
│   ├── artifact_layering.py                 # artifact removal via hue-based layering
│   ├── crop.py                              # region extraction
│   ├── sam.py                               # Segment Anything integration
│   └── Archived_Preprocessing/             # earlier approaches (kept for reference)
│       ├── artifact_hueBased.py
│       ├── artifact_templatematch.py
│       ├── preprocess_datasets.py
│       └── sunboat_crop.py
│
├── eda_and_dataset_downloads/               # exploratory data analysis
│   ├── optical_imagery_eda.py               # optical image analysis & HTML report
│   ├── rank_images.py                       # CLIP + CV image quality ranking
│   ├── download_and_rank.py                 # download + rank pipeline
│   └── download_dataset.py                  # dataset download utility
│
├── water_splatting/                         # volumetric rendering extension
│
├── docker/
│   ├── Dockerfile                           # CUDA 12.4 + PyTorch 2.6.0 unified image
│   ├── requirements.txt                     # all pip dependencies
│   ├── build.sh                             # build image (~15–20 min CUDA compile)
│   └── run.sh                               # create / resume / attach to container
│
├── wiki/                                    # in-depth documentation
│   ├── Overview.md                          # research motivation, model comparison
│   ├── Physics.md                           # loss functions and math derivations
│   ├── Architecture.md                      # code structure, key classes, gradient paths
│   ├── Datasets.md                          # dataset descriptions and acquisition details
│   ├── Training.md                          # full training reference for every dataset
│   ├── Results.md                           # benchmark runs, ablations, key findings
│   └── Implementation.md                    # patches applied, known pitfalls
│
├── requirements.txt                         # consolidated Python dependencies
├── pyproject.toml                           # project metadata
└── README.md                                # this file
```

---

## Datasets

| Dataset | Scene | Modality | Frames | GT Mesh |
|---------|-------|----------|--------|---------|
| `monohansett_3D` | Real shipwreck | Sonar only | 1,228 | Yes |
| `concrete_piling_3D` | Concrete piling, open water | Sonar only | 687 | Yes |
| `infra_360_1` | Infrastructure, 360° sweep | Sonar only | 336 | No |
| AONeuS | Synthetic turtle | RGB + sonar | 60 | — |
| Sunboat | Open-water structure | Sonar (Oculus M1200d) | — | — |

Dataset paths are machine-specific. Update the paths in the training scripts or pass them as arguments. See [wiki/Datasets.md](wiki/Datasets.md) for acquisition and formatting details.

---

## Benchmark Results

### SonarSplat v2 — monohansett_3D

| Run | Steps | PSNR | SSIM | Notes |
|-----|-------|------|------|-------|
| Baseline (L1) | 2K | 15.93 | 0.632 | upstream baseline |
| v2 Run 2 | 40K | 16.08 | 0.629 | — |
| v2 Run 3 | 40K | **19.29** | 0.291 | best PSNR |

### Z-Splat v2 — AONeuS

| Run | Steps | PSNR | Notes |
|-----|-------|------|-------|
| Baseline (`train.py`, L1) | 30K | 36.94 | number to beat |
| v2 Run 3 (Z loss disabled) | 30K | 37.77 | — |
| v2 Run 4 (z_density fix) | 30K | 36.76 | — |
| Hparam best (Optuna trial 8) | 20K | **38.29** | — |

See [wiki/Results.md](wiki/Results.md) for full ablation tables and training curves.

---

## Docker (Recommended)

A pre-built image covers both models, all CUDA extensions, Python 3.10, and PyTorch 2.6.0+cu124. No conda required inside the container.

**Build** (once, ~15–20 min — compiles gsplat, diff-gaussian-rasterization, simple-knn):

```bash
bash docker/build.sh
```

GPU architecture defaults to `8.6` (RTX 3080/A5000). Override for other GPUs:

```bash
TORCH_CUDA_ARCH=8.9 bash docker/build.sh   # RTX 4090
TORCH_CUDA_ARCH=8.0 bash docker/build.sh   # A100
```

**Run** (drops into a bash shell with GPU access and datasets mounted):

```bash
export DATASET_PATH=/path/to/datasets
bash docker/run.sh
```

Inside the container:

```bash
# SonarSplat v2
cd /workspace/sonar_splat
bash scripts/run_v2.sh monohansett_3D /results 30000

# Z-Splat v2
cd /workspace/z_splatting
bash scripts/run_aoneus_v2.sh /results 30000
```

Datasets are available at `/data` inside the container. The repo is live-mounted at `/workspace` — host edits are instantly visible.

| Script | What it does |
|--------|-------------|
| `docker/build.sh` | Build image tagged `$(whoami)_lockheed1:latest` |
| `docker/run.sh` | Create / resume / attach to container |
| `docker/run.sh restart` | Force-recreate the container |

---

## Local Conda Environment

```bash
# Python 3.10, PyTorch 2.6.0+cu124, CUDA 12.4
conda create -n sonarsplat python=3.10 -y
conda activate sonarsplat

# PyTorch with CUDA 12.4
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# CUDA compiler
conda install -c "nvidia/label/cuda-12.4.1" cuda-nvcc -y

# gsplat CUDA extensions (from sonar_splat/)
cd sonar_splat
pip install ninja setuptools wheel
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="8.6"   # adjust for your GPU
pip install --no-build-isolation -e .

# Remaining dependencies
pip install -r requirements.txt
pip install -r examples/requirements.txt
```

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set automatically by both v2 trainers to mitigate GPU memory fragmentation on large scenes.

---

## Pipeline Components

### 1. SonarSplat v2

Physics-informed Gaussian splatting for imaging sonar. Implements:

- **Gamma NLL loss** — MLE for multiplicative (speckle) sonar noise: `L = mean(s_gt/s_pred + log(s_pred))`
- **ULA beam-pattern weighting** — per-Gaussian azimuth (sinc) and elevation (cosine) weights
- **Elevation constraint loss** — enforces range consistency between 3D Gaussians and measured sonar range
- **Reflectivity regularizer** — kNN-based spatial smoothness on per-Gaussian reflectivity `r_tilde`

See [wiki/Physics.md](wiki/Physics.md) for full derivations.

### 2. Z-Splat v2

Multi-modal Gaussian splatting (RGB + sonar depth). Extends SonarSplat physics with:

- **Depth histogram supervision** — differentiable 1D sonar range histogram matched against rendered z-density
- **GaussianModelV2** — adds per-Gaussian reflectivity `r_tilde` in logit space
- **RL loss controller** — optional Optuna-backed adaptive weighting of sonar vs. camera losses

### 3. Dataset Tools

```bash
# Convert any sonar PKL dataset to COLMAP format for Z-Splat
cd "Download Datasets"
python convert_to_zsplat.py --data_dir /path/to/dataset

# Download and rank images (CLIP + CV heuristics)
python eda_and_dataset_downloads/download_and_rank.py --outdir ranked/ --k 200
```

### 4. Preprocessing

Active preprocessing scripts (in `preprocessing/`):

| Script | Purpose |
|--------|---------|
| `artifact_edge_detection.py` | Artifact removal via edge detection |
| `artifact_layering.py` | Artifact removal via hue-based layering |
| `crop.py` | Region extraction / cropping |
| `sam.py` | Segment Anything-based masking |

Archived/earlier approaches are in `preprocessing/Archived_Preprocessing/` and kept for reference.

```bash
# Artifact removal via edge detection
python preprocessing/artifact_edge_detection.py --input /path/to/images --output /path/to/cleaned

# Segment Anything-based masking
python preprocessing/sam.py --input /path/to/images
```

### 5. Evaluation

```bash
# Image quality (PSNR, SSIM, LPIPS)
cd sonar_splat
python scripts/evaluate_imgs.py --root_folder /path/to/results

# Point cloud metrics (Chamfer Distance vs. GT mesh)
python scripts/compute_pcd_metrics_ply.py --gt_root /path/to/gt --pred_root /path/to/pred

# Z-Splat full evaluation
cd z_splatting
python full_eval.py --model_path /path/to/checkpoint
```

---

## Wiki

In-depth documentation is in [`wiki/`](wiki/):

| Page | Contents |
|------|----------|
| [Overview](wiki/Overview.md) | Research motivation, model comparison |
| [Physics](wiki/Physics.md) | All loss functions and math derivations |
| [Architecture](wiki/Architecture.md) | Code structure, key classes, gradient paths |
| [Datasets](wiki/Datasets.md) | Dataset descriptions, paths, acquisition details |
| [Training](wiki/Training.md) | Full training reference for every dataset |
| [Results](wiki/Results.md) | All benchmark runs, ablations, key findings |
| [Implementation Notes](wiki/Implementation.md) | Bugs fixed, patches applied, known pitfalls |

---

## Dependencies

### Core (both models)

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.6.0+cu124 | deep learning framework |
| `torchvision` | 0.21.0+cu124 | image transforms |
| `numpy` | ≥1.24 | numerical computing |
| `opencv-python` | ≥4.8 | image I/O and processing |
| `open3d` | 0.19.0 | 3D point cloud / mesh handling |
| `wandb` | — | experiment tracking |
| `torchmetrics` | — | PSNR, SSIM, LPIPS |

### SonarSplat-specific

`gsplat` (custom CUDA extension, built from `sonar_splat/`), `nerfview`, `viser`, `fused-ssim`, `nerfacc`

### Z-Splat-specific

`diff-gaussian-rasterization` (submodule), `simple-knn` (submodule), `pytorch-msssim`, `optuna`

### Data Analysis / EDA

`transformers` (CLIP), `ultralytics` (YOLOv8), `pandas`, `seaborn`, `jupyterlab`

See `requirements.txt` and `docker/requirements.txt` for pinned versions.

---

## Publication

This work builds on:

> Sethuraman et al., *"Novel View Synthesis of Imaging Sonar via Gaussian Splatting"*,
> IEEE Robotics and Automation Letters, 2025.
> arXiv: [2504.00159](https://arxiv.org/abs/2504.00159) |
> [Project page](https://umfieldrobotics.github.io/sonarsplat3D/)

---

## Acknowledgments

- **Nittany AI Advance** — Penn State research group
- **Lockheed Martin** — project sponsor (NAISS program)
- **SonarSplat Team** (University of Michigan Robotics) — original sonar 3DGS implementation
- **WaterSplatting Team** — original underwater optical 3DGS implementation
- **Nerfstudio Community** — training framework and tooling
