# 26-S-Lockheed-1

**Physics-informed 3D Gaussian Splatting for underwater imaging sonar.**

Two complementary models live on this branch (`z_splatting`):

| Model | Trainer | Input | Loss |
|-------|---------|-------|------|
| **SonarSplat v2** | `sonar_simple_trainer_v2.py` | Sonar-only PKL | Gamma NLL |
| **Z-Splat v2** | `train_v2.py` | RGB + sonar depth | Gamma NLL + L_cam |

Both add the same physics layer on top of their upstream baselines: per-Gaussian reflectivity `r_tilde`, ULA beam-pattern weighting, elevation constraint loss, and a reflectivity spatial regularizer.

---

## Quick start

```bash
conda activate sonarsplat

# SonarSplat v2 — any dataset, any duration
cd sonar_splat
bash scripts/run_v2.sh monohansett_3D "/media/priyanshu/2TB SSD/results" 30000

# Z-Splat v2 — any sonar-only PKL dataset (one-time conversion, then train)
cd "Download Datasets"
python convert_to_zsplat.py --data_dir "/media/priyanshu/2TB SSD/sonarsplat_dataset/monohansett_3D"

cd ../z_splatting
# We provide an interactive runner that asks for dataset path and version (v1/v2).
python run.py

# Alternatively, run interactively via Docker:
export DATASET_PATH="/media/priyanshu/2TB SSD"
bash docker_run.sh
```

All scripts accept extra flags at the end to override any hyperparameter:

```bash
# L1 baseline ablation
bash scripts/run_v2.sh monohansett_3D /tmp/test 5000 --z_loss_weight 0.0

# Sunboat (Oculus M1200d hardware)
bash scripts/run_sonar_v2.sh ~/datasets/sunboat_zsplat "/media/.../sunboat_pkl" /tmp/results 30000 \
    --n_array_elements 256 --element_spacing 0.000625 --center_frequency 1200000.0
```

All results land in a timestamped subdirectory; logs go to `logs/`.

---

## Datasets

| Dataset | Scene | Frames | Has GT mesh |
|---------|-------|--------|-------------|
| `monohansett_3D` | Real shipwreck | 1228 | yes |
| `concrete_piling_3D` | Piling, open water | 687 | yes |
| `infra_360_1` | Infrastructure, 360° | 336 | no |
| AONeuS | Synthetic turtle (RGB+sonar) | 60 | — |

All SonarSplat datasets: `/media/priyanshu/2TB SSD/sonarsplat_dataset/`  
AONeuS: `/media/priyanshu/2TB SSD/aoneus_dataset/`

---

## Benchmark results

### SonarSplat — monohansett_3D

| Run | Steps | PSNR | SSIM |
|-----|-------|------|------|
| Baseline (L1) | 2K | 15.93 | 0.632 |
| v2 Run 2 | 40K | 16.08 | 0.629 |
| v2 Run 3 | 40K | **19.29** | 0.291 |

### Z-Splat — AONeuS

| Run | Steps | PSNR | Notes |
|-----|-------|------|-------|
| Baseline (`train.py`, L1) | 30K | 36.94 | number to beat |
| v2 Run 3 (Z loss disabled) | 30K | 37.77 | — |
| v2 Run 4 (z_density fix) | 30K | 36.76 | — |
| Hparam best (Optuna trial 8) | 20K | **38.29** | — |

---

## Repository layout

```
26-S-Lockheed-1/
├── sonar_splat/                         # SonarSplat model (gsplat-based)
│   ├── examples/
│   │   ├── sonar_simple_trainer_v2.py   # v2 trainer — main entry point
│   │   └── sonar_simple_trainer.py      # upstream baseline
│   └── scripts/
│       ├── run_v2.sh                    # general launcher (any dataset/steps)
│       └── run_3D_monohansett_v2.sh     # dataset-specific legacy script
├── z_splatting/       # Z-Splat model (3DGS-based)
│   ├── train_v2.py                      # v2 trainer — main entry point
│   ├── train.py                         # upstream baseline
│   ├── gaussian_renderer/__init__.py    # patched: handles 2 or 4 rasterizer outputs
│   ├── scene/dataset_readers.py         # patched: depth histogram + zero-filter
│   └── scripts/
│       ├── run_sonar_v2.sh              # Universal sonar-only launcher (any PKL dataset)
│       ├── run_aoneus_v2.sh             # AONeuS launcher (RGB+sonar)
│       └── run_monohansett_v2.sh        # Monohansett-specific launcher (legacy)
├── Download Datasets/
│   ├── convert_to_zsplat.py             # Universal PKL → COLMAP converter (any dataset)
│   ├── create_valid_z_splat_scene.py    # AONeuS → COLMAP converter (one-time)
│   ├── convert_monohansett_zsplat.py    # Monohansett-specific converter (legacy)
│   └── convert_sunboat.py              # Sunboat → SonarSplat PKL converter
├── docker/
│   ├── Dockerfile                       # unified image (both models)
│   ├── requirements.txt                 # all pip deps
│   ├── build.sh                         # build image
│   └── run.sh                           # run / attach to container
├── logs/                                # all training logs
└── wiki/                                # detailed documentation
```

---

## Wiki

In-depth documentation lives in [`wiki/`](wiki/):

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

## Docker (recommended)

A pre-built image covers both models, all CUDA extensions, and Python 3.10 + PyTorch 2.6.0+cu124. No conda required inside the container.

**Build** (once, ~15–20 min — compiles gsplat, diff-gaussian-rasterization, simple-knn):

```bash
bash docker/build.sh
```

**Run** (drops you into a bash shell with GPU access and datasets mounted):

```bash
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

Datasets from the 2TB SSD are available at `/data` inside the container.  
The repo is live-mounted at `/workspace` — edits on the host are instantly visible.

| Script | What it does |
|--------|-------------|
| `docker/build.sh` | Build image tagged `$(whoami)_lockheed1:latest` |
| `docker/run.sh` | Create / resume / attach to container |
| `docker/run.sh restart` | Force-recreate the container |
| `docker/Dockerfile` | Image definition |
| `docker/requirements.txt` | All pip dependencies |

GPU architecture defaults to `8.6` (RTX 3080). Override with `TORCH_CUDA_ARCH=8.9 bash docker/build.sh` for RTX 4090.

---

## Local conda environment

```bash
conda activate sonarsplat   # Python 3.10, PyTorch 2.6.0+cu124, CUDA 12.4
```

`lockheed1` and `base` do not have gsplat — always use `sonarsplat`.

Kill Sunshine before long runs if GPU memory is tight.  
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set automatically by both v2 trainers.
# Underwater 3D Reconstruction Pipeline

**Complete end-to-end pipeline for underwater 3D reconstruction using Gaussian Splatting**

A unified pipeline for underwater dataset acquisition, quality analysis, image preprocessing, and 3D Gaussian splatting reconstruction using both optical and sonar imagery. This repository consolidates tools for the complete workflow: from raw dataset download through image classification and artifact removal to novel view synthesis and 3D scene reconstruction for the Lockheed Martin NAISS project.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Setup and Installation](#setup-and-installation)
- [Pipeline Components](#pipeline-components)
  - [1. Dataset Acquisition and Ranking](#1-dataset-acquisition-and-ranking)
  - [2. Exploratory Data Analysis](#2-exploratory-data-analysis)
  - [3. Image Classification and Preprocessing](#3-image-classification-and-preprocessing)
  - [4. Artifact Detection and Cropping](#4-artifact-detection-and-cropping)
  - [5. 3D Reconstruction](#5-3d-reconstruction)
    - [SonarSplat (Sonar Imagery)](#sonarsplat-sonar-imagery)
    - [WaterSplatting (Optical Imagery)](#watersplatting-optical-imagery)
- [Usage Guide](#usage-guide)
- [Dependencies](#dependencies)
- [Branch Origins](#branch-origins)
- [Troubleshooting](#troubleshooting)
- [Resources](#resources)

---

## Project Overview

This project addresses the challenge of generating synthetic 3D models of underwater environments using Gaussian splatting techniques. The pipeline handles two complementary data modalities:

- **Optical imagery** - RGB camera frames from underwater vehicles (ROVs/AUVs), used for visual 3D reconstruction of scenes like shipwrecks, coral reefs, and seafloor terrain using **WaterSplatting**
- **Acoustic sonar imagery** - Imaging sonar data (side-scan sonar, forward-looking sonar, multibeam) for 3D reconstruction in low-visibility or turbid conditions where optical cameras fail using **SonarSplat**

The pipeline is organized into stages that can be run independently or chained together:

1. **Acquire** datasets from Harvard Dataverse and other sources
2. **Analyze** dataset quality through EDA and statistical profiling
3. **Classify** images using CLIP zero-shot classification to filter out unusable frames
4. **Clean** images by detecting and cropping camera housing artifacts
5. **Reconstruct** 3D scenes using SonarSplat (for sonar data) or WaterSplatting (for optical data)

---

## Architecture

```
                    +-------------------+
                    | Harvard Dataverse |
                    | (FLSea dataset)   |
                    +--------+----------+
                             |
                    download_dataset.py
                    download_and_rank.py
                             |
                             v
                    +-------------------+
                    | Raw Image Dataset |
                    +--------+----------+
                             |
              +--------------+--------------+
              |                             |
     rank_images.py              preprocess_datasets.py
     (CLIP + CV scoring)         (CLIP zero-shot classification)
              |                             |
              v                             v
     +----------------+          +--------------------+
     | Top-K Ranked   |          | Per-image category |
     | Images (JSONL) |          | labels (CSV/JSON)  |
     +----------------+          +---------+----------+
                                           |
                              +------------+------------+
                              |            |            |
                    sunboat_crop.py   vehicle_     preprocess_
                    sunboat_template  artifact_    datasets.py
                    matching.py       crop.py      (clean phase)
                              |            |            |
                              v            v            v
                         +----------------------------+
                         | Cleaned, Cropped Dataset   |
                         +-------------+--------------+
                                       |
                         +-------------+--------------+
                         |                            |
                    Optical Pipeline            Sonar Pipeline
                    (WaterSplatting)            (SonarSplat)
                         |                            |
                         v                            v
                    +-----------+             +-----------+
                    | 3D Model  |             | 3D Model  |
                    | (optical) |             | (sonar)   |
                    +-----------+             +-----------+
```

---

## Repository Structure

```
Lockheed_Sonar/
├── water_splatting/          # WaterSplatting 3D Gaussian Splatting implementation
│   ├── cuda/                 # CUDA kernels for GPU acceleration
│   ├── __init__.py           # Python bindings
│   └── _torch_impl.py        # PyTorch implementation
│
├── sonar_splat/              # SonarSplat Gaussian splatting framework
│   ├── README.md             # SonarSplat documentation and usage
│   ├── SETUP_LINUX.md        # Linux installation guide
│   ├── SETUP_WINDOWS.md      # Windows installation guide
│   ├── setup.py              # Package installation (gsplat with CUDA extensions)
│   ├── requirements.txt      # SonarSplat-specific dependencies
│   ├── gsplat/               # Core Gaussian splatting library
│   │   ├── cuda/             # CUDA kernels for rasterization
│   │   ├── compression/      # Point cloud compression utilities
│   │   ├── rendering.py      # Rasterization rendering pipeline
│   │   └── strategy/         # Gaussian densification strategies
│   ├── sonar/                # Sonar-specific modules
│   │   ├── dataset/dataloader.py      # Sonar image loading and point initialization
│   │   ├── convert_to_cartesian.py    # Polar-to-Cartesian sonar image conversion
│   │   ├── utils.py                   # 3D Gaussian visualization and sonar utilities
│   │   └── img_metrics.py            # Image quality metrics (PSNR, SSIM, LPIPS)
│   ├── examples/
│   │   ├── sonar_simple_trainer.py    # Main sonar training script
│   │   ├── sonar_image_fitting.py     # Single-image sonar fitting
│   │   └── simple_trainer.py          # Standard (optical) trainer
│   ├── scripts/
│   │   ├── run_3D_monohansett.sh      # 3D reconstruction training script
│   │   ├── run_nvs_infra_360_1.sh     # Novel view synthesis training script
│   │   ├── compute_pcd_metrics_ply.py # Point cloud evaluation metrics
│   │   ├── evaluate_imgs.py          # Image quality evaluation
│   │   └── mesh_gaussian.py          # Convert Gaussians to mesh
│   └── docker/                        # Docker build configuration
│
├── scripts/                  # Analysis and preprocessing tools
│   ├── dataset_tools/        # Dataset download and ranking
│   │   ├── download_and_rank.py      # Incremental download + CLIP ranking
│   │   ├── download_dataset.py       # Harvard Dataverse downloader
│   │   └── rank_images.py            # Image quality ranking with CLIP
│   ├── eda/                  # Exploratory Data Analysis
│   │   ├── optical_imagery_eda.py    # Comprehensive underwater EDA with HTML reports
│   │   └── underwater_optical_datasets_analysis.py  # Quick dataset analysis
│   ├── preprocessing/        # Image preprocessing pipeline
│   │   └── preprocess_datasets.py    # CLIP-based classification and artifact removal
│   └── template_matching/    # Object detection and isolation
│       ├── sunboat_crop.py           # Sunboat scene cropping
│       ├── sunboat_templatematching.py  # Template-based object detection
│       └── vehicle_artifact_crop.py  # Vehicle detection and cropping
│
├── Download Datasets/        # Scene creation and data preparation
│   ├── create_valid_scene.py         # Convert images to 3DGS-ready scenes
│   └── download_seaThruNerf.py       # SeaThru NeRF dataset tools
│
├── notebooks/                # Jupyter notebooks for interactive analysis
│   ├── sonar_datasets_eda.ipynb      # Sonar dataset exploration
│   └── underwater_object_detection_pipeline.ipynb  # Object detection pipeline
│
├── outputs/                  # Analysis outputs and reports
│   ├── sonar_datasets_eda.html       # Interactive sonar analysis report
│   ├── sonar_datasets_summary.csv    # Dataset statistics
│   ├── sonar_analysis_report.txt     # Text summary
│   └── sonar_dataset_analysis.png    # Visualization
│
├── requirements.txt          # Consolidated Python dependencies
├── setup.py                  # WaterSplatting package installation
├── pyproject.toml            # Build configuration
└── README.md                 # This file
```

---

## Setup and Installation

### Prerequisites
- Python 3.8+ (Python 3.9+ for data analysis tools)
- CUDA 11.8 (for WaterSplatting) or CUDA 12.4 (for SonarSplat)
- GCC 11 (for CUDA compilation)
- Git

### Option 1: Data Analysis and Preprocessing Only

If you only need the data analysis tools (ranking, classification, preprocessing):

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Option 2: WaterSplatting (Optical 3D Reconstruction)

For optical imagery 3D reconstruction with WaterSplatting:

```bash
# Create environment
conda create -n watersplatting python=3.8 -y
conda activate watersplatting

# Install PyTorch with CUDA 11.8
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Install CUDA Toolkit
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
conda install -c conda-forge gcc=11 gxx=11

# Install dependencies
pip install -r requirements.txt

# Install tiny-cuda-nn (required for WaterSplatting)
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# Install Nerfstudio
pip install nerfstudio==1.1.4
ns-install-cli

# Install WaterSplatting in editable mode
pip install --no-use-pep517 -e .
```

### Option 3: SonarSplat (Sonar 3D Reconstruction)

For sonar imagery 3D reconstruction with SonarSplat:

```bash
# Create environment
conda create -n sonarsplat python=3.10 -y
conda activate sonarsplat

# Install PyTorch with CUDA 12.4
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# Install CUDA nvcc
conda install -c "nvidia/label/cuda-12.4.1" cuda-nvcc -y

# Build gsplat with CUDA extensions
cd sonar_splat
pip install ninja setuptools wheel
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="8.6"  # adjust for your GPU
pip install --no-build-isolation -e .

# Install remaining dependencies
pip install -r requirements.txt
pip install -r examples/requirements.txt
```

### Verify Installation

```bash
# Check CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check WaterSplatting (if installed)
python -c "import water_splatting; print('WaterSplatting OK')"

# Check CLIP (for data analysis)
python -c "import transformers; print('CLIP OK')"
```

---

## Pipeline Components

### 1. Dataset Acquisition and Ranking

#### download_dataset.py

Downloads all files from a Harvard Dataverse dataset (default: FLSea, DOI `10.7910/DVN/VZD5S6`), organizing them by file type (sonar, images, metadata, ROS bags).

```bash
python scripts/dataset_tools/download_dataset.py
```

#### download_and_rank.py

Combined download and ranking pipeline. Downloads archives from Harvard Dataverse one at a time, extracts images, scores them using CLIP similarity + computer vision heuristics, and retains only the top-K most useful images (default K=200). Keeps disk usage bounded.

```bash
# Basic usage (auto-detects CUDA, keeps top 200)
python scripts/dataset_tools/download_and_rank.py --outdir ranked_dataset --k 200

# With GPU acceleration and larger batch
python scripts/dataset_tools/download_and_rank.py --outdir ranked_dataset --device cuda --batch-size 32

# Custom scoring weights
python scripts/dataset_tools/download_and_rank.py --weight-clip-positive 0.5 --weight-entropy 0.1
```

**Scoring dimensions** (configurable weights):
- `clip_positive` (50%) - CLIP similarity to prompts describing objects of interest (shipwrecks, debris, structures)
- `clip_negative` (20%) - CLIP dissimilarity to prompts describing empty/boring scenes
- `entropy` (5%) - Shannon entropy of grayscale histogram (information content)
- `laplacian_var` (5%) - Laplacian variance (image sharpness)
- `saturation_penalty` (10%) - Penalty for over/under-exposed pixels
- `edge_density` (10%) - Canny edge density (structural content)

#### rank_images.py

Standalone two-pass image ranker that works on a local directory of images (no download step). Uses the same CLIP + CV heuristics scoring as `download_and_rank.py`.

```bash
python scripts/dataset_tools/rank_images.py --root /path/to/images --k 200 --outdir results/
```

### 2. Exploratory Data Analysis

#### sonar_datasets_eda.ipynb

Jupyter notebook performing comprehensive EDA across 21 open sonar datasets. Analyzes sonar type distributions (SSS, FLS, MBES, MSIS, SAS), annotation tasks, dataset sizes, temporal trends, data quality, and completeness. Exports summary statistics and visualizations.

Viewable as HTML: `outputs/sonar_datasets_eda.html`

#### optical_imagery_eda.py

Generates comprehensive HTML reports with:
- Color profile analysis (blue/red ratio, brightness distribution)
- Quality metrics (blur detection, sharpness, edge density)
- Content classification (coral, rock, seafloor, open water)
- COLMAP reconstruction analysis (registration rate, reprojection error)
- Gaussian Splatting readiness assessment

**Output**: Interactive HTML dashboard with 18+ charts

```bash
python scripts/eda/optical_imagery_eda.py /path/to/dataset
# Opens: underwater_eda_report.html in browser
```

#### underwater_optical_datasets_analysis.py

Catalogs underwater optical datasets scored for Gaussian splatting suitability. Covers enhancement/restoration datasets (SQUID, UIEB, EUVP, UVEB, etc.), detection datasets, segmentation datasets, tracking datasets, and classification datasets. Each scored 0-100 for optical suitability.

```bash
python scripts/eda/underwater_optical_datasets_analysis.py
```

### 3. Image Classification and Preprocessing

#### preprocess_datasets.py

Two-phase pipeline for preparing underwater image datasets for 3D reconstruction.

**Phase 1 - Analyze**: Uses CLIP zero-shot classification to categorize every image into one of 14 underwater categories (fish, coral, marine_animal, vegetation, sand_seafloor, rocks, shipwreck_debris, man_made_structure, diver, open_water, murky_turbid, dark_overexposed, blurry_corrupt). Also detects static border artifacts using cross-image statistical analysis.

```bash
python scripts/preprocessing/preprocess_datasets.py analyze --outdir classification_output --device auto --batch-size 16
```

**Phase 2 - Clean**: Reads Phase 1 results, removes images classified as "empty" (open water, murky, dark, blurry), crops detected artifacts, and exports the cleaned dataset.

```bash
python scripts/preprocessing/preprocess_datasets.py clean --analysis-dir classification_output --outdir cleaned_datasets
```

**Supported datasets** (configurable in `DATASET_CONFIGS`):
- `flsea_vi` - FLSEA-VI underwater scenes (TIFF images)
- `shipwreck` - Shipwreck survey recordings (JPEG images)
- `sunboat` - Sunboat mission recordings (PNG images)

### 4. Artifact Detection and Cropping

Three specialized tools for detecting and removing camera housing artifacts from underwater images:

#### vehicle_artifact_crop.py

Generic vehicle/camera housing artifact detector using median compositing and brightness profiling. Works with any underwater dataset. Uses three detection signals: median image analysis, brightness profile analysis, and gradient-based boundary detection.

```bash
# Detect artifacts and generate diagnostic plots
python scripts/template_matching/vehicle_artifact_crop.py detect --datasets sunboat shipwreck flsea_vi

# Preview before/after crop on sample images
python scripts/template_matching/vehicle_artifact_crop.py preview --datasets sunboat --crop bottom=80

# Apply crop to all images
python scripts/template_matching/vehicle_artifact_crop.py apply --datasets sunboat --output-dir cropped_datasets
```

#### sunboat_crop.py

Specialized camera mount detector for the Sunboat dataset. Uses connected component analysis on the yellow-green hue band (hue 25-75) to identify the camera mount bar, which always appears on the right side of the image.

```bash
python scripts/template_matching/sunboat_crop.py              # Preview only
python scripts/template_matching/sunboat_crop.py --crop       # Preview + crop all images
```

#### sunboat_templatematching.py

Alternative Sunboat mount detector using OpenCV template matching (`cv2.TM_CCOEFF_NORMED`) against a pre-cropped reference image of the camera mount corner.

```bash
python scripts/template_matching/sunboat_templatematching.py              # Preview only
python scripts/template_matching/sunboat_templatematching.py --crop       # Preview + crop all images
```

### 5. 3D Reconstruction

#### SonarSplat (Sonar Imagery)

SonarSplat is a Gaussian splatting framework for imaging sonar that enables novel view synthesis and 3D reconstruction of underwater scenes from sonar data. Based on the IEEE RA-L 2025 paper by Sethuraman et al.

**Key capabilities**:
- Novel view synthesis of imaging sonar (+3.2 dB PSNR vs. state-of-the-art)
- 3D reconstruction from sonar imagery (77% lower Chamfer Distance)
- Azimuth streak modeling and removal
- Polar-to-Cartesian coordinate conversion for sonar images

For full SonarSplat documentation, see `/Users/priyanshudey/Code/Lockheed_Sonar/sonar_splat/README.md`.

##### Novel View Synthesis Training

```bash
cd sonar_splat
bash scripts/run_nvs_infra_360_1.sh <data_dir> <results_dir>
```

##### 3D Reconstruction Training

```bash
cd sonar_splat
bash scripts/run_3D_monohansett.sh <data_dir> <results_dir>
```

##### Evaluation

```bash
# Image quality metrics (PSNR, SSIM, LPIPS)
python scripts/evaluate_imgs.py --root_folder <root_folder>

# Point cloud metrics (Chamfer Distance)
python scripts/compute_pcd_metrics_ply.py --gt_root <gt_dir> --pred_root <pred_dir>
```

##### Polar-to-Cartesian Conversion

Convert sonar images from polar (range/azimuth) to Cartesian coordinates:

```bash
python sonar/convert_to_cartesian.py \
    --input sonar_image.png \
    --output cartesian.png \
    --hfov 130 \
    --max_range 10.0 \
    --cmap viridis
```

#### WaterSplatting (Optical Imagery)

WaterSplatting is an underwater-specific 3D Gaussian Splatting implementation optimized for underwater light propagation and scattering.

**Key Features**:
- CUDA-accelerated rendering
- Custom CUDA kernels for forward/backward passes
- Integration with Nerfstudio
- Support for multi-view underwater imagery
- Custom underwater light model

##### Workflow

```bash
# 1. Create a scene from images
cd "Download Datasets"
python create_valid_scene.py
# Input: Path to image folder
# Output: ../watersplatting_data/{dataset}/{scene}/

# 2. Train a model
ns-train water-splatting \
    --data ../watersplatting_data/{dataset}/{scene} \
    --output-dir outputs/{scene}_run1

# 3. View results
ns-viewer --load-config outputs/{scene}_run1/water-splatting/{timestamp}/config.yml
# Access at http://localhost:7007
```

---

## Usage Guide

### Typical Workflow

#### For Optical Imagery (WaterSplatting)

```bash
# 1. Download and Rank Dataset
python scripts/dataset_tools/download_and_rank.py \
    --outdir ranked_dataset \
    --k 200 \
    --device cuda \
    --batch-size 32

# 2. Run EDA to Assess Quality
python scripts/eda/optical_imagery_eda.py ranked_dataset/images

# 3. Preprocess Dataset
python scripts/preprocessing/preprocess_datasets.py analyze --outdir preprocessed
python scripts/preprocessing/preprocess_datasets.py clean --outdir cleaned_images

# 4. Create 3DGS Scene
cd "Download Datasets"
python create_valid_scene.py

# 5. Train WaterSplatting Model
ns-train water-splatting \
    --data ../watersplatting_data/{dataset}/{scene} \
    --output-dir outputs/{scene}_run1

# 6. View Results
ns-viewer --load-config outputs/{scene}_run1/water-splatting/{timestamp}/config.yml
```

#### For Sonar Imagery (SonarSplat)

```bash
# 1. Download and prepare sonar dataset
python scripts/dataset_tools/download_dataset.py

# 2. Analyze sonar datasets
jupyter notebook notebooks/sonar_datasets_eda.ipynb

# 3. Train SonarSplat model
cd sonar_splat
bash scripts/run_nvs_infra_360_1.sh <data_dir> <results_dir>

# 4. Evaluate results
python scripts/evaluate_imgs.py --root_folder <root_folder>
```

---

## Dependencies

### Data Analysis Pipeline (`requirements.txt`)

| Package | Purpose |
|---------|---------|
| `torch >= 2.0.0` | Deep learning framework |
| `torchvision >= 0.15.0` | Image transforms and models |
| `transformers >= 4.30.0` | CLIP model for zero-shot classification and ranking |
| `ultralytics >= 8.0.0` | YOLOv8 object detection |
| `Pillow >= 10.0.0` | Image I/O and manipulation |
| `opencv-python >= 4.8.0` | Computer vision (edge detection, template matching) |
| `numpy >= 1.24.0` | Numerical computing |
| `scipy >= 1.10.0` | Scientific computing (Laplacian, interpolation) |
| `py7zr >= 0.20.0` | 7z archive extraction |
| `tqdm >= 4.65.0` | Progress bars |
| `requests >= 2.31.0` | HTTP client for Dataverse API |
| `pandas >= 2.0.0` | Data analysis |
| `matplotlib >= 3.7.0` | Plotting |
| `seaborn >= 0.12.0` | Statistical visualization |
| `datasets >= 2.14.0` | HuggingFace datasets |

### SonarSplat (`sonar_splat/requirements.txt`)

Requires PyTorch with CUDA support, gsplat (custom CUDA extension built from source), open3d, wandb, nerfview, viser, torchmetrics, fused-ssim, and other specialized dependencies. See `/Users/priyanshudey/Code/Lockheed_Sonar/sonar_splat/SETUP_LINUX.md` for complete installation instructions.

### WaterSplatting

Requires PyTorch with CUDA 11.8 support, tiny-cuda-nn, Nerfstudio, and custom CUDA extensions. See installation instructions above.

---

## Branch Origins

This unified `main` branch consolidates work from two specialized branches:

| Branch | Contribution | Key Files |
|--------|-------------|-----------|
| `main2` | SonarSplat Gaussian splatting framework for imaging sonar, dataset download/ranking pipeline, FLSea EDA, sonar dataset analysis, CLIP zero-shot classification, image preprocessing, artifact detection/cropping | `sonar_splat/` directory (gsplat core, sonar modules, training scripts, evaluation tools), `download_and_rank.py`, `download_dataset.py`, `rank_images.py`, `sonar_datasets_eda.ipynb`, `sonar_analysis_report.txt`, `preprocess_datasets.py`, `sunboat_crop.py`, `sunboat_templatematching.py`, `vehicle_artifact_crop.py`, `eda_slides_prompt.md` |
| `main3` | WaterSplatting 3D Gaussian Splatting for underwater optical scenes, comprehensive optical imagery analysis, scene creation tools | `water_splatting/` directory (CUDA kernels, PyTorch implementation), `Download Datasets/create_valid_scene.py`, `Download Datasets/download_seaThruNerf.py`, `scripts/eda/optical_imagery_eda.py`, `setup.py`, `pyproject.toml` |

**Overlap Resolution**: Where files existed in both branches, the versions were intelligently merged or the more comprehensive version was selected. Scripts were organized into logical directories (`scripts/dataset_tools/`, `scripts/eda/`, `scripts/preprocessing/`, `scripts/template_matching/`) for better code organization.

---

## Troubleshooting

### CUDA Compilation Errors

```bash
# Verify GCC version (must be 11.x for WaterSplatting)
gcc --version

# Check CUDA paths
echo $LD_LIBRARY_PATH
which nvcc
```

### Port Forwarding (Remote Server)

```bash
# Single-hop (recommended)
ssh -L 7007:localhost:7007 -J user@gateway user@compute

# Then open: http://localhost:7007
```

### ModuleNotFoundError

```bash
# Ensure PyTorch installed before WaterSplatting
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118

# Then install WaterSplatting
pip install -e .
```

---

## Resources

### Papers
- [WaterSplatting](https://arxiv.org/pdf/2408.08206) - 3D Gaussian Splatting for Underwater Scenes
- [SonarSplat](https://ieeexplore.ieee.org/) - IEEE RA-L 2025 paper on Gaussian Splatting for Imaging Sonar
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) - Original 3DGS paper
- [SeaThru-NeRF](https://sea-thru-nerf.github.io/) - Underwater neural radiance fields

### Datasets
- [REMARO OpenSonarDatasets](https://remaro.eu/) - 21 sonar datasets
- [Harvard Dataverse](https://dataverse.harvard.edu/) - BenthiCat, FishNet, UXO datasets
- [FLSEA-VI](https://www.kaggle.com/) - Forward-looking sonar underwater scenes

### Tools
- [Nerfstudio](https://docs.nerf.studio/) - NeRF training framework
- [COLMAP](https://colmap.github.io/) - Structure-from-Motion
- [gsplat](https://github.com/nerfstudio-project/gsplat) - Gaussian Splatting library

---

## Contributing

This is a research project for Lockheed Martin NAISS. For questions or collaboration:
- Review the code in each `scripts/` subdirectory
- Check `notebooks/` for interactive examples
- See `outputs/` for example analysis reports

---

## Acknowledgments

- **Nittany AI Advance** - Penn State research group
- **Lockheed Martin** - Project sponsor
- **SonarSplat Team** - Original sonar 3DGS implementation
- **WaterSplatting Team** - Original underwater optical 3DGS implementation
- **Nerfstudio Community** - Training framework and tools

---

**Last Updated**: March 2026
**Main Branch**: `main` (consolidated from `main2` and `main3`)
