# Underwater 3D Reconstruction Pipeline

**3D Gaussian Splatting for Synthetic Underwater 3D Model Generation**

A unified pipeline for underwater dataset acquisition, quality analysis, image preprocessing, and 3D Gaussian splatting reconstruction using both optical and sonar imagery. This repository consolidates tools for the complete workflow: from raw dataset download through image classification and artifact removal to novel view synthesis and 3D scene reconstruction.

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
  - [5. SonarSplat: Gaussian Splatting for Sonar](#5-sonarsplat-gaussian-splatting-for-sonar)
- [Dependencies](#dependencies)
- [Branch Origins](#branch-origins)

---

## Project Overview

This project addresses the challenge of generating synthetic 3D models of underwater environments using Gaussian splatting techniques. The pipeline handles two complementary data modalities:

- **Optical imagery** -- RGB camera frames from underwater vehicles (ROVs/AUVs), used for visual 3D reconstruction of scenes like shipwrecks, coral reefs, and seafloor terrain.
- **Acoustic sonar imagery** -- Imaging sonar data (side-scan sonar, forward-looking sonar, multibeam) for 3D reconstruction in low-visibility or turbid conditions where optical cameras fail.

The pipeline is organized into stages that can be run independently or chained together:

1. **Acquire** datasets from Harvard Dataverse and other sources
2. **Analyze** dataset quality through EDA and statistical profiling
3. **Classify** images using CLIP zero-shot classification to filter out unusable frames
4. **Clean** images by detecting and cropping camera housing artifacts
5. **Reconstruct** 3D scenes using SonarSplat (for sonar data) or other splatting frameworks (for optical data)

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
                    (WaterSplatting /           (SonarSplat)
                     other frameworks)
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
.
|-- download_dataset.py                 # Download files from Harvard Dataverse
|-- download_and_rank.py                # Download + incremental CLIP ranking pipeline
|-- rank_images.py                      # Standalone two-pass image ranking (CLIP + CV)
|-- preprocess_datasets.py              # CLIP zero-shot classification + artifact detection + cleaning
|-- sunboat_crop.py                     # Sunboat camera mount detection (connected component)
|-- sunboat_templatematching.py         # Sunboat camera mount detection (template matching)
|-- vehicle_artifact_crop.py            # Generic vehicle housing artifact detection + cropping
|-- underwater_optical_datasets_analysis.py  # Catalog of underwater datasets with suitability scores
|-- underwater_object_detection_pipeline.ipynb  # YOLO + CLIP object detection notebook
|-- sonar_datasets_eda.ipynb            # Exploratory data analysis for sonar datasets
|-- sonar_datasets_eda.html             # Exported HTML of EDA notebook
|-- sonar_datasets_summary.csv          # Summary statistics for 21 sonar datasets
|-- sonar_analysis_report.txt           # FLSea dataset analysis report
|-- sonar_dataset_analysis.png          # FLSea dataset visualization
|-- eda_slides_prompt.md                # Prompt template for generating EDA presentation
|-- requirements.txt                    # Python dependencies for data analysis pipeline
|
|-- sonar_splat/                        # SonarSplat Gaussian splatting framework
|   |-- README.md                       # SonarSplat documentation and usage
|   |-- SETUP_LINUX.md                  # Linux installation guide
|   |-- SETUP_WINDOWS.md               # Windows installation guide
|   |-- setup.py                        # Package installation (gsplat with CUDA extensions)
|   |-- requirements.txt               # SonarSplat-specific dependencies
|   |-- gsplat/                         # Core Gaussian splatting library
|   |   |-- cuda/                       # CUDA kernels for rasterization
|   |   |-- compression/               # Point cloud compression utilities
|   |   |-- rendering.py               # Rasterization rendering pipeline
|   |   |-- strategy/                   # Gaussian densification strategies
|   |   +-- ...
|   |-- sonar/                          # Sonar-specific modules
|   |   |-- dataset/dataloader.py       # Sonar image loading and point initialization
|   |   |-- convert_to_cartesian.py     # Polar-to-Cartesian sonar image conversion
|   |   |-- utils.py                    # 3D Gaussian visualization and sonar utilities
|   |   +-- img_metrics.py             # Image quality metrics (PSNR, SSIM, LPIPS)
|   |-- examples/
|   |   |-- sonar_simple_trainer.py     # Main sonar training script
|   |   |-- sonar_image_fitting.py      # Single-image sonar fitting
|   |   |-- simple_trainer.py           # Standard (optical) trainer
|   |   +-- ...
|   |-- scripts/
|   |   |-- run_3D_monohansett.sh       # 3D reconstruction training script
|   |   |-- run_nvs_infra_360_1.sh      # Novel view synthesis training script
|   |   |-- compute_pcd_metrics_ply.py  # Point cloud evaluation metrics
|   |   |-- evaluate_imgs.py           # Image quality evaluation
|   |   +-- mesh_gaussian.py           # Convert Gaussians to mesh
|   +-- docker/                         # Docker build configuration
+-- README.md                           # This file
```

---

## Setup and Installation

### Data Analysis Pipeline (Ranking, Classification, Preprocessing)

The data analysis tools require Python 3.9+ and standard ML libraries.

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

**Key dependencies**: `torch`, `transformers` (CLIP), `ultralytics` (YOLOv8), `opencv-python`, `Pillow`, `scipy`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `tqdm`, `requests`, `py7zr`

### SonarSplat (Gaussian Splatting for Sonar)

SonarSplat requires a CUDA-capable GPU and has its own installation process. See `sonar_splat/SETUP_LINUX.md` or `sonar_splat/SETUP_WINDOWS.md` for detailed platform-specific instructions.

Quick start (Linux):

```bash
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

---

## Pipeline Components

### 1. Dataset Acquisition and Ranking

#### download_dataset.py

Downloads all files from a Harvard Dataverse dataset (default: FLSea, DOI `10.7910/DVN/VZD5S6`), organizing them by file type (sonar, images, metadata, ROS bags).

```bash
python download_dataset.py
```

#### download_and_rank.py

Combined download and ranking pipeline. Downloads archives from Harvard Dataverse one at a time, extracts images, scores them using CLIP similarity + computer vision heuristics, and retains only the top-K most useful images (default K=200). Keeps disk usage bounded.

```bash
# Basic usage (auto-detects CUDA, keeps top 200)
python download_and_rank.py --outdir ranked_dataset --k 200

# With GPU acceleration and larger batch
python download_and_rank.py --outdir ranked_dataset --device cuda --batch-size 32

# Custom scoring weights
python download_and_rank.py --weight-clip-positive 0.5 --weight-entropy 0.1
```

**Scoring dimensions** (configurable weights):
- `clip_positive` (50%) -- CLIP similarity to prompts describing objects of interest (shipwrecks, debris, structures)
- `clip_negative` (20%) -- CLIP dissimilarity to prompts describing empty/boring scenes
- `entropy` (5%) -- Shannon entropy of grayscale histogram (information content)
- `laplacian_var` (5%) -- Laplacian variance (image sharpness)
- `saturation_penalty` (10%) -- Penalty for over/under-exposed pixels
- `edge_density` (10%) -- Canny edge density (structural content)

**Output**: `ranked_dataset/results/` containing `top_images.jsonl`, `top_images.csv`, and `summary.json` with per-image scores and neighbor context.

#### rank_images.py

Standalone two-pass image ranker that works on a local directory of images (no download step). Uses the same CLIP + CV heuristics scoring as `download_and_rank.py`.

```bash
python rank_images.py --root /path/to/images --k 200 --outdir results/
```

### 2. Exploratory Data Analysis

#### sonar_datasets_eda.ipynb

Jupyter notebook performing comprehensive EDA across 21 open sonar datasets. Analyzes sonar type distributions (SSS, FLS, MBES, MSIS, SAS), annotation tasks, dataset sizes, temporal trends, data quality, and completeness. Exports summary statistics and visualizations.

Viewable as HTML: `sonar_datasets_eda.html`

#### underwater_optical_datasets_analysis.py

Catalogs underwater optical datasets scored for Gaussian splatting suitability. Covers enhancement/restoration datasets (SQUID, UIEB, EUVP, UVEB, etc.), detection datasets, segmentation datasets, tracking datasets, and classification datasets. Each scored 0-100 for optical suitability.

```bash
python underwater_optical_datasets_analysis.py
```

#### sonar_datasets_summary.csv

Tabular summary of 21 sonar datasets with columns: name, sonar type, annotation task, sample count, year, paper status, setup availability, and completeness score.

### 3. Image Classification and Preprocessing

#### preprocess_datasets.py

Two-phase pipeline for preparing underwater image datasets for 3D reconstruction.

**Phase 1 -- Analyze**: Uses CLIP zero-shot classification to categorize every image into one of 14 underwater categories (fish, coral, marine_animal, vegetation, sand_seafloor, rocks, shipwreck_debris, man_made_structure, diver, open_water, murky_turbid, dark_overexposed, blurry_corrupt). Also detects static border artifacts using cross-image statistical analysis.

```bash
python preprocess_datasets.py analyze --outdir classification_output --device auto --batch-size 16
```

**Phase 2 -- Clean**: Reads Phase 1 results, removes images classified as "empty" (open water, murky, dark, blurry), crops detected artifacts, and exports the cleaned dataset.

```bash
python preprocess_datasets.py clean --analysis-dir classification_output --outdir cleaned_datasets
```

**Supported datasets** (configurable in `DATASET_CONFIGS`):
- `flsea_vi` -- FLSEA-VI underwater scenes (TIFF images)
- `shipwreck` -- Shipwreck survey recordings (JPEG images)
- `sunboat` -- Sunboat mission recordings (PNG images)

### 4. Artifact Detection and Cropping

Three specialized tools for detecting and removing camera housing artifacts from underwater images:

#### vehicle_artifact_crop.py

Generic vehicle/camera housing artifact detector using median compositing and brightness profiling. Works with any underwater dataset. Uses three detection signals: median image analysis, brightness profile analysis, and gradient-based boundary detection.

```bash
# Detect artifacts and generate diagnostic plots
python vehicle_artifact_crop.py detect --datasets sunboat shipwreck flsea_vi

# Preview before/after crop on sample images
python vehicle_artifact_crop.py preview --datasets sunboat --crop bottom=80

# Apply crop to all images
python vehicle_artifact_crop.py apply --datasets sunboat --output-dir cropped_datasets
```

#### sunboat_crop.py

Specialized camera mount detector for the Sunboat dataset. Uses connected component analysis on the yellow-green hue band (hue 25-75) to identify the camera mount bar, which always appears on the right side of the image.

```bash
python sunboat_crop.py              # Preview only
python sunboat_crop.py --crop       # Preview + crop all images
```

#### sunboat_templatematching.py

Alternative Sunboat mount detector using OpenCV template matching (`cv2.TM_CCOEFF_NORMED`) against a pre-cropped reference image of the camera mount corner.

```bash
python sunboat_templatematching.py              # Preview only
python sunboat_templatematching.py --crop       # Preview + crop all images
```

### 5. SonarSplat: Gaussian Splatting for Sonar

SonarSplat is a Gaussian splatting framework for imaging sonar that enables novel view synthesis and 3D reconstruction of underwater scenes from sonar data. Based on the IEEE RA-L 2025 paper by Sethuraman et al.

**Key capabilities**:
- Novel view synthesis of imaging sonar (+3.2 dB PSNR vs. state-of-the-art)
- 3D reconstruction from sonar imagery (77% lower Chamfer Distance)
- Azimuth streak modeling and removal
- Polar-to-Cartesian coordinate conversion for sonar images

For full SonarSplat documentation, see `sonar_splat/README.md`.

#### Novel View Synthesis Training

```bash
cd sonar_splat
bash scripts/run_nvs_infra_360_1.sh <data_dir> <results_dir>
```

#### 3D Reconstruction Training

```bash
cd sonar_splat
bash scripts/run_3D_monohansett.sh <data_dir> <results_dir>
```

#### Evaluation

```bash
# Image quality metrics (PSNR, SSIM, LPIPS)
python scripts/evaluate_imgs.py --root_folder <root_folder>

# Point cloud metrics (Chamfer Distance)
python scripts/compute_pcd_metrics_ply.py --gt_root <gt_dir> --pred_root <pred_dir>
```

#### Polar-to-Cartesian Conversion

Convert sonar images from polar (range/azimuth) to Cartesian coordinates:

```bash
python sonar/convert_to_cartesian.py \
    --input sonar_image.png \
    --output cartesian.png \
    --hfov 130 \
    --max_range 10.0 \
    --cmap viridis
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

Requires PyTorch with CUDA support, gsplat (custom CUDA extension built from source), open3d, wandb, nerfview, viser, torchmetrics, fused-ssim, and other specialized dependencies. See `sonar_splat/SETUP_LINUX.md` for complete installation instructions.

---

## Branch Origins

This unified branch consolidates work from three feature branches:

| Branch | Contribution | Key Files |
|--------|-------------|-----------|
| `flSea_analysis` | Dataset download/ranking pipeline, FLSea EDA, sonar dataset analysis | `download_and_rank.py`, `download_dataset.py`, `rank_images.py`, `sonar_datasets_eda.ipynb`, `sonar_analysis_report.txt` |
| `image_class` | CLIP zero-shot classification, image preprocessing, artifact detection/cropping | `preprocess_datasets.py`, `sunboat_crop.py`, `sunboat_templatematching.py`, `vehicle_artifact_crop.py`, `eda_slides_prompt.md` |
| `sonar-splat` | SonarSplat Gaussian splatting framework for imaging sonar | `sonar_splat/` directory (gsplat core, sonar modules, training scripts, evaluation tools) |

**Shared between flSea_analysis and image_class**: `download_and_rank.py`, `download_dataset.py`, `rank_images.py`, `requirements.txt`, `underwater_optical_datasets_analysis.py`, `underwater_object_detection_pipeline.ipynb`. Where versions differed (EDA notebook, CSV summary), the more comprehensive `image_class` version was kept.
