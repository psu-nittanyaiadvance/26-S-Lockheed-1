# Underwater 3D Reconstruction Pipeline

**Complete end-to-end pipeline for underwater 3D reconstruction using Gaussian Splatting**

This repository combines dataset exploration, preprocessing, and 3D reconstruction tools for underwater imagery, specifically designed for the Lockheed Martin NAISS project on synthetic underwater 3D model generation.

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Components](#components)
4. [Installation](#installation)
5. [Usage Guide](#usage-guide)
6. [Branch Origins & Integration](#branch-origins--integration)
7. [Contributing](#contributing)
8. [Resources](#resources)

---

## Overview

This repository provides a complete pipeline for underwater 3D reconstruction:

1. **Dataset Exploration** - Analyze sonar and optical underwater datasets
2. **Image Classification & Ranking** - Use CLIP to identify high-quality reconstruction targets
3. **Preprocessing** - Clean images, remove artifacts, filter low-quality frames
4. **Template Matching** - Detect and isolate specific objects (shipwrecks, vehicles)
5. **3D Reconstruction** - Generate 3D models using WaterSplatting (Gaussian Splatting for underwater scenes)

The pipeline is designed to work with multiple underwater dataset types:
- Side-scan sonar (SSS)
- Forward-looking sonar (FLS)
- Optical underwater imagery
- Multi-view video sequences

---

## Repository Structure

```
Lockheed_Sonar/
├── water_splatting/          # WaterSplatting 3D Gaussian Splatting implementation
│   ├── cuda/                 # CUDA kernels for GPU acceleration
│   ├── __init__.py           # Python bindings
│   └── _torch_impl.py        # PyTorch implementation
│
├── scripts/                  # Analysis and preprocessing tools
│   ├── dataset_tools/        # Dataset download and ranking
│   │   ├── download_and_rank.py      # Incremental download + CLIP ranking
│   │   ├── download_dataset.py       # Harvard Dataverse downloader
│   │   └── rank_images.py            # Image quality ranking with CLIP
│   │
│   ├── eda/                  # Exploratory Data Analysis
│   │   ├── optical_imagery_eda.py    # Comprehensive underwater EDA with HTML reports
│   │   └── underwater_optical_datasets_analysis.py  # Quick dataset analysis
│   │
│   ├── preprocessing/        # Image preprocessing pipeline
│   │   └── preprocess_datasets.py    # CLIP-based classification and artifact removal
│   │
│   └── template_matching/    # Object detection and isolation
│       ├── sunboat_crop.py           # Sunboat scene cropping
│       ├── sunboat_templatematching.py  # Template-based object detection
│       └── vehicle_artifact_crop.py  # Vehicle detection and cropping
│
├── Download Datasets/        # Scene creation and data preparation
│   ├── create_valid_scene.py         # Convert images to 3DGS-ready scenes
│   └── download_seaThruNerf.py      # SeaThr u NeRF dataset tools
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

## Components

### 1. WaterSplatting (3D Gaussian Splatting)

**Purpose**: Underwater-specific 3D reconstruction using Gaussian Splatting

**Key Features**:
- CUDA-accelerated rendering
- Custom CUDA kernels for forward/backward passes
- Integration with Nerfstudio
- Support for multi-view underwater imagery

**Workflow**:
```bash
# 1. Install WaterSplatting
python setup.py install

# 2. Create a scene from images
python "Download Datasets/create_valid_scene.py"

# 3. Train a model
ns-train water-splatting --data path/to/scene

# 4. View results
ns-viewer --load-config outputs/config.yml
```

### 2. Dataset Tools

**Purpose**: Download, rank, and filter underwater datasets

**Scripts**:
- `download_and_rank.py`: Downloads Harvard Dataverse datasets and ranks images using CLIP
- `download_dataset.py`: General-purpose dataset downloader with organization
- `rank_images.py`: Standalone CLIP-based image ranking

**Use Cases**:
- Finding the best N images from large datasets for reconstruction
- Filtering out empty/featureless underwater scenes
- Identifying shipwrecks and man-made objects

**Example**:
```bash
python scripts/dataset_tools/download_and_rank.py --k 200 --outdir ranked_dataset
```

### 3. Exploratory Data Analysis (EDA)

**Purpose**: Analyze dataset quality, content, and 3DGS readiness

**Scripts**:
- `optical_imagery_eda.py`: Generates comprehensive HTML reports with:
  - Color profile analysis (blue/red ratio, brightness distribution)
  - Quality metrics (blur detection, sharpness, edge density)
  - Content classification (coral, rock, seafloor, open water)
  - COLMAP reconstruction analysis (registration rate, reprojection error)
  - Gaussian Splatting readiness assessment

**Output**: Interactive HTML dashboard with 18+ charts

**Example**:
```bash
python scripts/eda/optical_imagery_eda.py /path/to/dataset
# Opens underwater_eda_report.html in browser
```

### 4. Preprocessing

**Purpose**: Clean and prepare datasets for 3D reconstruction

**Features**:
- **Phase 1 (Analyze)**: CLIP zero-shot classification, artifact detection
- **Phase 2 (Clean)**: Remove empty/blurry images, crop camera housing artifacts

**Example**:
```bash
# Phase 1: Analyze
python scripts/preprocessing/preprocess_datasets.py analyze --outdir analysis_output

# Phase 2: Clean
python scripts/preprocessing/preprocess_datasets.py clean --outdir cleaned_datasets
```

### 5. Template Matching

**Purpose**: Detect and isolate specific objects using computer vision

**Scripts**:
- `sunboat_templatematching.py`: Template-based shipwreck detection
- `vehicle_artifact_crop.py`: Vehicle detection and cropping
- `sunboat_crop.py`: Scene-specific cropping

**Use Case**: When you want to focus 3D reconstruction on a specific object (e.g., a shipwreck) rather than the entire scene

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8 (for WaterSplatting)
- GCC 11 (for CUDA compilation)
- Git

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd Lockheed_Sonar
git checkout main3  # Use the consolidated branch
```

### Step 2: Create Environment
```bash
conda create -n underwater_recon python=3.8 -y
conda activate underwater_recon
```

### Step 3: Install PyTorch with CUDA 11.8
```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Install CUDA Toolkit (for WaterSplatting)
```bash
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
conda install -c conda-forge gcc=11 gxx=11
```

### Step 5: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 6: Install WaterSplatting
```bash
# Install tiny-cuda-nn (required for WaterSplatting)
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# Install Nerfstudio
pip install nerfstudio==1.1.4
ns-install-cli

# Install WaterSplatting in editable mode
pip install --no-use-pep517 -e .
```

### Verify Installation
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import water_splatting; print('WaterSplatting OK')"
python -c "import transformers; print('CLIP OK')"
```

---

## Usage Guide

### Typical Workflow

#### 1. Download and Rank Dataset
```bash
# Download Harvard Dataverse dataset and keep top 200 images
python scripts/dataset_tools/download_and_rank.py \
    --outdir ranked_dataset \
    --k 200 \
    --device cuda \
    --batch-size 32
```

#### 2. Run EDA to Assess Quality
```bash
# Generate comprehensive analysis report
python scripts/eda/optical_imagery_eda.py ranked_dataset/images
# Opens: ranked_dataset/images/underwater_eda_report.html
```

#### 3. Preprocess Dataset
```bash
# Analyze and clean
python scripts/preprocessing/preprocess_datasets.py analyze --outdir preprocessed
python scripts/preprocessing/preprocess_datasets.py clean --outdir cleaned_images
```

#### 4. Create 3DGS Scene
```bash
# Convert images to COLMAP-ready scene
cd "Download Datasets"
python create_valid_scene.py
# Input: Path to image folder
# Output: ../watersplatting_data/{dataset}/{scene}/
```

#### 5. Train WaterSplatting Model
```bash
ns-train water-splatting \
    --data ../watersplatting_data/{dataset}/{scene} \
    --output-dir outputs/{scene}_run1
```

#### 6. View Results
```bash
ns-viewer --load-config outputs/{scene}_run1/water-splatting/{timestamp}/config.yml
# Access at http://localhost:7007
```

---

## Branch Origins & Integration

This repository is the result of merging three specialized branches:

### 1. **image_class** Branch
**Original Purpose**: Dataset exploration and image classification

**Contributed**:
- `sonar_datasets_eda.ipynb` - Sonar dataset analysis
- `download_and_rank.py` - CLIP-based image ranking
- `preprocess_datasets.py` - Dataset preprocessing pipeline
- `rank_images.py` - Image quality ranking
- Template matching scripts

**Key Innovation**: Uses OpenAI CLIP for zero-shot classification of underwater imagery to identify reconstruction-worthy scenes

### 2. **watersplatting_initial** Branch
**Original Purpose**: 3D Gaussian Splatting for underwater scenes

**Contributed**:
- `water_splatting/` - Complete WaterSplatting implementation
- `Download Datasets/create_valid_scene.py` - Scene creation tools
- `setup.py`, `pyproject.toml` - Package configuration
- WaterSplatting-specific dependencies

**Key Innovation**: CUDA-accelerated Gaussian Splatting optimized for underwater light propagation and scattering

### 3. **optical-eda** Branch
**Original Purpose**: Comprehensive optical imagery analysis

**Contributed**:
- `optical_imagery_eda.py` - Advanced EDA with HTML reports
- `underwater_optical_datasets_analysis.py` - Quick dataset inspection
- COLMAP reconstruction analysis
- Quality/color/content classification

**Key Innovation**: Single-pass image analysis (3x faster) with 18+ interactive visualizations for dataset quality assessment

### Integration Strategy

The merge consolidated:
- **Requirements**: Combined into a single comprehensive `requirements.txt` with flexible versioning where appropriate and pinned versions for critical dependencies (PyTorch, CUDA)
- **.gitignore**: Merged to include all Python, C/CUDA, IDE, and environment patterns
- **Code Organization**: Restructured into logical directories (`scripts/`, `notebooks/`, `outputs/`)
- **Removed Redundancy**: Kept best implementations where overlap existed (e.g., single EDA tool instead of three similar scripts)

---

## Key Features & Capabilities

### Dataset Support
- Sonar imagery (SSS, FLS, MSIS, MBES, SAS)
- Optical underwater imagery
- Multi-view video sequences
- COLMAP sparse reconstructions
- Poses + bounds (LLFF format)

### Analysis Capabilities
- **Color Analysis**: Blue/red ratio, brightness, saturation, hue
- **Quality Metrics**: Laplacian variance (blur), contrast, edge density, SNR
- **Content Classification**: Coral, rock, seafloor, open water, shipwrecks, marine life
- **COLMAP Metrics**: Registration rate, reprojection error, track length, 3D point count
- **Gaussian Splatting Readiness**: Automated scoring for reconstruction viability

### Preprocessing Features
- Static border artifact detection (camera housing, overlays)
- CLIP-based scene classification (12 categories)
- Empty/featureless image filtering
- Automatic cropping
- TIFF → PNG conversion

### 3D Reconstruction
- Multi-view geometry (COLMAP integration)
- Custom underwater light model
- Real-time viewer
- Nerfstudio compatibility

---

## Troubleshooting

### CUDA Compilation Errors
```bash
# Verify GCC version (must be 11.x)
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

## License

See [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Nittany AI Advance** - Penn State research group
- **Lockheed Martin** - Project sponsor
- **WaterSplatting Team** - Original underwater 3DGS implementation
- **Nerfstudio Community** - Training framework and tools

---

**Last Updated**: March 2026
**Main Branch**: `main3` (consolidated from `image_class`, `watersplatting_initial`, `optical-eda`)
