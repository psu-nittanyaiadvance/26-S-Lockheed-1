# Project Structure

## Directory Tree

```
.
├── README.md                                    Project overview
├── requirements.txt                             Python dependencies for data/preprocessing pipeline
├── setup.py                                     WaterSplatting package installation
├── pyproject.toml                               Build configuration
├── LICENSE                                      Project license
├── .gitignore                                   Git exclusions
├── .gitmodules                                  Git submodule configuration
│
│   ── Preprocessing Pipeline ───────────────────────────────────
│
├── preprocessing/
│   ├── artifact_layering.py                     Step 1: Median image + variance map
│   ├── artifact_edge_detection.py               Step 2: Sobel edge-based boundary detection
│   ├── sam.py                                   Step 3: SAM artifact segmentation
│   ├── crop.py                                  Step 4: Manual image cropping
│   └── Archived_Preprocessing/                  Older versions of the preprocessing pipeline (useful for reference)
│
│   ── EDA and Dataset Downloads (Optional) ─────────────────────
│
├── eda_and_dataset_downloads/
│   ├── download_and_rank.py                     Download + CLIP ranking (FLSea specific)
│   ├── download_dataset.py                      Bulk download from Harvard Dataverse
│   ├── rank_images.py                           Standalone image ranking (local files)
│   ├── optical_imagery_eda.py                   HTML EDA report for optical datasets
│   ├── underwater_optical_datasets_analysis.py  Optical dataset catalog with GS scores
│   ├── sonar_datasets_eda.ipynb                 Jupyter notebook: sonar dataset survey
│   └── Download Datasets/
│       ├── create_valid_scene.py                COLMAP scene creator for WaterSplatting
│       └── download_seaThruNerf.py              SeaThru-NeRF dataset downloader
│
│   ── WaterSplatting (Optical 3D Reconstruction) ───────────────
│
├── water_splatting/
│   ├── water_splatting.py                       Main model: Gaussians + water medium MLP
│   ├── water_splatting_config.py                Nerfstudio method configuration
│   ├── project_gaussians.py                     Gaussian projection to image plane
│   ├── rasterize.py                             Gaussian rasterization
│   ├── sh.py                                    Spherical harmonics for view-dependent color
│   ├── utils.py                                 Utility functions
│   ├── _torch_impl.py                           Pure PyTorch implementations
│   └── cuda/                                    Custom CUDA kernels
│
│   ── SonarSplat (Sonar 3D Reconstruction) ─────────────────────
│
├── sonar_splat/
│   ├── README.md                                SonarSplat documentation
│   ├── SETUP_LINUX.md                           Linux installation guide
│   ├── SETUP_WINDOWS.md                         Windows installation guide
│   ├── setup.py                                 Package installer (builds gsplat with CUDA)
│   ├── requirements.txt                         SonarSplat-specific dependencies
│   │
│   ├── gsplat/                                  Core Gaussian splatting library
│   │   ├── rendering.py                         Rasterization pipeline
│   │   ├── cuda/                                Custom CUDA kernels
│   │   ├── strategy/                            Gaussian densification/culling strategies
│   │   ├── compression/                         Point cloud compression
│   │   └── optimizers/                          Optimizer implementations
│   │
│   ├── sonar/                                   Sonar-specific modules
│   │   ├── dataset/dataloader.py                Sonar data loading + point initialization
│   │   ├── convert_to_cartesian.py              Polar to Cartesian conversion
│   │   ├── utils.py                             3D Gaussian visualization + sonar utilities
│   │   └── img_metrics.py                       Image quality metrics (PSNR, SSIM, LPIPS)
│   │
│   ├── examples/                                Training scripts
│   │   ├── sonar_simple_trainer.py              Main sonar training script
│   │   ├── sonar_image_fitting.py               Single-image sonar fitting
│   │   └── simple_trainer.py                    Standard optical trainer
│   │
│   ├── scripts/                                 Evaluation and utility scripts
│   │   ├── run_nvs_infra_360_1.sh               Novel view synthesis training launcher
│   │   ├── run_3D_monohansett.sh                3D reconstruction training launcher
│   │   ├── evaluate_imgs.py                     Image quality evaluation
│   │   ├── compute_pcd_metrics_ply.py           Point cloud metrics (Chamfer Distance)
│   │   └── mesh_gaussian.py                     Gaussian to mesh conversion
│   │
│   ├── lpipsPyTorch/                            LPIPS perceptual metric
│   ├── fused-ssim-src/                          Fused SSIM metric
│   ├── tests/                                   Unit tests
│   └── docker/                                  Docker build configurations
│
│   ── Output Directories (generated, not in git) ───────────────
│
├── outputs/                                     Analysis results and artifacts
│   ├── sonar_datasets_eda.html                  Interactive sonar analysis report
│   ├── sonar_datasets_summary.csv               Sonar dataset statistics
│   ├── analysis_output/                         EDA outputs
│   ├── artifact_analysis/                       Artifact detection results
│   ├── classification_output/                   Image classification results
│   ├── cleaned_datasets/                        Post-cleaning outputs
│   └── ranked_output/                           Ranked image outputs
│
└── yolov8m.pt                                   YOLOv8 medium model weights (~50 MB)
```

---

## What's Not in the Repo

Large binary outputs and downloaded data are gitignored:

- Model checkpoints and training outputs
- Downloaded dataset archives and extracted images
- `watersplatting_data/` directory (created by WaterSplatting data scripts)
- Rendered images and evaluation results
- `.venv/` virtual environment
- SAM checkpoint (`sam_vit_h_4b8939.pth`)
