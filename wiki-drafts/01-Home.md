# Underwater 3D Reconstruction Pipeline

A pipeline for 3D underwater scene reconstruction using Gaussian splatting on both optical (RGB camera) and acoustic (imaging sonar) data. Developed as part of the NAISS Advance Spring 2026 project with Lockheed Martin.

The pipeline covers image preprocessing and artifact removal through to novel view synthesis and 3D scene reconstruction. It supports two complementary data modalities:

- **Optical imagery** — RGB camera frames processed via WaterSplatting for 3D reconstruction in clear-to-moderate visibility
- **Acoustic sonar imagery** — Imaging sonar data processed via SonarSplat for 3D reconstruction in turbid or low-visibility conditions

## Pipeline at a Glance

```
Raw Underwater Images          (OPTIONAL) eda_and_dataset_downloads/
(from ROV/AUV cameras)         Download, rank, and analyze datasets.
        |                      These scripts were written for our
        |                      specific datasets and will not work
        |                      on arbitrary datasets out of the box.
        |                           |
        +----------+----------------+
                   |
                   v
         Step 1: artifact_layering.py
           Compute median image + variance map
           to identify persistent artifacts
                   |
                   v
         Step 2: artifact_edge_detection.py
           Detect artifact boundaries using
           Sobel edge detection
                   |
                   v
         Step 3: sam.py
           Segment artifacts using SAM
           (Segment Anything Model)
                   |
                   v
         Step 4: crop.py
           Crop artifacts from all images
           using results from Steps 1-3
                   |
                   v
         Cleaned, Artifact-Free Images
                   |
        +----------+----------+
        |                     |
        v                     v
  WaterSplatting          SonarSplat
  (optical images)        (sonar images)
        |                     |
        v                     v
   3D Model               3D Model
   (optical)              (sonar)
```

## Wiki Pages

| Page | Description |
|------|-------------|
| [[Getting Started]] | Prerequisites, installation for all three environments, verification |
| [[Pipeline Overview]] | Full workflow from raw data to 3D reconstruction |
| [[Preprocessing Pipeline]] | The 4-step artifact detection and removal pipeline: layering, edge detection, SAM, crop |
| [[EDA and Dataset Downloads (Optional)]] | Optional scripts we used for our own exploratory data analysis and dataset downloads (dataset-specific, not general-purpose) |
| [[SonarSplat]] | Gaussian splatting for imaging sonar — training, evaluation, 3D reconstruction |
| [[WaterSplatting]] | Gaussian splatting for optical underwater RGB — Nerfstudio plugin, COLMAP setup |
| [[Project Structure]] | Annotated directory tree |
| [[Troubleshooting]] | Common errors, hardware requirements, FAQ |

## Key Links

- [SonarSplat Paper (arXiv)](https://arxiv.org/abs/2504.00159)
- [SonarSplat Project Page](https://umfieldrobotics.github.io/sonarsplat3D/)
- [SonarSplat IEEE Xplore](https://ieeexplore.ieee.org/document/11223217)
- [WaterSplatting Paper (arXiv)](https://arxiv.org/pdf/2408.08206)
- [WaterSplatting GitHub](https://github.com/water-splatting/water-splatting)
