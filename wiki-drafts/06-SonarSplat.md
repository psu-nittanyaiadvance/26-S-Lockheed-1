# SonarSplat

SonarSplat is a Gaussian splatting framework designed for imaging sonar. It enables novel view synthesis (rendering sonar images from new viewpoints) and 3D reconstruction (producing point clouds and meshes) from real-world sonar data.

**Publication:** IEEE Robotics and Automation Letters, 2025, Volume 10, Issue 12, Pages 13312-13319

**Authors:** Advaith V. Sethuraman, Max Rucker, Onur Bagoren, Pou-Chun Kung, Nibarkavi N.B. Amutha, Katherine A. Skinner — Department of Robotics, University of Michigan, Ann Arbor

**Key Results:**
- +3.2 dB PSNR improvement in novel view synthesis vs. state-of-the-art
- 77% lower Chamfer Distance in 3D reconstruction vs. baselines

---

## How It Works

### Scene Representation

SonarSplat represents underwater scenes as a collection of 3D Gaussians. Each Gaussian has:
- **Position** (x, y, z) in 3D space
- **Covariance** matrix (controls shape, size, and orientation)
- **Acoustic reflectance** — how strongly the surface reflects sonar signals
- **Saturation properties** — sonar-specific rendering parameters

### Custom Sonar Rasterizer

Unlike optical Gaussian splatting (which projects onto a 2D perspective image), SonarSplat projects Gaussians into a **range/azimuth image** that models the acoustic image formation process:

- The x-axis represents **azimuth** (horizontal angle)
- The y-axis represents **range** (distance from the sensor)
- Pixels encode **acoustic intensity** (how much sound was reflected back)

This projection is implemented as custom CUDA kernels in `gsplat/cuda/`.

### Azimuth Streak Modeling

Imaging sonar suffers from "azimuth streaking" — bright horizontal artifacts caused by limited angular resolution in the elevation dimension. SonarSplat explicitly models this phenomenon during training. After training, the streak model can be disabled for clean rendered images.

### Training Loop

1. Select a training viewpoint with known sensor pose
2. Rasterize all Gaussians from that viewpoint to produce a rendered sonar image
3. Compare the rendered image against the real sonar image (photometric loss)
4. Backpropagate gradients through the differentiable rasterizer
5. Update Gaussian parameters
6. Periodically densify (split/clone) and cull (remove transparent) Gaussians

---

## Installation

See [[Getting Started]] for the full setup. Summary:
- Python 3.10, conda environment
- PyTorch 2.6.0 with CUDA 12.4
- CUDA nvcc 12.4 compiler
- gsplat built from source with CUDA extensions
- fused-ssim built from source

---

## Data Format

SonarSplat expects data organized as follows:

```
dataset/
├── Data/                        Sensor data files
│   ├── 000000.pkl               Frame 0
│   ├── 000001.pkl               Frame 1
│   └── ...
├── sonar_images/                PNG renders for visual inspection
│   ├── 000000.png
│   └── ...
├── bounds.txt                   3D bounding box of the scene
├── Config.json                  Sonar sensor parameters (FOV, range, resolution)
└── gt.ply                       Ground truth point cloud (3D reconstruction only)
```

**`.pkl` files:** Each pickle file contains a dictionary with:
- `data['PoseSensor']` — 4x4 transformation matrix (sensor pose in world frame)
- `data['ImagingSonar']` — numpy array (sonar image data)

**`bounds.txt`:** 3D bounding box used to normalize Gaussian positions during training.

**`Config.json`:** Sonar sensor parameters (horizontal FOV, maximum range, angular/range resolution).

**`gt.ply`:** Ground truth point cloud, only for 3D reconstruction datasets. Used for Chamfer Distance evaluation.

### Available Datasets

Download from the [Google Drive link](https://drive.google.com/file/d/1sDGprDT-kS-Eunt5XjXAFMkMe1t3GIbd/view?usp=drive_link) in the SonarSplat README.

| Dataset | Environment | Use |
|---------|------------|-----|
| `basin_horizontal_empty1` | Controlled test tank (empty) | Baseline / calibration |
| `basin_horizontal_infra_1` | Test tank with infrastructure | Novel view synthesis |
| `monohansett_3D` | Real-world lake (Monohansett Shipwreck) | 3D reconstruction |

---

## Novel View Synthesis

Train SonarSplat to render sonar images from viewpoints not seen during training.

### Training

```bash
cd sonar_splat
bash scripts/run_nvs_infra_360_1.sh --data_dir <data_dir> --results_dir <results_dir>
```

### Evaluation

Three image quality metrics are used:
- **PSNR** (dB, higher is better) — pixel-level accuracy
- **SSIM** (0-1, higher is better) — perceptual structural similarity
- **LPIPS** (0-1, lower is better) — learned perceptual similarity

Organize rendered images:

```
root_dir/
├── baseline1/
│   └── scene1/
│       └── sonar_renders/
│           ├── gt_sonar_images/     Ground truth
│           └── sonar_images/        Rendered
└── sonarsplat/
    └── ...
```

```bash
python examples/evaluate_imgs.py --root_folder <root_folder>
```

---

## 3D Reconstruction

### Training

```bash
cd sonar_splat
bash scripts/run_3D_monohansett.sh --data_dir <data_dir> --results_dir <results_dir>
```

### Mesh Conversion

```bash
python scripts/mesh_gaussian.py --ply_path <results_dir>/renders/output_step<iter>.ply
```

### Alignment

Before computing metrics, predicted and ground truth point clouds **must** be aligned into the same coordinate frame using ICP (Iterative Closest Point) or manual alignment. Without alignment, Chamfer Distance is meaningless.

### Evaluation

```bash
python scripts/compute_pcd_metrics_ply.py --gt_root <gt_dir> --pred_root <pred_dir>
```

**Chamfer Distance** measures the average nearest-neighbor distance between predicted and ground truth point clouds. Lower is better.

---

## Polar-to-Cartesian Conversion

Converts sonar images from native polar coordinates (range/azimuth) to Cartesian (x/y) for visualization:

```bash
python sonar/convert_to_cartesian.py \
    --input sonar_image.png \
    --output cartesian.png \
    --hfov 130 \
    --max_range 10.0 \
    --cmap viridis
```

---

## Key Source Files

| File | Purpose |
|------|---------|
| `examples/sonar_simple_trainer.py` | Main training script |
| `sonar/dataset/dataloader.py` | Data loading and point cloud initialization |
| `gsplat/rendering.py` | Rasterization pipeline |
| `gsplat/cuda/` | Custom CUDA kernels |
| `sonar/convert_to_cartesian.py` | Polar to Cartesian conversion |
| `sonar/img_metrics.py` | PSNR, SSIM, LPIPS computation |
| `scripts/run_nvs_infra_360_1.sh` | Novel view synthesis training launcher |
| `scripts/run_3D_monohansett.sh` | 3D reconstruction training launcher |
| `scripts/evaluate_imgs.py` | Image quality evaluation |
| `scripts/compute_pcd_metrics_ply.py` | Point cloud metrics |
| `scripts/mesh_gaussian.py` | Gaussian to mesh conversion |

---

## Citation

```bibtex
@ARTICLE{11223217,
  author={Sethuraman, Advaith V. and Rucker, Max and Bagoren, Onur and Kung, Pou-Chun
          and Amutha, Nibarkavi N.B. and Skinner, Katherine A.},
  journal={IEEE Robotics and Automation Letters},
  title={SonarSplat: Novel View Synthesis of Imaging Sonar via Gaussian Splatting},
  year={2025},
  volume={10},
  number={12},
  pages={13312-13319},
  doi={10.1109/LRA.2025.3627089}
}
```

## License

SonarSplat is licensed under [CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/) — non-commercial use, share-alike, attribution required.
