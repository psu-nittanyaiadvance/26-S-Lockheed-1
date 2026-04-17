# WaterSplatting

WaterSplatting is a Gaussian splatting method for **optical (RGB) underwater imagery** that combines standard 3D Gaussian Splatting with a learned volumetric medium model to account for underwater light scattering and absorption. It is the optical counterpart to SonarSplat.

Built as a [Nerfstudio](https://docs.nerf.studio/) plugin.

**Paper:** [WaterSplatting (arXiv 2408.08206)](https://arxiv.org/pdf/2408.08206)
**Authors:** Huapeng Li, Wenxuan Song, Tianao Xu, Alexandre Elsig, Jonas Kulhanek
**Original Repository:** [github.com/water-splatting/water-splatting](https://github.com/water-splatting/water-splatting)

---

## Why a Water-Specific Method

Standard Gaussian splatting assumes clear air between the camera and the scene. Underwater, the water column scatters and absorbs light — images appear blue/green tinted, low contrast, and hazy. Objects farther from the camera are more affected. Standard 3DGS cannot distinguish between the scene's actual appearance and the water's optical effects, leading to color-shifted, foggy reconstructions.

WaterSplatting solves this by modeling the water and the scene separately:

**Scene rendering:** Standard 3D Gaussians are projected and rasterized into an image (same as vanilla 3DGS).

**Medium rendering:** A small MLP takes viewing direction as input and outputs per-pixel water medium properties — attenuation and scattering color.

**Combined rendering:** Scene and medium renders are composited together to produce the final image. Both are optimized jointly during training.

**Water removal:** After training, render with the medium disabled to see the scene without water effects — useful for inspection and analysis.

---

## How It Fits in the Pipeline

```
Cleaned images (from preprocessing pipeline)
        |
        v
create_valid_scene.py     Run COLMAP to estimate camera poses
        |
        v
watersplatting_data/      COLMAP reconstruction + images
        |
        v
ns-train water-splatting  Train the model
        |
        v
ns-viewer                 Interactive 3D viewer in browser
```

---

## Installation

See [[Getting Started]] for the complete setup. Key requirements:
- Python 3.8 (strict — Nerfstudio 1.1.4 requires it)
- PyTorch 2.1.2 with CUDA 11.8
- GCC 11 (not 12+)
- Nerfstudio 1.1.4
- tiny-cuda-nn (compiled from source)

---

## Preparing Data

### Option A: SeaThru-NeRF Dataset (Pre-processed)

Download a dataset that comes with white-balanced images and pre-computed COLMAP camera poses:

```bash
conda activate water_splatting
python "eda_and_dataset_downloads/Download Datasets/download_seaThruNerf.py"
```

This downloads and extracts to `../watersplatting_data/SeathruNeRF_dataset/`. Each scene is ready to train on immediately.

### Option B: Your Own Images

Convert a folder of underwater images into a WaterSplatting-ready scene:

```bash
python "eda_and_dataset_downloads/Download Datasets/create_valid_scene.py"
```

This script runs the full COLMAP pipeline automatically:
1. Copies images to `../watersplatting_data/<dataset>/<scene>/images_wb/`
2. Extracts SIFT-like features from each image
3. Exhaustive matching to find correspondences between all image pairs (slowest step)
4. Sparse reconstruction to estimate camera poses and triangulate 3D points
5. Splits train/eval (every 8th image held out for evaluation)

**Tips for good COLMAP results:**
- Use sequential video frames with 60%+ overlap between adjacent frames
- The camera must translate, not just rotate — pure rotation prevents triangulation
- 50-200 images per scene is ideal
- Avoid featureless images (empty sand, murky water) — too few matchable features
- If COLMAP produces 0 models, add more overlapping images

### Expected Data Format

```
watersplatting_data/
└── <dataset_name>/
    └── <scene_name>/
        ├── images_wb/              White-balanced RGB images
        │   ├── frame_000001.png
        │   └── ...
        └── sparse/
            └── 0/                  COLMAP sparse reconstruction
                ├── cameras.bin     Camera intrinsic parameters
                ├── images.bin      Camera poses per image
                └── points3D.bin    Sparse 3D point cloud
```

---

## Training

```bash
conda activate water_splatting
ns-train water-splatting --data <path_to_scene_dir> --output-dir outputs/
```

Two method variants are available:
- `water-splatting` — standard configuration (15,000 steps)
- `water-splatting-big` — larger model for higher quality

### Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_steps` | 15,000 | Total training iterations |
| `warmup_length` | 500 | Steps before Gaussian refinement starts |
| `refine_every` | 100 | Densify and cull Gaussians every N steps |
| `num_downscales` | 2 | Start at 1/4 resolution, scale up during training |
| `background_color` | `"black"` | Background color (black for underwater) |
| `zero_medium` | `False` | Set `True` to disable water medium modeling |

---

## Viewing Results

```bash
ns-viewer --load-config outputs/<run_name>/config.yml
```

The viewer starts a web server at `http://localhost:7007` with an interactive 3D interface where you can orbit the camera and see real-time renders.

### Remote Server Access

If running on a remote machine, set up SSH port forwarding:

```bash
# Single hop
ssh -L 7007:localhost:7007 user@remote-server

# Through a jump host
ssh -L 7007:localhost:7007 -J user@jump-host user@compute-node
```

Then open `http://localhost:7007` in your local browser.

---

## WaterSplatting vs SonarSplat

| | WaterSplatting | SonarSplat |
|---|---|---|
| **Data type** | Optical RGB camera images | Imaging sonar data |
| **Framework** | Nerfstudio plugin | Custom gsplat fork |
| **Medium modeling** | Learned water MLP (scattering + absorption) | Acoustic azimuth streak model |
| **Pose source** | COLMAP sparse reconstruction | Sensor poses from `.pkl` files |
| **Python** | 3.8 | 3.10 |
| **CUDA** | 11.8 | 12.4 |
| **Viewer** | Nerfstudio web GUI (`ns-viewer`) | Custom viser-based viewer |
| **Best for** | Clear-to-moderate visibility | Turbid / low-visibility water |
| **License** | Apache 2.0 | CC BY-NC-SA 4.0 |

Both frameworks can be used on the same environment if you have both optical and sonar data — they produce complementary 3D models.

---

## Key Source Files

| File | Purpose |
|------|---------|
| `water_splatting/water_splatting.py` | Main model: Gaussians + water medium MLP |
| `water_splatting/water_splatting_config.py` | Nerfstudio method configuration |
| `water_splatting/project_gaussians.py` | Gaussian projection to image plane |
| `water_splatting/rasterize.py` | Gaussian rasterization |
| `water_splatting/sh.py` | Spherical harmonics for view-dependent color |
| `water_splatting/cuda/` | Custom CUDA kernels |
| `eda_and_dataset_downloads/Download Datasets/create_valid_scene.py` | COLMAP pipeline for custom images |
| `eda_and_dataset_downloads/Download Datasets/download_seaThruNerf.py` | SeaThru-NeRF dataset downloader |

---

## License

WaterSplatting is licensed under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).
