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

# Z-Splat v2 — AONeuS RGB+sonar
cd z_splatting
bash scripts/run_aoneus_v2.sh "/media/priyanshu/2TB SSD/results" 30000
```

Both scripts accept extra flags at the end to override any hyperparameter:

```bash
# L1 baseline ablation (disable all physics losses)
bash scripts/run_v2.sh monohansett_3D /tmp/test 5000 --z_loss_weight 0.0

# Z-Splat with custom loss weights
bash scripts/run_aoneus_v2.sh /tmp/test 30000 --z_loss_weight 0.5 --camera_loss_weight 0.3
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
│       └── run_aoneus_v2.sh             # AONeuS launcher (steps + passthrough args)
├── Download Datasets/
│   ├── create_valid_z_splat_scene.py    # AONeuS → COLMAP converter (one-time)
│   └── convert_sunboat.py              # Sunboat dataset converter
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
