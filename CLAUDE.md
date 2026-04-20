# Project Context: 26-S-Lockheed-1

## What this repo is

A research project combining two Gaussian splatting models for underwater sonar data:

1. **SonarSplat** (`sonar_splat/`) — the upstream paper baseline (IEEE RA-L 2025, U-Michigan). Novel view synthesis of imaging sonar via 3D Gaussian splatting.
2. **Z-Splat** (`gaussian-splatting-with-depth/`) — the Z-Axis Gaussian Splatting model (IEEE TPAMI 2024). Fuses RGB camera + sonar depth via a modified 3DGS rasterizer.

Priyanshu's original work lives on the `z_splatting` branch. The main training scripts are `sonar_simple_trainer_v2.py` (SonarSplat v2) and `train_v2.py` (Z-Splat v2).

Key additions over the SonarSplat baseline:
- Gamma NLL loss (replaces L1) — models sonar speckle noise statistically
- Sigmoid reflectivity `r_tilde` per Gaussian
- ULA beam pattern weighting via `torch.sinc` formulation
- Elevation constraint loss (annealed: w_e=1.0 → 0.1 over 10K steps)
- Reflectivity spatial regularizer (kNN, 4096 subsample every 100 steps)
- `sigma_r = speed_of_sound / (4 * bandwidth) = 1.25 cm`
- Differentiable z_density via `scatter_add` (implemented 2026-04-11) — fixes r_tilde receiving zero gradient from Z loss

This is **novel research**, not a paper reproduction. Benchmark against SonarSplat baseline (main branch, L1 loss).

## Directory structure

```
26-S-Lockheed-1/
├── sonar_splat/                            # SonarSplat model
│   ├── examples/
│   │   ├── sonar_simple_trainer.py         # upstream baseline trainer
│   │   └── sonar_simple_trainer_v2.py      # Priyanshu's v2 trainer (gamma NLL, r_tilde, etc.)
│   └── scripts/
│       ├── run_3D_monohansett.sh           # baseline training command
│       ├── run_3D_monohansett_v2.sh        # v2 training command (sonar_simple_trainer_v2.py)
│       └── run_nvs_infra_360_1.sh
├── gaussian-splatting-with-depth/          # Z-Splat model
│   ├── train.py                            # upstream Z-Splat trainer
│   ├── train_v2.py                         # Priyanshu's v2 trainer (z_density, r_tilde, etc.)
│   ├── scripts/
│   │   └── run_aoneus_v2.sh               # Z-Splat v2 training command
│   ├── gaussian_renderer/__init__.py       # patched: handles 2 or 4 rasterizer outputs
│   ├── scene/dataset_readers.py           # patched: scipy.tukey fix + zero-depth filter
│   └── submodules/
│       ├── diff-gaussian-rasterization/   # cloned + patched (cstdint fix for CUDA 12+)
│       └── simple-knn/
├── Download Datasets/
│   └── create_valid_z_splat_scene.py      # converts AONeuS data → COLMAP format
└── logs/                                  # all training logs here
```

## Datasets (on 2TB SSD at `/media/priyanshu/2TB SSD/`)

### SonarSplat dataset (`sonarsplat_dataset/`) — sonar pkl format, ready to use
| Scene | PKL files | Has GT mesh |
|---|---|---|
| `monohansett_3D` | 1228 | yes — real-world shipwreck |
| `concrete_piling_3D` | 687 | yes |
| `infra_360_1` | 336 | no |
| `basin_horizontal_infra_1` | 171 | no |
| `rock_semicircle1` | 161 | no |
| `basin_horizontal_empty1` | 163 | no |
| `basin_horizontal_piling_up_down_4` | 111 | no |
| `basin_horizontal_piling_1` | 101 | no |
| `pole_qual1` | 29 | no |

### AONeuS dataset (`aoneus_dataset/`) — for Z-Splat (RGB+sonar)
- `data/reduced_baseline_0.6x_rgb/` — 60 RGB images + `cameras_sphere.npz`
- `data/reduced_baseline_0.6x_sonar/` — 60 pkl sonar files + `Config.json`
- `transformed_data/` — COLMAP format output (images/, sparse/0/, depth/*.npy)
- Synthetic turtle scene — the only public AONeuS data released
- Sonar max range: **5.0 m**, `depth_scale ≈ 1.315` (auto-calibrated: median depth / avg cam dist)

## Environment

- **Conda env**: `sonarsplat` — use for everything. `lockheed1`/base do NOT have gsplat.
- **Python**: 3.10, **PyTorch**: 2.6.0+cu124, **CUDA**: 12.4
- **GPU**: single GPU (12 GB VRAM). Sunshine game streaming (~500 MB) + Xorg (~1.9 GB) eat into this — kill Sunshine before long runs.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** is set at the top of `sonar_simple_trainer_v2.py` — do not remove it.

## Benchmark results

### SonarSplat (real sonar, monohansett_3D)
| Run | Steps | Test PSNR | Test SSIM | Notes |
|---|---|---|---|---|
| Baseline (main, L1) | 2K | 15.93 | 0.632 | upstream model |
| Run 2 (v2, gamma NLL) | 40K | 16.08 | 0.629 | beat PSNR |
| Run 3 (v2, gamma NLL) | 40K | **19.29** | 0.291 | PSNR +3.36 dB; SSIM/LPIPS worse (gamma renders high-intensity but structurally different) |

### Z-Splat (AONeuS RGB+sonar)
| Run | Steps | Test PSNR | Notes |
|---|---|---|---|
| Baseline (`train.py`, L1 + z_density=zeros) | 30K | 36.94 | number to beat |
| Run 3 (train_v2.py, before z_density fix) | 30K | **37.77** | +0.83 dB — r_tilde stuck at 0.5, Z loss was no-op |
| **Run 4** (train_v2.py, z_density fix) | 30K | **36.76** | -0.18 dB — Z loss moving (-2.47→-2.76), but RGB not improved yet |
| Ablation A (z_loss_weight=0) | 20K | 35.98 | RGB-only, Z loss disabled |
| Ablation B (z_loss_weight=0.5) | 20K | ~36.0 | Z loss active but too strong |
| Ablation C (z_loss_weight=0.5) | 40K | — | extended ablation B |
| Hparam search best (trial 8, 20K) | 20K | **38.29** | Optuna TPE, 15 trials — lucky init, params: z_loss_wt=0.12, r_tilde_lr=0.17, refl_reg=0.85, λ_reg=0.04 |
| **Run 5** (Optuna best params, 40K) | 40K | **36.70** | Same params as trial 8 but different random init — confirms high variance across seeds |

**Key finding — seed variance, not Z-loss interference**: `r_tilde` is completely absent from `gaussian_renderer/__init__.py`. Z-loss gradient path is `ZL → z_density → scatter_add → eff_opacity → r_tilde` with `opacity.detach()` — it cannot touch xyz/opacity/SH used in RGB rendering. The elevation constraint is a no-op without `--sonar_data_dir` (r_sonar = ‖xyz.detach()‖, loss ≈ 0). PSNR differences across runs are pure random-seed variance in 3DGS initialisation, not caused by Z-loss. Trial 8 (Z-loss active) at 38.29 beats Run 3 (Z-loss no-op) at 37.77 — confirming Z-loss can help.

**Current Z-loss weakness**: `h_res=w_res=1` (default). The histogram has no spatial resolution — just one global bin comparing "all Gaussians' ranges" vs "all sonar ranges". r_tilde has almost nothing to learn from it. To make the sonar loss meaningful, either: (a) pass `--sonar_data_dir` to use the full 2D az×range intensity image (proper spec formulation, needs gsplat `_sonar_rasterization`), or (b) increase `--h_res`/`--w_res` to 4+ strips for spatial depth supervision.

**Recommended next experiments**:
1. Multi-seed run: 3–5 seeds with trial 8 params at 30K, take best (reliable way to sample a good init)
2. Proper sonar image supervision: pass `--sonar_data_dir` (uses `_sonar_rasterization`, already confirmed available) — 2D intensity comparison gives r_tilde real gradient signal with azimuthal resolution

### SonarSplat v2 on AONeuS sonar-only
- Peak PSNR: **28.27** (mid-training), final: 26.37, SSIM=0.924, LPIPS=0.189
- 210K Gaussians at convergence, 60 frames (52 train / 8 test)

## How to run

### SonarSplat v2 (real sonar, monohansett_3D)
```bash
conda activate sonarsplat
cd sonar_splat
nohup bash scripts/run_3D_monohansett_v2.sh \
  "/media/priyanshu/2TB SSD/sonarsplat_dataset/monohansett_3D" \
  "/media/priyanshu/2TB SSD/results" \
  > ../logs/run_v2_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Z-Splat v2 (AONeuS RGB+sonar)
```bash
conda activate sonarsplat
cd gaussian-splatting-with-depth
nohup bash scripts/run_aoneus_v2.sh \
  > ../logs/aoneus_v2_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

Key args for Z-Splat v2:
- `--sonar_max_range 5.0` (AONeuS sonar range)
- `--camera_loss_weight 0.3` (w_c; sonar is primary at implicit weight 1.0)
- `--sonar_data_dir` already set in script (enables full sonar render path; `SONAR_PATH=full_render` printed at start)
- `--depth_scale` auto-calibrated (printed at start, ≈1.315 for AONeuS)
- Startup prints: `SONAR_PATH=`, loss weights, sigma_r, depth_scale
- Eval prints every checkpoint: `Loss breakdown | ZL=  L_cam=  elev=  reg=  total=`
- Eval prints: `Sonar test Gamma-NLL:` + saves `sonar_eval_iter_{N}.npz` (pred + gt arrays)
- Gradient norm print every 500 iters: `∇norms: xyz=  r_tilde=  opacity=  f_dc=`

### Convert AONeuS data to COLMAP format (one-time)
```bash
conda activate sonarsplat
cd "Download Datasets"
python create_valid_z_splat_scene.py
```

### Upstream Z-Splat (baseline)
```bash
conda activate sonarsplat
cd gaussian-splatting-with-depth
python train.py \
  -s "/media/priyanshu/2TB SSD/aoneus_dataset/transformed_data" \
  -m "/media/priyanshu/2TB SSD/aoneus_dataset/outputs/aoneus" \
  --depth_loss --eval -i images
```
Note: quote the path — it has a space. Run as one line.

## Differentiable z_density (implemented 2026-04-11)

The standard `diff-gaussian-rasterization` only returns `(color, radii)`. Previous code read `render_pkg["z_density_h"]` which was always `torch.zeros()` — r_tilde received zero gradient from Z loss.

**Fix**: `compute_z_density_diff()` in `train_v2.py`:
1. Project Gaussian centers to camera space (geometry detached)
2. Compute Euclidean range × `depth_scale` → physical metres
3. Soft histogram: linear interpolation between floor/ceil bins
4. `scatter_add` accumulates `eff_opacity = opacity × r_tilde × beam` into bins
5. Reshape and normalise per strip

Gradient path: `ZL → z_density → scatter_add → eff_opacity → r_tilde` ✓

**Important**: opacity is `.detach()` in `eff_opacity` — Z loss cannot influence RGB rendering parameters. `r_tilde` itself does not appear in `gaussian_renderer/__init__.py` (confirmed). The two losses are fully decoupled.

**Limitation of histogram path**: default `h_res=w_res=1` means the histogram has zero spatial resolution (one global bin). r_tilde gets only a weak global depth signal. The proper fix is `--sonar_data_dir` which triggers the 2D `render_sonar_image()` path using gsplat's `_sonar_rasterization` — this provides full az×range supervision and is confirmed available in the `sonarsplat` env.

## Z-Splat two sonar loss paths

`train_v2.py` has two sonar loss modes, selected automatically:

| Condition | Path | What's compared | Spatial res |
|---|---|---|---|
| `--sonar_data_dir` given + gsplat available | `render_sonar_image()` → gamma NLL | full 2D rendered sonar (az×range) vs GT pkl image | full (96az × 256rng) |
| No `--sonar_data_dir` | `compute_z_density_diff()` → gamma NLL | 1D depth histogram vs GT from depth.npy | h_res × w_res strips (default 1×1) |

**`render_sonar_image()` opacity**: `get_opacity()` is **NOT** detached (reverted 2026-04-16). Both `render_sonar_image()` and the histogram fallback `eff_opacity` allow ∇L_sonar to reach both `opacity` and `r_tilde`. Detaching opacity was confirmed to prevent r_tilde from learning when RGB-trained opacity was near zero for acoustically relevant Gaussians — the sonar loss needs to adjust opacity as well as reflectivity.

## Known bugs fixed — do not reintroduce

- **`visualize_gaussians()` hangs** (`dataloader.py` ~line 383): `o3d.visualization.draw_geometries()` blocks on headless servers forever. Kept commented out.
- **`sat_mask` shape mismatch**: sonar pixels are NHWC `[B,H,W,C]`. `pixels[:,0,:,:]` selects azimuth row 0, not channel. Must expand mask to match `Z_hat`.
- **`reflectivity_reg` OOM**: `torch.cdist` on all Gaussians post-densification. Subsampled to max 4096 with `torch.randperm`.
- **`elevation_loss` OOM**: `sonar_image[az_idx,:]` → `[N, W_range]` tensor. Fixed with `argmax` precompute.
- **Hardcoded `device="cuda"`** in `gsplat/rendering.py:1646` → `device=means.device`.
- **`skip_frames=0` ZeroDivisionError**: use `skip_frames=1`.
- **GT histogram zero-depth spike** (`dataset_readers.py`): depth.npy zeros = "no echo" (95.5% of pixels). Filter `strip[strip > 0]` before histogramming.
- **`depth_scale` 20× too large**: old formula `sonar_max_range / cameras_extent` used X/Y camera spread (0.198) not scene distance. Fixed with auto-calibration: `median_nonzero_depth_m / avg_cam_dist_wu ≈ 1.315`.
- **`elevation_constraint_loss` uses world origin in `train_v2.py`**: was `||xyz||` instead of `||xyz - cam_center||`. Fixed 2026-04-13 — all three uses (r_bin_center, no-op fallback, loss call) now use `viewpoint_cam.camera_center.detach()`.
- **`elevation_constraint_loss` uses world origin in `sonar_simple_trainer_v2.py`**: same bug in the sonar-only trainer. Fixed 2026-04-16 — `r_bin_center` now uses `torch.norm(means - cam_center_w)` where `cam_center_w = camtoworlds[0,:3,3].detach()`. Function signature updated to accept `cam_center` param.
- **NaN gradients in beam pattern backward**: replaced `torch.where` with `torch.sinc`.
- **NaN guards** added after `loss.backward()` and after optimizer step in v2 trainer.
- **`densification_postfix`/`prune_points`**: must pass ALL tensors including `r_tilde` to `cat_tensors_to_optimizer`/`_prune_optimizer` in one shot — do NOT call super().
- **`diff-gaussian-rasterization` CUDA 12+ build**: needs `#include <cstdint>` in `cuda_rasterizer/rasterizer_impl.h`.
- **`scipy.signal.tukey`**: moved to `scipy.signal.windows.tukey` in newer scipy. Fixed with try/except in `scene/dataset_readers.py`.
