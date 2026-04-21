# Architecture

## SonarSplat v2

### File: `sonar_splat/examples/sonar_simple_trainer_v2.py`

~2040 lines. Self-contained: dataset loading, model init, training loop, evaluation.

**Key classes and sections:**

| Location | Purpose |
|----------|---------|
| `Config` dataclass (lines 73–309) | All hyperparameters with defaults |
| `create_splats_with_optimizers` (lines 311–416) | Initialise Gaussians + separate optimiser for `r_tilde` |
| `beam_pattern` (lines 419–443) | ULA array gain per Gaussian per view |
| `energy_loss` (lines 447–455) | Global energy conservation |
| `gamma_nll_loss` (lines 459–465) | MLE loss for speckle noise |
| `elevation_constraint_loss` (lines 469–480) | Range-anchored position regulariser |
| `reflectivity_reg` (lines 484–492) | kNN spatial smoothness + mean reversion |
| `train` method (lines 828–1583) | Full training loop |
| `eval` method (lines 1585–1792) | PSNR/SSIM/LPIPS + foreground metrics |

**Gaussian parameter initialisation:**

```
means     : random or from point cloud
scales    : log(init_scale) per axis
opacities : logit(0.1) ≈ -2.2
quats     : random unit quaternions
SH dc     : RGB2SH(default_colour)
SH high   : zeros
r_tilde   : full(N, sigmoid_inv(0.982)) = 4.0  →  sigmoid = 0.982
```

**Optimiser learning rates (scaled by √(batch × world_size)):**

| Parameter | LR |
|-----------|----|
| means | 1.6e-4 × scene_scale |
| scales | 5e-3 |
| quats | 1e-3 |
| opacities | 5e-2 |
| SH dc | 2.5e-2 |
| SH high | 2.5e-2 / 20 |
| r_tilde | `--reflectivity_lr` (default 0.01) |

**Training phases:**

| Phase | Condition | Effect |
|-------|-----------|--------|
| Normal | Outside streak interval | Full loss, all gradients active |
| Streak (Period B) | In streak interval | Geometry grads zeroed, focus on saturation modeling |
| Post-streak | After `streak_end_step` | Full loss + saturation regularisation |

**Best checkpoint scoring:**

```
score = fg_psnr  -  0.5 × fg_l1  -  0.2 × |log(energy_ratio)|
```

Valid only when `best_ckpt_min_energy_ratio ≤ energy_ratio ≤ best_ckpt_max_energy_ratio`.

---

## Z-Splat v2

### File: `gaussian-splatting-with-depth/train_v2.py`

~956 lines. Uses standard 3DGS codebase classes (`GaussianModelV2`, `Scene`, `Camera`).

**Key functions:**

| Function | Lines | Purpose |
|----------|-------|---------|
| `compute_z_density_diff` | 100–214 | Differentiable depth histogram via scatter_add |
| `SonarDataCache` | 221–279 | Load and match AONeuS sonar PKL files |
| `render_sonar_image` | 282–338 | Full 2D sonar render via `_sonar_rasterization` |
| `training` | 345–796 | Main loop |

**Two sonar loss paths:**

| Condition | Path | Spatial resolution |
|-----------|------|--------------------|
| `--sonar_data_dir` given | `render_sonar_image()` → Gamma NLL | Full 2D (96 az × 256 range) |
| No `--sonar_data_dir` | `compute_z_density_diff()` → Gamma NLL | `h_res × w_res` strips (default 1×1) |

**Differentiable z_density (`compute_z_density_diff`):**

The standard `diff-gaussian-rasterization` only returns `(colour, radii)`. The CUDA z_density output was always zeros, so `r_tilde` received no gradient from the Z loss. Fix:

1. Project Gaussian centres to camera space (geometry detached)
2. Compute Euclidean range × `depth_scale` → metres
3. Linear soft-binning between floor/ceil bins
4. `scatter_add` accumulates `eff_opacity = opacity × r_tilde × beam` into bins
5. Normalise per strip

Gradient flow: `L_sonar → z_density → scatter_add → eff_opacity → r_tilde ✓`

**Opacity detach status (important):**

- In `compute_z_density_diff`: opacity is **detached** — Z loss cannot reach xyz/SH/opacity.
- In `render_sonar_image`: opacity is **NOT detached** (reverted 2026-04-16) — sonar loss can adjust both opacity and `r_tilde`. Detaching blocked learning when RGB-trained opacity near zero for acoustically relevant Gaussians.

**Gradient management:**

r_tilde parameters are clipped independently from RGB parameters:

```python
clip_grad_norm_(rgb_params, max_norm=1.0)
clip_grad_norm_(r_tilde_params, max_norm=0.5)
```

Clipping jointly dropped test PSNR ~2 dB.

**RL loss controller (optional):**

Enabled with `--use_rl_controller`. A PPO-based agent reads the current PSNR and adjusts `z_loss_weight` / `camera_loss_weight` dynamically. Policy saved to `<model_dir>/rl_policy.pt` for cross-run warm-start.

---

## Patched files

### `gaussian_renderer/__init__.py`

The upstream renderer called `rasterizer(...)` and unpacked exactly 2 outputs: `(colour, radii)`. The patched version handles both the old (2-output) and new (4-output) rasterizer:

```python
out = rasterizer(...)
if len(out) == 4:
    colour, radii, z_density_h, z_density_w = out
else:
    colour, radii = out
    z_density_h = torch.zeros(200, h_res)
    z_density_w = torch.zeros(200, w_res)
```

### `scene/dataset_readers.py`

Two patches:

1. **`scipy.signal.tukey` → `scipy.signal.windows.tukey`** (moved in newer scipy): handled with try/except.

2. **Zero-depth filter** in depth histogram extraction: `depth.npy` zeros mean "no echo" (95.5% of pixels in AONeuS sonar). Without filtering, there is a large spurious spike at bin 0 that the model can never match. Fix: `strip = strip[strip > 0]` before histogramming.

### `diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h`

CUDA 12+ removed implicit `<cstdint>` includes. Added `#include <cstdint>` at the top.

---

## GaussianModelV2

`scene/gaussian_model_v2.py` extends the standard `GaussianModel` with:

- `r_tilde` parameter tensor (N, 1), sigmoid-activated
- Extended `densification_postfix` and `prune_points` to include `r_tilde` in `cat_tensors_to_optimizer` / `_prune_optimizer` — **must pass all tensors in one shot**, not via `super()`.

---

## Dataset conversion: AONeuS → COLMAP

`Download Datasets/create_valid_z_splat_scene.py`

One-time conversion pipeline:

1. Parse `cameras_sphere.npz` → COLMAP camera intrinsics + extrinsics
2. Copy RGB images to `transformed_data/images/`
3. Run `pycolmap` feature extraction + matching + triangulation (falls back to random seed points if unavailable)
4. For each sonar PKL: extract per-pixel slant range → save `transformed_data/depth/{stem}.npy`

```bash
conda activate sonarsplat
cd "Download Datasets"
python create_valid_z_splat_scene.py
```

Output: `transformed_data/` with `images/`, `sparse/0/`, and `depth/`.
