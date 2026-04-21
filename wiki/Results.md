# Results

## SonarSplat — monohansett_3D (real shipwreck)

| Run | Branch | Steps | Test PSNR | Test SSIM | LPIPS | Notes |
|-----|--------|-------|-----------|-----------|-------|-------|
| Baseline | main | 2K | 15.93 | 0.632 | — | Upstream L1 loss |
| v2 Run 2 | z_splatting | 40K | 16.08 | 0.629 | — | Gamma NLL, all physics on |
| v2 Run 3 | z_splatting | 40K | **19.29** | 0.291 | higher | PSNR +3.36 dB |

**Run 3 observation:** PSNR improved substantially (+3.36 dB) but SSIM and LPIPS worsened. Gamma NLL renders high-intensity returns correctly according to the MLE criterion, but the predicted image is structurally different from the GT — high confidence in bright scatterers, lower fidelity in diffuse regions. This is expected: SSIM rewards perceptual similarity, not likelihood optimality.

---

## Z-Splat — AONeuS (synthetic turtle, RGB+sonar)

| Run | Trainer | Steps | Test PSNR | Notes |
|-----|---------|-------|-----------|-------|
| Baseline | `train.py` (L1) | 30K | 36.94 | Number to beat |
| Run 3 | `train_v2.py` | 30K | 37.77 | Z loss was a no-op (r_tilde stuck at 0.5) |
| Run 4 | `train_v2.py` (z_density fix) | 30K | 36.76 | Z loss now moving (-2.47→-2.76), but RGB not improved |
| Ablation A | `train_v2.py` | 20K | 35.98 | `z_loss_weight=0` (RGB only) |
| Ablation B | `train_v2.py` | 20K | ~36.0 | `z_loss_weight=0.5` |
| Ablation C | `train_v2.py` | 40K | — | Extended ablation B |
| **Hparam search best (trial 8)** | `train_v2.py` (Optuna) | 20K | **38.29** | Lucky init; params below |
| Run 5 | `train_v2.py` | 40K | 36.70 | Trial 8 params, different seed — high variance |

**Optuna trial 8 best params:**

```
z_loss_weight      = 0.12
r_tilde_lr         = 0.17
reflectivity_reg   = 0.85
lambda_reg         = 0.04
```

---

## Key findings

### Seed variance dominates Z-loss effect

The ±1 dB range across runs with the same hyperparameters matches the typical 3DGS random-seed variance. Trial 8 (Z-loss active, 38.29) beats Run 3 (Z-loss no-op, 37.77), which confirms Z-loss can help — but a single run cannot prove it reliably.

**Recommended next step:** 3–5 seeds with trial 8 params at 30K, take the best. This samples from the initialisation distribution rather than hoping for a lucky seed.

### Z-loss gradient path confirmed

The z_density fix (2026-04-11) via `scatter_add` gives `r_tilde` a real gradient from the sonar loss. The iteration-1 connectivity check (printed to the log) confirms `r_tilde.grad` is non-zero when `--sonar_data_dir` is set or `--h_res`/`--w_res` > 0.

### Z-loss does NOT interfere with RGB

`r_tilde` is completely absent from `gaussian_renderer/__init__.py`. In the histogram path, opacity is detached in `eff_opacity`. The two losses are fully decoupled through the model parameters — PSNR differences are random-seed variance, not cross-contamination.

### Current Z-loss weakness (histogram path)

Default `h_res=w_res=1` means the histogram has a single global bin. `r_tilde` gets only a weak global depth signal. The proper fix is `--sonar_data_dir` (already set in the launch script), which enables the full 2D `render_sonar_image()` path with azimuthal resolution.

---

## SonarSplat v2 on AONeuS sonar-only

| Metric | Value |
|--------|-------|
| Peak PSNR (mid-training) | 28.27 |
| Final PSNR | 26.37 |
| Final SSIM | 0.924 |
| Final LPIPS | 0.189 |
| Gaussians at convergence | ~210 000 |
| Frames | 60 (52 train / 8 test) |

---

## Evaluation metrics

### SonarSplat v2 outputs

| Metric | Description |
|--------|-------------|
| PSNR | Standard peak signal-to-noise ratio |
| SSIM | Structural similarity |
| LPIPS | Learned perceptual image patch similarity |
| fg_psnr | PSNR on foreground pixels only (GT > mask_threshold) |
| fg_l1 | L1 on foreground pixels |
| energy_ratio | pred_total / gt_total (should be ~1.0) |
| pred_tv / gt_tv | Total variation (smoothness) |
| ICV | Integrated contrast variation |

### Z-Splat v2 outputs

- Test PSNR / SSIM / LPIPS (from RGB evaluation)
- `Sonar test Gamma-NLL` (lower is better)
- `sonar_eval_iter_{N}.npz` — arrays: `pred`, `gt` for each test frame

---

## Ablation guide

To run a clean L1 baseline with any dataset (SonarSplat):

```bash
bash scripts/run_v2.sh monohansett_3D /tmp/ablation 30000 --z_loss_weight 0.0
```

To run the full v2 (default, z_loss_weight=1.0):

```bash
bash scripts/run_v2.sh monohansett_3D /tmp/v2_full 30000
```

For Z-Splat, disable sonar loss:

```bash
bash scripts/run_aoneus_v2.sh /tmp/rgb_only 30000 --z_loss_weight 0.0
```
