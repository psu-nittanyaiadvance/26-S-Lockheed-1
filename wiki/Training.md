# Training

## Environment setup

Always use the `sonarsplat` conda environment. `lockheed1` and `base` do not have gsplat installed.

```bash
conda activate sonarsplat
```

Kill Sunshine (game-streaming service) before long runs — it occupies ~500 MB of VRAM:

```bash
sudo systemctl stop sunshine   # or: pkill sunshine
```

Both v2 trainers set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` automatically.

---

## SonarSplat v2 — general launcher

The general launcher `sonar_splat/scripts/run_v2.sh` accepts any known dataset name, any results directory, and any step count. All remaining arguments are passed through verbatim to the trainer.

```
Usage: bash scripts/run_v2.sh <dataset> <results_dir> [steps] [extra args...]
```

**Positional arguments:**

| Argument | Description |
|----------|-------------|
| `dataset` | Dataset name (see table below) or full absolute path |
| `results_dir` | Base output directory (timestamped subdir created automatically) |
| `steps` | Training steps. Default: 30000 |
| `extra args` | Any `--flag value` pairs, appended to the trainer call |

**Known dataset names:**

| Name | Data path | far/near | energy_wt |
|------|-----------|----------|-----------|
| `monohansett_3D` | `sonarsplat_dataset/monohansett_3D` | 10 / -10 | 0.1 |
| `concrete_piling_3D` | `sonarsplat_dataset/concrete_piling_3D` | 10 / -10 | 0.1 |
| `infra_360_1` | `sonarsplat_dataset/infra_360_1` | 5 / -5 | 0.0 |
| `basin_horizontal_infra_1` | `sonarsplat_dataset/basin_horizontal_infra_1` | 5 / -5 | 0.0 |
| `rock_semicircle1` | `sonarsplat_dataset/rock_semicircle1` | 5 / -5 | 0.0 |
| `basin_horizontal_empty1` | `sonarsplat_dataset/basin_horizontal_empty1` | 5 / -5 | 0.0 |
| `basin_horizontal_piling_up_down_4` | `sonarsplat_dataset/basin_horizontal_piling_up_down_4` | 5 / -5 | 0.0 |
| `basin_horizontal_piling_1` | `sonarsplat_dataset/basin_horizontal_piling_1` | 5 / -5 | 0.0 |
| `pole_qual1` | `sonarsplat_dataset/pole_qual1` | 5 / -5 | 0.0 |
| `aoneus_sonar` | `aoneus_dataset/data/reduced_baseline_0.6x_sonar` | 5 / -5 | 0.0 |

**Examples:**

```bash
cd sonar_splat

# Standard 30K run on monohansett
bash scripts/run_v2.sh monohansett_3D "/media/priyanshu/2TB SSD/results" 30000

# 40K run on monohansett with nohup logging
nohup bash scripts/run_v2.sh monohansett_3D "/media/priyanshu/2TB SSD/results" 40000 \
  > ../logs/mono_v2_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# L1 baseline ablation (disable all physics losses)
bash scripts/run_v2.sh monohansett_3D "/media/priyanshu/2TB SSD/results" 30000 --z_loss_weight 0.0

# Quick smoke test (2K steps)
bash scripts/run_v2.sh infra_360_1 /tmp/test 2000

# Custom dataset path
bash scripts/run_v2.sh /my/custom/dataset "/media/priyanshu/2TB SSD/results" 30000
```

---

## Z-Splat v2 — AONeuS

```
Usage: bash scripts/run_aoneus_v2.sh <results_dir> [steps] [extra args...]
```

```bash
cd z_splatting

# Standard 30K run
bash scripts/run_aoneus_v2.sh "/media/priyanshu/2TB SSD/results" 30000

# 40K run with logging
nohup bash scripts/run_aoneus_v2.sh "/media/priyanshu/2TB SSD/results" 40000 \
  > ../logs/aoneus_v2_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Disable Z loss (camera-only ablation)
bash scripts/run_aoneus_v2.sh "/media/priyanshu/2TB SSD/results" 30000 --z_loss_weight 0.0

# Custom loss weights
bash scripts/run_aoneus_v2.sh "/media/priyanshu/2TB SSD/results" 30000 \
  --z_loss_weight 0.5 --camera_loss_weight 0.3
```

Checkpoint iterations are computed automatically as even fractions of whatever step count you pass.

---

## Monitoring training

**Follow a log file live:**

```bash
tail -f logs/<filename>.log
```

**Key log lines to watch:**

```
SONAR_PATH=full_render          # confirms sonar render path active (Z-Splat v2)
depth_scale = 1.315             # auto-calibrated depth scale (printed once at start)

# Every test checkpoint (Z-Splat v2):
Loss breakdown | ZL= 0.234  L_cam= 1.45  elev= 0.012  reg= 0.089  total= 1.821
Sonar test Gamma-NLL: 0.341    # lower is better

# Every 500 iters (Z-Splat v2):
∇norms: xyz= 0.0023  r_tilde= 0.0071  opacity= 0.0041  f_dc= 0.0019

# Every test step (SonarSplat v2):
Test PSNR: 19.2  SSIM: 0.287  LPIPS: 0.512
```

**Check GPU usage:**

```bash
nvidia-smi -l 1
```

---

## Key hyperparameters

### SonarSplat v2

| Flag | Default | Notes |
|------|---------|-------|
| `--max_steps` | 30000 | Set via positional arg in run_v2.sh |
| `--z_loss_weight` | 1.0 | 0 = L1 baseline, 1 = full v2 physics |
| `--reflectivity_lr` | 0.01 | r_tilde learning rate |
| `--w_e` | 1.0 | Elevation weight start |
| `--w_e_final` | 0.1 | Elevation weight after annealing |
| `--w_e_anneal_steps` | 10000 | Annealing duration |
| `--reflectivity_reg_weight` | 0.1 | Spatial regulariser strength |
| `--lambda_reg` | 0.01 | Mean-reversion vs smoothness |
| `--energy_loss_weight` | 0.0 | Energy preservation (0.1 for monohansett) |

### Z-Splat v2

| Flag | Default | Notes |
|------|---------|-------|
| `--iterations` | 30000 | Set via positional arg in run_aoneus_v2.sh |
| `--z_loss_weight` | 0.12 | Sonar loss weight (Optuna best) |
| `--camera_loss_weight` | 1.0 | RGB photometric loss weight |
| `--r_tilde_lr` | 0.01 | r_tilde learning rate |
| `--sonar_max_range` | 5.0 | AONeuS sonar max range (metres) |
| `--sonar_data_dir` | (set in script) | Enable full 2D sonar render path |
| `--depth_scale` | auto | Override depth_scale calibration |
| `--use_rl_controller` | off | Enable adaptive loss weight control |

---

## Upstream baselines

**SonarSplat baseline (L1):**

```bash
cd sonar_splat
bash scripts/run_3D_monohansett.sh \
  "/media/priyanshu/2TB SSD/sonarsplat_dataset/monohansett_3D" \
  "/media/priyanshu/2TB SSD/results"
```

**Z-Splat baseline (L1 + static depth):**

```bash
cd z_splatting
python train.py \
  -s "/media/priyanshu/2TB SSD/aoneus_dataset/transformed_data" \
  -m "/media/priyanshu/2TB SSD/aoneus_dataset/outputs/aoneus" \
  --depth_loss --eval -i images
```
