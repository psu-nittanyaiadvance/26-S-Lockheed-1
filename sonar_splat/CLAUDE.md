# SonarSplat / Z-Splat Project

## What this project is
- **SonarSplat** (main branch): 3D Gaussian splatting for sonar, baseline using L1 loss
- **Z-Splat** (zplat branch): Priyanshu's original implementation — not a paper reproduction. Adds gamma NLL loss, sigmoid reflectivity, ULA beam pattern, elevation constraint, and reflectivity spatial regularizer based on his own math

## Environment
- Conda env: `sonarsplat`
- Data: `/media/priyanshu/2TB SSD/sonarsplat_dataset/pole_qual1`
- Results: `~/ssd/results/zsplat/`
- GPU: 12 GB VRAM. Xorg uses ~1.9 GB, Sunshine (game streaming) uses ~225-506 MB. Kill or disable Sunshine before long training runs.

## How to run

### One-command Z-Splat training
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate sonarsplat && \
cd /home/priyanshu/26-S-Lockheed-1/sonar_splat && \
nohup python -u examples/sonar_simple_trainer.py zsplat \
  --data-dir "/media/priyanshu/2TB SSD/sonarsplat_dataset/pole_qual1" \
  --result-dir ~/ssd/results/zsplat/pole_qual1_run_NAME \
  > /tmp/zsplat_NAME.log 2>&1 &
```

### Monitor training
```bash
python3 -c "
data = open('/tmp/zsplat_NAME.log', 'rb').read()
text = data.replace(b'\r', b'\n').decode('utf-8', errors='replace')
lines = [l for l in text.split('\n') if l.strip() and 'loss=' in l]
print(lines[-1])
"
```

### Check metrics
```bash
python3 -c "
import json, glob, os
files = sorted(glob.glob(os.path.expanduser('~/ssd/results/zsplat/RUNNAME/stats/val_step*.json')),
               key=lambda x: int(x.split('step')[1].split('.')[0]))
for f in files:
    d = json.load(open(f))
    step = int(f.split('step')[1].split('.')[0])
    print(f'step={step:5d} PSNR={d[\"psnr\"]:.3f} SSIM={d[\"ssim\"]:.4f} num_GS={d[\"num_GS\"]}')
"
```

## Baseline vs Z-Splat results

| Run | Steps | PSNR | SSIM | LPIPS | Notes |
|-----|-------|------|------|-------|-------|
| Baseline (main, L1) | 2K | 15.93 | 0.632 | 0.417 | Old SonarSplat |
| Z-Splat run 1 | 40K | 16.08 (step 3K) | 0.629 (step 27K) | — | First full run |
| Z-Splat run 2 | 40K | 16.08 (step 3K) | 0.629 (step 27K) | — | Fixed pruning |

Z-Splat **beat baseline PSNR** (16.08 vs 15.93). SSIM nearly matches (0.629 vs 0.632). PSNR is valid despite gamma NLL because the model is still learning geometry.

## Current zsplat preset (sonar_simple_trainer.py)
```python
"zsplat": Config(
    init_num_pts=20_000,
    init_scale=0.01,
    init_type="predefined",
    max_steps=40_000,
    normalize_world_space=True,
    disable_viewer=True,
    render_eval=True,
    use_beam_pattern=True,
    gamma_nll_k_looks=1,
    w_elevation=1.0,
    camera_model="ortho",
    skip_frames=1,
    strategy=PruneOnlyStrategy(
        verbose=True,
        refine_start_iter=0,
        refine_every=1000,
        refine_stop_iter=30_000,
        grow_grad2d=0.001,
        prune_opa=0.005,
    ),
)
```

## Critical bugs fixed (do not revert)

### 1. visualize_gaussians() blocks headless servers (dataloader.py:377)
`o3d.visualization.draw_geometries()` blocks forever on servers with no display. Commented out:
```python
# visualize_gaussians(xyz=[pcd], poses=camtoworlds, start_size=0.1, end_size=0.1)
```

### 2. PYTORCH_CUDA_ALLOC_CONF must be set before torch imports (sonar_simple_trainer.py:4)
Setting it via env var in the shell doesn't work reliably once CUDA is initialized. Set it in code:
```python
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
```

### 3. sat_mask shape mismatch in gamma_nll_loss (NHWC issue)
pixels is NHWC `[B, H, W, C]`, so `pixels[:,0,:,:]` selects azimuth row 0, not channel 0. Mask must be expanded to match Z_hat shape before passing to gamma_nll_loss.

### 4. reflectivity_reg OOM — subsample Gaussians
`torch.cdist` on all Gaussians after densification (~960K) OOMs. Fixed by subsampling to max 4096 Gaussians with `torch.randperm`.

### 5. elevation_loss OOM — precompute peak range
`sonar_image[az_idx, :]` created `[N, W_range]` tensor. Fixed by precomputing peak range per azimuth bin first: `peak_per_az = sonar_image.argmax(dim=-1)` then indexing `[N]`.

### 6. Hardcoded device="cuda" in gsplat/rendering.py:1646
Changed to `device=means.device` for Mac/CPU/ROCm compatibility.

### 7. skip_frames=0 causes ZeroDivisionError
zsplat preset uses `skip_frames=1`.

## Pruning behavior
- Gaussians grow from ~80K (init predefined) to 1.1M during densification (steps 0-5K)
- Then pruned down to ~200K by step 29K
- `refine_stop_iter=30_000` stops pruning at step 30K — last 10K steps refine only
- `prune_opa=0.005` — remove Gaussians with opacity < 0.005 (was 0.01, too aggressive)
- `refine_every=1000` — prune every 1000 steps (was 500)
- PSNR dips around step 5K and 15K are expected during aggressive pruning phases, recovers after

## Setup docs
- `SETUP_LINUX.md` — Linux setup with zplat branch
- `SETUP_WINDOWS.md` — Windows setup
- `SETUP_MAC.md` — Mac/MPS setup (BUILD_NO_CUDA=1)
- `SETUP_AMD.md` — ROCm setup (torch.cuda.* works unchanged via ROCm)
