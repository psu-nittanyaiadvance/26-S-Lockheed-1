# Z-Loss Investigation Checklist

## Status
- **Docker files: COMPLETE** ✓
- **Z Loss Bugs: FIXED** ✓  (Run 4 launched 2026-04-12, PID 200955)
  - Run 3: Z loss constant at -4.065 — now FIXED
  - Run 4: Z loss moving (-2.47→-2.30→-2.43...), depth_scale=1.315 auto-calibrated
  - Two root causes found and fixed (see below)

## Root Causes Found & Fixed

### Bug 1 — GT histogram includes zero-depth pixels (FIXED)
**File**: `gaussian-splatting-with-depth/scene/dataset_readers.py:136-144`

depth.npy stores sonar RANGE values (0 to 5m). Zero pixels = "no echo" (95.5% of pixels). Including them created a spike at bin 0 that the model's Gaussians can never match (they're never at distance 0 from the camera). This drove Z loss to `log(eps) × mean_GT_with_zeros ≈ -4.065` — constant.

**Fix**: Filter `strip[strip > 0]` before `np.histogram()`. Also added safe division (replace zero row max with 1.0 to avoid NaN).

### Bug 2 — depth_scale 20× too large (FIXED)
**File**: `gaussian-splatting-with-depth/train_v2.py:235`

Old formula: `depth_scale = sonar_max_range / cameras_extent = 5.0 / 0.198 = 25.24`

`cameras_extent` (= 0.198) is the **spread** of camera X/Y positions — NOT the camera-to-scene distance. AONeuS cameras are at z=-2.25 world units from the scene, but cameras_extent only captures the ±0.18 wu spread in X. So depth_scale = 25.24 mapped Gaussians at 2.25 wu → 56.8m → clamped to bin 199. Zero overlap with GT bins 60-116.

**Fix**: Auto-calibrate depth_scale from:
- `median_nonzero_depth_m` from depth.npy (= 2.961m) 
- `avg_cam_dist_wu` from camera_center positions (= 2.252 wu)
- `depth_scale = 2.961 / 2.252 = 1.315`

Also added `--depth_scale` CLI argument for explicit override if needed.

**Result**: Gaussians at avg range 2.25wu → range_m=2.96m → bin 74. GT peak: bin 72 (2.88m). ✓

---

## Files to Examine

### 1. GT z_density computation (dataset_readers.py)
- **Location**: `gaussian-splatting-with-depth/scene/dataset_readers.py:134-144`
- **For Colmap/AONeuS data**:
  ```python
  bin_edges = np.linspace(0, 8, num=201)  # Line 134
  hist_h[i], _ = np.histogram(depth[i * h_res_window: (i + 1) * h_res_window], bins=bin_edges)
  hist_h = hist_h / hist_h.max(axis=1, keepdims=True)
  hist_h = hist_h.T  # Shape: (200, h_res) — 200 bins, h_res height strips
  ```
- **Key facts**:
  - `bin_edges = np.linspace(0, 8, num=201)` → bins cover 0–8 metres (200 bins × 0.04m/bin)
  - `depth` shape: (H, W) — from `np.load(depth_folder/image_name.npy)`
  - Normalized per height-strip: `hist_h = hist_h / hist_h.max(axis=1, keepdims=True)`
  - Final shape after transpose: `(200, h_res)` — **200 depth bins, then h_res height strips**

### 2. Computed z_density (train_v2.py)
- **Location**: `gaussian-splatting-with-depth/train_v2.py`, function `compute_z_density_diff()`
- **Binning formula** (line ~60-75 in compute_z_density_diff):
  ```python
  z_cam = xyz_cam[:, 2].clamp(min=1e-4)
  z_phys = z_cam * depth_scale  # depth_scale = 38.013
  bin_float = z_phys / sonar_max_range * n_depth_bins  # sonar_max_range=5.0, n_depth_bins=200
  ```
  - This places physical depth into bins: `bin_idx = (z_metres / 5.0) * 200`
  - Range: 0–5 metres (not 0–8 like GT!)

### 3. What needs investigation

**CRITICAL QUESTIONS**:

1. **Bin edge mismatch**: 
   - GT uses `np.linspace(0, 8, 201)` — bins 0–8 metres
   - Computed uses `z_phys / 5.0 * 200` — bins 0–5 metres
   - **Action**: Check what `sonar_max_range` should actually be for AONeuS. Is it 5m or 8m?
   - **Where to check**: `run_aoneus_v2.sh` line 64 has `--sonar_max_range 5.0`

2. **Depth scale correctness**:
   - `depth_scale = scene.cameras_extent / sonar_max_range` (computed in train_v2.py startup)
   - For AONeuS: `depth_scale = 0.132 / 5.0 = 0.0264` (actually the inverse: `5.0 / 0.132 = 37.88`)
   - **Action**: Verify the formula. Is it `scene.cameras_extent / sonar_max_range` or the inverse?

3. **Depth units in .npy files**:
   - Are the values in `depth/*.npy` already in metres, or world units?
   - If they're world units, the GT histogram is computing `world_units / 8.0` as the bin index, not metres!
   - **Action**: Load one depth.npy from AONeuS and check: `min(depth)`, `max(depth)`, `mean(depth)`, `dtype`
   - **File path**: `/media/priyanshu/2TB\ SSD/aoneus_dataset/transformed_data/depth/*.npy`

4. **World-to-camera transform convention**:
   - The code uses `world_view_transform` from the camera object
   - Is this column-major (OpenGL, 3DGS convention) or row-major?
   - Wrong convention → projected depth would be negative or inverted
   - **Action**: Print the world_view_transform for one camera and verify xyz_cam has positive z values

5. **Gaussian distribution check**:
   - With depth_scale=38 and sonar_max_range=5.0, Gaussians with z_cam in [0.13, 0.27] map to bins [0, 200]
   - **Action**: In train_v2.py, after `compute_z_density_diff()` call, print:
     - `z_phys.min(), z_phys.max()`
     - `bin_float.min(), bin_float.max()`
     - Compare to GT histogram peak bins (use `np.argmax(z_density_h)` for each height strip)

## Next Steps for New Session

1. **Load one AONeuS depth.npy and inspect it** (5 min)
   ```python
   depth = np.load("/media/priyanshu/2TB\ SSD/aoneus_dataset/transformed_data/depth/[image_name].npy")
   print(f"Shape: {depth.shape}, dtype: {depth.dtype}, min: {depth.min()}, max: {depth.max()}, mean: {depth.mean()}")
   ```

2. **Check sonar_max_range value** — should it be 5.0 or 8.0? (2 min)
   - Look for any comments in train_v2.py or run_aoneus_v2.sh about sonar geometry

3. **Add debug prints to train_v2.py** (5 min)
   - In `compute_z_density_diff()`, after binning, print z_phys range and bin_float range
   - In training loop after Z loss, print `z_density_h.argmax(dim=0)` to see which bins are populated

4. **Compare GT vs computed histograms** (5 min)
   - At step 0, before any training, print both and compare shapes and peak locations
   - If they're in completely different bins, that's the root cause

5. **Fix depth_scale or bin_edges** (once cause is identified)
   - Either: change `sonar_max_range` from 5.0 to 8.0, OR
   - Change GT bin_edges from `linspace(0, 8, 201)` to `linspace(0, 5, 201)`, OR
   - Fix the depth_scale formula if it's inverted

## Test Script Template

```bash
#!/bin/bash
# Quick debug script to check depth data and GT histograms
cd /home/priyanshu/26-S-Lockheed-1

python3 << 'PYTHON_EOF'
import numpy as np
import os

# Load one depth file
depth_path = "/media/priyanshu/2TB\ SSD/aoneus_dataset/transformed_data/depth/[FIRST_IMAGE_NAME].npy"
if not os.path.exists(depth_path):
    # Find first depth file
    depth_dir = "/media/priyanshu/2TB\ SSD/aoneus_dataset/transformed_data/depth"
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.npy')])
    depth_path = os.path.join(depth_dir, depth_files[0])
    print(f"Using: {depth_files[0]}")

depth = np.load(depth_path)
print(f"Depth shape: {depth.shape}, dtype: {depth.dtype}")
print(f"Depth range: [{depth.min():.4f}, {depth.max():.4f}], mean: {depth.mean():.4f}")

# Check GT histogram binning
bin_edges_8 = np.linspace(0, 8, num=201)
bin_edges_5 = np.linspace(0, 5, num=201)
hist_8, _ = np.histogram(depth.flatten(), bins=bin_edges_8)
hist_5, _ = np.histogram(depth.flatten(), bins=bin_edges_5)
print(f"GT histogram (0-8m): peak at bin {hist_8.argmax()} = {bin_edges_8[hist_8.argmax()]:.3f}m")
print(f"GT histogram (0-5m): peak at bin {hist_5.argmax()} = {bin_edges_5[hist_5.argmax()]:.3f}m")
PYTHON_EOF
```

## Summary for New Session

**Problem**: Z loss stuck at constant -4.065, meaning the predicted depth histogram never aligns with GT.

**Hypothesis**: The GT histogram bins depth into 0–8 metres, but the computed histogram bins into 0–5 metres (or vice versa, or uses wrong scaling). As a result, the gamma NLL loss sees zero overlap and never improves.

**Quick win if confirmed**: Change `sonar_max_range` from 5.0 to 8.0 (or vice versa based on findings) and re-run.

**Files already prepared**:
- Docker setup complete (4 files created)
- Z loss code in place (compute_z_density_diff in train_v2.py)
- run_aoneus_v2.sh ready to use

---
*Session ended due to token budget. See MEMORY.md for broader context.*
