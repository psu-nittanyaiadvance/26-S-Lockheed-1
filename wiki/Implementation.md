# Implementation Notes

## Bugs fixed — do not reintroduce

These bugs have been fixed and are documented here so they are not accidentally re-introduced in future refactors.

---

### `visualize_gaussians()` hangs on headless servers

`o3d.visualization.draw_geometries()` blocks forever on headless servers. The call is kept but commented out in `dataloader.py` (~line 383).

---

### `sat_mask` shape mismatch

Sonar pixels are NHWC: `[B, H, W, C]`. The incorrect indexing `pixels[:, 0, :, :]` selects azimuth row 0, not the channel dimension. The mask must be expanded to match `Z_hat`.

---

### `reflectivity_reg` OOM

`torch.cdist` on all Gaussians post-densification (200K+) exceeds VRAM. Fixed by subsampling to at most 4096 Gaussians with `torch.randperm` before computing pairwise distances.

---

### `elevation_loss` OOM

`sonar_image[az_idx, :]` returns an `[N, W_range]` tensor rather than a scalar per Gaussian. Fixed with `argmax` precompute — peak range index extracted before indexing.

---

### Hardcoded `device="cuda"` in gsplat

`gsplat/rendering.py:1646` had `device="cuda"` hardcoded. Changed to `device=means.device` to support any device.

---

### `skip_frames=0` ZeroDivisionError

Stride of 0 causes a division-by-zero in the frame sampler. Use `skip_frames=1` (take every frame with no skipping).

---

### GT histogram zero-depth spike

`depth.npy` in the AONeuS dataset contains exact zeros for "no echo" pixels — 95.5% of all pixels. Without filtering, histogramming creates a massive spike at bin 0 that the model cannot reproduce and that dominates the loss. Fixed in `scene/dataset_readers.py`:

```python
strip = strip[strip > 0]   # exclude no-echo pixels before histogramming
```

---

### `depth_scale` 20× too large

The original formula `sonar_max_range / cameras_extent` used the XY camera spread (≈ 0.198 world units) rather than the actual camera-to-scene distance. This made depth_scale ≈ 25 — far too large. Fixed with auto-calibration:

```python
depth_scale = median(nonzero sonar depth in metres) / mean(camera distance from scene centre)
            ≈ 1.315  for AONeuS
```

---

### `elevation_constraint_loss` uses world origin (both trainers)

**`train_v2.py` (fixed 2026-04-13):** The loss computed `r_predicted = ||xyz||` (distance from world origin). Should be `||xyz - cam_center||`. All three uses (r_bin_center computation, no-op fallback, loss call) now use `viewpoint_cam.camera_center.detach()`.

**`sonar_simple_trainer_v2.py` (fixed 2026-04-16):** Same bug. `r_bin_center` now uses `torch.norm(means - cam_center_w)` where `cam_center_w = camtoworlds[0, :3, 3].detach()`. Function signature updated to accept `cam_center` param.

---

### NaN gradients in beam pattern backward

`torch.where` masking a `sin(x)/x` expression evaluates both branches before masking, so `x=0` still produces NaN in the backward pass. Fixed by replacing with `torch.sinc`, which handles x=0 analytically.

---

### NaN parameter guards

After `loss.backward()` and after the optimizer step, both v2 trainers check for NaN/Inf and reset affected parameters to zero:

```python
for p in [xyz, opacity, r_tilde]:
    if torch.isnan(p.data).any() or torch.isinf(p.data).any():
        p.data[torch.isnan(p.data) | torch.isinf(p.data)] = 0.0
```

---

### `densification_postfix` / `prune_points` must include `r_tilde`

`cat_tensors_to_optimizer` and `_prune_optimizer` must receive ALL tensors (including `r_tilde`) in a single call. Calling `super()` first and then handling `r_tilde` separately corrupts the optimizer state. The fix is to override both functions in `GaussianModelV2` and not call `super()`.

---

### `diff-gaussian-rasterization` build failure on CUDA 12+

CUDA 12+ removed implicit `<cstdint>` includes from device headers. Building the submodule fails with `uint32_t` undeclared. Fixed by adding:

```cpp
#include <cstdint>
```

at the top of `cuda_rasterizer/rasterizer_impl.h`.

---

### `scipy.signal.tukey` → `scipy.signal.windows.tukey`

`scipy.signal.tukey` was moved to `scipy.signal.windows.tukey` in newer scipy versions. Fixed in `scene/dataset_readers.py` with:

```python
try:
    from scipy.signal import tukey
except ImportError:
    from scipy.signal.windows import tukey
```

---

## Patches applied — context

### `gaussian_renderer/__init__.py`

The upstream renderer assumed exactly 2 rasterizer outputs. After patching, it handles both 2-output (upstream) and 4-output (patched rasterizer with z_density) variants gracefully.

### `scene/dataset_readers.py`

Two patches: tukey import fix + zero-depth filter in histogram extraction. The depth histogram is stored in `CameraInfo.depth` as a `(n_bins, h_res)` and `(n_bins, w_res)` pair.

### `diff-gaussian-rasterization`

Submodule at `z_splatting/submodules/diff-gaussian-rasterization`. The `cstdint` fix is the only patch — no functional changes to the rasterizer logic.

---

## Diagnostic prints reference

Both v2 trainers emit structured prints to make log monitoring easy.

### Z-Splat v2 startup

```
SONAR_PATH=full_render          # or "histogram" or "none"
depth_scale = 1.315
sigma_r = 0.0125 m
sonar_max_range = 5.0 m
z_loss_weight = 0.12
camera_loss_weight = 1.0
```

### Z-Splat v2 every test iteration

```
Loss breakdown | ZL= X.XX  L_cam= X.XX  elev= X.XX  reg= X.XX  total= X.XX
Sonar test Gamma-NLL: X.XXX
```

### Z-Splat v2 every 500 iterations

```
∇norms: xyz= X.XXXX  r_tilde= X.XXXX  opacity= X.XXXX  f_dc= X.XXXX
Sonar GT diagnostics: shape=(96, 256)  max=X.XX  nonzero=XXXX
```

### Iteration 1 gradient check (Z-Splat v2)

```
[iter 1] r_tilde.grad norm: X.XXXX   ← must be non-zero if Z loss path is active
```

If this prints 0.0000, the Z loss path is not reaching `r_tilde`. Check:
1. Is `--sonar_data_dir` set and the path valid?
2. Are sonar PKL files matched to image stems?
3. Is `h_res` / `w_res` > 0 for the histogram path?
