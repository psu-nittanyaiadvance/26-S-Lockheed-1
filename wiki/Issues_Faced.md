# Issues Faced and Resolved

This document outlines the primary development challenges, legacy technical debt, and issues we faced—and subsequently resolved—while streamlining the `z_splatting` execution pipeline and Docker workflow.

## 1. Fragmented Execution Scripts
**The Issue:**
Previously, running the various branches and datasets of Z-Splat (e.g., standard AONeuS vs. Sonar-only) required invoking a plethora of hardcoded shell scripts inside the `z_splatting/scripts/` directory such as `run_aoneus_v2.sh`, `run_monohansett_v2.sh`, and `run_sonar_v2.sh`. These scripts:
- Contained hardcoded paths corresponding to specific development environments (`/media/priyanshu/2TB SSD/...`).
- Relied on rigid positional arguments which made it difficult to change parameters on the fly without memorizing flags.

**The Solution:**
We deprecated and completely removed these redundant bash scripts. In their place, we instituted `run.py`. This interactive Python wrapper dynamically formulates the correct CLI arguments, detects the appropriate dataset path provided by the user, and launches the requisite version (`v1` vs. `v2`) eliminating the fragmented script mess.

## 2. Docker Container Usability and Path Constraints
**The Issue:**
The previous Docker deployment explicitly relied on a rigid `DATASET_PATH` environment variable. If a user invoked the launch scripts without manually exporting this variable beforehand, the script would unexpectedly crash and exit, leading to a poor developer experience. It essentially demanded that the user structure their bash environment correctly rather than assisting them.

**The Solution:**
`docker_run.sh` was rewritten to be robust and interactive. Instead of aggressively terminating upon discovering a missing path, the script now gracefully prompts the user in real-time (`"Please enter the path to the dataset you wish to mount:"`) while continuing to leverage the existing heavyweight CUDA-compiled Docker image.

## 3. Cluttered Workspace Environment
**The Issue:**
Because the codebase acts as an intersection point for both `sonar_splat` and native `z_splatting`, frequent local testing was polluting the directory with Python cache items (`__pycache__`, `.pyc`) and dangling legacy artifacts. This made reading the directory structure confusing for new contributors.

**The Solution:**
We enforced a deep directory purge to strip out the compiled caches and legacy launch files, maintaining an organized and uncluttered environment specifically focused entirely around `train.py`, `train_v2.py`, and `run.py`.

---

## 4. CUDA Illegal Memory Access — NaN in Sonar Rasterizer
**The Issue:**
Training on the Sunboat dataset (Oculus M1200d, 256 array elements) crashed twice mid-run with a CUDA illegal memory access inside `_quat_scale_to_covar_preci`. The crash pattern was:
1. `sonar_pred` from `_sonar_rasterization` would silently produce NaN values at a certain iteration.
2. NaN propagated through the Gamma NLL loss into gradients.
3. The NaN gradient check in place only zeroed gradients for `xyz`, `opacity`, and `r_tilde` — it did **not** cover `_scaling` and `_rotation` (quaternions).
4. The optimizer applied NaN gradients to scales and quats.
5. Next iteration, `_quat_scale_to_covar_preci` received NaN inputs and issued an illegal memory access on the GPU.

The crash happened at iter 651 on the first attempt and iter 7,774 on the second, making it appear non-deterministic at first.

**The Solution:**
Two guards were added to `train_v2.py`:
1. After `render_sonar_image()`, explicitly check for NaN/Inf in `sonar_pred` and skip the sonar loss entirely for that iteration rather than letting NaN enter the backward pass.
2. Extended the NaN gradient zeroing loop to cover **all** parameter groups in `gaussians.optimizer`, not just the three originally listed.

The root cause of *why* `sonar_pred` produces NaN for this scene (likely related to the very large scene extent at `depth_scale=1.848` and `max_range=26.784 m`) was not fully identified and remains an open issue.

---

## 5. ULA Beam Pattern Too Narrow for Large Arrays — Gradient Starvation
**The Issue:**
The Oculus M1200d sonar has 256 array elements, giving a half-power beamwidth of ~0.45°. When the full ULA beam pattern was applied from iteration 1, essentially all Gaussians fell outside the main lobe for every camera pose. Their effective opacity in the sonar render was ~0, the rasterizer produced near-zero or NaN alpha values, and the gradient back to `r_tilde` was zero.

On the Oculus M750d datasets (64 elements, ~1.8° beamwidth) this was never a problem — enough Gaussians are always inside the lobe to get meaningful gradients.

**The Solution:**
Beam annealing was implemented: `beam_w = beam_alpha * physics_beam + (1 - beam_alpha)`, where `beam_alpha` linearly ramps from 0 (flat beam — all Gaussians equally weighted) to 1 (full ULA physics) over `--beam_anneal_steps` iterations. For Sunboat, `--beam_anneal_steps 15000` was used. This is a heuristic schedule, not a principled derivation, and the optimal ramp rate is still unknown.

---

## 6. SSIM Crash When `camera_loss_weight=0`
**The Issue:**
On sonar-only datasets (`camera_loss_weight=0`), the training still computed the SSIM component of the camera loss. When NaN values from the sonar path corrupted GPU memory, the SSIM convolution kernel called `window.cuda(img1.get_device())` on a corrupted device state and issued a fatal CUDA error. This was a secondary crash mode distinct from the quat/scale crash above.

**The Solution:**
Added a guard in `train_v2.py`:
```python
if camera_loss_weight > 0 and opt.lambda_dssim > 0:
    L_cam = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
else:
    L_cam = Ll1
```
SSIM is now skipped entirely when `camera_loss_weight=0`, avoiding the crash and saving the computation.

---

## 7. r_tilde Gradient Zero — Missing CUDA z_density Output
**The Issue:**
In early Z-Splat v2 runs (before April 2026), `r_tilde` was stuck at 0.5 for the entire 30K-step run on AONeuS. The Z-loss was being computed and logged, but the gradient was never reaching `r_tilde`. Root cause: the CUDA `_sonar_rasterization` kernel outputs a `z_density` tensor but it was returning all-zeros, so the gradient path from `ZL → z_density → eff_opacity → r_tilde` was dead.

**The Solution:**
Replaced the broken CUDA z_density output with a differentiable Python `scatter_add` implementation that accumulates per-Gaussian opacity contributions into range bins. This restored the gradient path. Confirmed at iteration 1 via a connectivity check: `r_tilde.grad` is printed and verified non-zero when `ZL > 0`.

---

## 8. Storage Exhaustion on 2TB SSD Mid-Conversion
**The Issue:**
When converting the Sunboat dataset (383 PKL files of sonar scans) to COLMAP format via `convert_to_zsplat.py`, the 2TB SSD reached 100% capacity mid-conversion. The converter was writing PNG images and binary COLMAP files to `/media/priyanshu/2TB SSD/`, which was already near capacity from accumulated training results.

**The Solution:**
Redirected the converter output to the NVMe drive (`~/datasets/sunboat_zsplat`) and all subsequent training results to `~/results/` on NVMe. Freed 61 GB from the 2TB SSD by deleting superseded training runs, keeping only the best checkpoints.

---

## 9. Elevation Constraint Divergence on Sunboat
**The Issue:**
The elevation constraint loss — which penalises the distance between each Gaussian's predicted slant range and the sonar-measured peak return — was expected to decrease to near zero as training progressed. On the Sunboat dataset it instead stayed between 10,000 and 15,000 throughout all 30,000 training steps, indicating the Gaussians were never anchored to the measured depth.

Likely cause: the `depth_scale` calibration (1.848 wu/m for Sunboat) converts world-unit distances to sonar bins differently than expected, so the elevation loss target `r_sonar` is computed in the wrong units relative to the Gaussian positions.

**Status:** Unresolved. Requires verifying the `depth_scale → sonar bin` conversion for scenes with max_range significantly larger than the camera extent.

---

## 10. Gamma NLL vs SSIM Metric Conflict
**The Issue:**
Gamma NLL is the statistically correct maximum likelihood estimator for exponential (speckle) noise — using it improves PSNR by +3.36 dB on monohansett (15.93 → 19.29). However, it explicitly does not optimise for spatial coherence. The result: SSIM dropped from 0.63 to 0.29 in the same run. Reviewers or automated benchmarks that use SSIM as the primary metric will report a regression despite the physically more correct reconstruction.

**Status:** Known limitation. The correct interpretation is that SSIM is a mis-specified metric for speckle-noise imagery. No code fix — needs to be addressed at the evaluation and framing level.

---

## 11. RL Controller Collapses Sonar Weight on AONeuS
**The Issue:**
The RL loss-balancing controller (`--use_rl_controller`) is designed to keep the sonar and camera losses contributing equally. On the AONeuS RGB+sonar dataset it auto-calibrated to `z_w=0.226, cam_w=4.43` at step 100, then progressively drove the sonar weight down to `z_w=0.025, cam_w=15.3` by step 14,000 — a 9× reduction. The sonar physics is effectively disabled mid-training by the controller.

Root cause: the sonar Gamma-NLL on AONeuS decreases rapidly in early training (scene is learning fast from RGB), making the controller interpret sonar as over-contributing and reduce its weight. Once the sonar weight is low, it never recovers because the controller has no memory of the intended balance.

**Status:** Unresolved. May require a minimum floor on `z_w` in the RL controller, or a different balancing objective that prevents runaway weight collapse.

---

## 12. Post-Densification Gamma-NLL Never Recovers
**The Issue:**
On the monohansett sonar-only run, the Sonar test Gamma-NLL followed this trajectory: 5.50 (5K steps) → 6.17 (17K, densification peak) → 5.82 (30K, partial recovery). The 3DGS densification procedure (splitting high-gradient Gaussians, pruning transparent ones) restructures the point cloud in a way that disrupts the sonar-adapted geometry. The model never recaptured the pre-densification minimum.

**Status:** Partially open. Possible mitigations include reducing densification aggressiveness (`--densify_grad_threshold`), stopping densification earlier, or checkpointing before densification and resuming with it disabled.
