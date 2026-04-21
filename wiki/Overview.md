# Overview

## What this project is

This repository implements two novel 3D Gaussian Splatting models for underwater sonar imaging. The work extends two published baselines — SonarSplat (IEEE RA-L 2025, U-Michigan) and Z-Splat (IEEE TPAMI 2024) — with a shared physics layer modelling how acoustic energy interacts with underwater targets.

The goal is novel view synthesis of imaging sonar: given a set of sonar sweeps taken from known poses, reconstruct a 3D scene and render photorealistic sonar images from unseen viewpoints.

---

## Why sonar is hard for standard NeRF / Gaussian splatting

Standard 3DGS assumes light transport: emitter far away, intensity proportional to albedo, additive Gaussian noise. Imaging sonar breaks all three:

1. **Multiplicative noise.** Sonar speckle is the coherent interference of many sub-wavelength scatterers. Its amplitude follows a Rayleigh distribution; its intensity follows an exponential. The MLE loss for exponential noise is Gamma NLL, not L1 or L2.

2. **Beam pattern.** A phased array of N elements has an angular gain pattern `B(θ) ∝ [sinc(N·u)/sinc(u)]²` where `u = (d/λ)sin(θ)`. Gaussians off boresight contribute less regardless of their reflectivity. Ignoring this bakes view-dependent artefacts into the geometry.

3. **Elevation ambiguity.** 2D imaging sonar gives range and azimuth but not elevation. A target at slant range r could be anywhere on an arc of radius r. The elevation constraint loss regularises this by anchoring Gaussian positions to their measured slant range.

4. **Reflectivity vs opacity.** RGB scenes encode colour in spherical harmonics; sonar scenes need a scalar acoustic reflectivity per surface element. A learned sigmoid parameter `r_tilde ∈ (0,1)` per Gaussian, separate from the 3DGS opacity used for geometry, serves this role.

---

## Two models

### SonarSplat v2

- **Upstream:** SonarSplat (gsplat library, orthographic projection)
- **Input:** Sonar-only PKL files (range × azimuth intensity images)
- **Purpose:** Pure sonar NVS, no RGB supervision
- **Trainer:** `sonar_splat/examples/sonar_simple_trainer_v2.py`
- **Launcher:** `sonar_splat/scripts/run_v2.sh`

The upstream model used L1 loss and no beam physics. v2 replaces L1 with Gamma NLL, adds ULA beam weighting, elevation constraint, reflectivity regularisation, and energy preservation. These are gated by `--z_loss_weight` (0 = L1 baseline, 1 = full v2).

### Z-Splat v2

- **Upstream:** Z-Axis Gaussian Splatting (standard 3DGS codebase, diff-gaussian-rasterization)
- **Input:** RGB images + sonar depth maps (AONeuS format)
- **Purpose:** Fuse camera appearance with sonar geometry
- **Trainer:** `gaussian-splatting-with-depth/train_v2.py`
- **Launcher:** `gaussian-splatting-with-depth/scripts/run_aoneus_v2.sh`

Z-Splat adds a depth-histogram Z-loss on top of the standard photometric loss. v2 replaces the static zero-density output (non-differentiable CUDA path) with a fully differentiable Python scatter-add histogram, so `r_tilde` receives real gradients from the sonar loss. When `--sonar_data_dir` is provided, the full 2D sonar render path (`render_sonar_image`) is used instead.

---

## Shared physics additions

Both v2 trainers implement the same physics layer:

| Component | Description |
|-----------|-------------|
| `r_tilde` | Per-Gaussian sigmoid reflectivity, separate optimizer |
| ULA beam weight | Array gain `B(θ,φ)` per Gaussian per view |
| Gamma NLL | MLE loss for sonar speckle |
| Elevation constraint | Range-based position regularisation |
| Reflectivity regulariser | kNN spatial smoothness + mean reversion |
| Energy preservation | Global energy conservation (SonarSplat v2 only) |

---

## Branch structure

| Branch | Contents |
|--------|----------|
| `main` | Upstream baselines, no physics additions |
| `z_splatting` | All of Priyanshu's work — this branch |
