# Physics

All physics losses and their derivations. Both v2 trainers implement every component below.

---

## Per-Gaussian reflectivity `r_tilde`

Each Gaussian carries a learned scalar reflectivity:

```
r_tilde = sigmoid(ρ),   ρ ∈ ℝ  (unconstrained, per-Gaussian parameter)
r_tilde ∈ (0, 1)
```

Initialised at `sigmoid(4.0) ≈ 0.982` (high initial reflectivity, so early gradients are informative).

`r_tilde` modulates the effective opacity used in sonar rendering:

```
eff_opacity = opacity × r_tilde × beam_weight
```

It lives in a **separate optimiser** from all 3DGS geometry/appearance parameters and is clipped independently to prevent Z-loss gradients from interfering with the photometric loss budget.

---

## ULA beam pattern

A Uniform Linear Array of N elements with inter-element spacing d at wavelength λ produces an array gain:

```
u_az  = (d/λ) × sin(θ)           azimuth angle θ from boresight
u_el  = (d/λ) × sin(φ)           elevation angle φ from boresight

B_az(θ) = [sinc(N × u_az) / sinc(u_az)]²
B_el(φ) = cos²(φ)                elevation roll-off

B(θ, φ) = B_az × B_el            combined gain, ∈ [0, 1]
```

Implementation uses `torch.sinc` (not a manual `sin(x)/x`) to avoid NaN gradients at u=0.

**Hardware defaults (Oculus M750d):**

| Parameter | Value |
|-----------|-------|
| `speed_of_sound` | 1500.0 m/s |
| `bandwidth` | 30 000 Hz |
| `n_array_elements` | 64 |
| `element_spacing` | 0.003 m |
| `center_frequency` | 1 100 000 Hz |
| `wavelength` | c / f_c = 1.36 mm |
| `sigma_r` | c / (4B) = **1.25 cm** (range resolution) |

---

## Gamma NLL loss

Sonar speckle is multiplicative exponential noise. If `S_measured ~ Exp(rate=1/S_hat)` then:

```
log p(S_measured | S_hat) = -S_measured/S_hat - log(S_hat)
```

Negating and averaging over all pixels gives the loss:

```
L_gamma = mean( S_measured / S_hat  +  log(S_hat) )
```

`S_hat` is clamped to ε = 0.01 to prevent log(0) during the early training phase before the model has converged.

This is the MLE estimator for exponential (Gamma with shape=1) noise, which is the correct model for fully developed sonar speckle. L1 assumes additive Gaussian noise and systematically underestimates bright returns.

---

## Elevation constraint loss

2D imaging sonar cannot resolve elevation. A target at slant range r_sonar could be at any elevation φ satisfying `||position||_cam = r_sonar`. The elevation loss anchors Gaussian centres to their measured slant range:

```
r_sonar_n  = extracted from range profile (3σ_r windowed peak detection)
r_predicted_n = ||μ_n - cam_center||        (camera-relative Euclidean range)

L_elevation = mean_n( (r_predicted_n - r_sonar_n)² )
```

**Critical detail:** `r_predicted_n` must be computed relative to `cam_center`, not the world origin. The fix (2026-04-13, train_v2.py) and (2026-04-16, sonar_simple_trainer_v2.py) uses `viewpoint_cam.camera_center.detach()` and `camtoworlds[0,:3,3].detach()` respectively.

The elevation weight is annealed:

```
w_e(t) = w_e_start  →  w_e_final   over  w_e_anneal_steps  iterations
         (1.0)          (0.1)              (10 000)
```

Early in training, elevation is a strong prior. As the scene self-organises, it relaxes to a soft regulariser.

---

## Reflectivity spatial regulariser

Prevents degenerate solutions where all Gaussians collapse to the same reflectivity. Applied every `reflectivity_reg_every` steps on a random subsample of 4096 Gaussians (prevents OOM from `torch.cdist` post-densification).

```
r_neighbors = mean of k=8 nearest neighbours (kNN in world space)
r_bar       = mean(r_tilde).detach()         (global mean, no gradient)

L_smooth   = mean( (r_tilde - r_neighbors)² )
L_revert   = mean( (r_tilde - r_bar)² )

L_reg = L_smooth  +  lambda_reg × L_revert
```

`L_smooth` enforces spatial coherence (nearby Gaussians should have similar reflectivity).  
`L_revert` prevents the distribution from drifting arbitrarily (weak pull toward the global mean).

---

## Energy preservation loss (SonarSplat v2 only)

A global conservation term preventing the optimizer from "winning" by collapsing all predicted intensities toward zero:

```
L_energy = ( (Σ S_pred - Σ S_gt) / (Σ S_gt + ε) )²
```

Applied at weight 0.1. Disabled in Z-Splat v2 (RGB supervision already provides global scale anchor).

---

## Total loss assembly

### SonarSplat v2

```
L = (1 - λ_ssim) × L_gamma  +  λ_ssim × L_ssim
  + w_max_size   × L_max_size
  + z_w × w_e(t) × L_elevation
  + z_w × w_reg  × L_reg
  + z_w × w_energy × L_energy
```

`z_loss_weight` (z_w) is the master ablation gate: set to 0 for a clean L1 baseline, 1.0 for full v2.

### Z-Splat v2

```
L = z_weight    × L_sonar       (gamma NLL on rendered sonar image)
  + cam_weight  × L_camera      (standard photometric loss on RGB)
  + w_e(t)      × L_elevation
  + w_reg       × L_reg
```

`z_weight` and `cam_weight` can be set manually or adapted by the optional RL controller.
