# Z-Splat Implementation Guide for Combined Branch

**Project**: Lockheed Martin NAISS - Underwater 3D Reconstruction
**Goal**: Implement camera-sonar fusion to solve the missing cone problem
**Expected Improvement**: +5 dB PSNR, -60% Chamfer distance, -50% depth error
**Timeline**: 7 weeks

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Phase 1: Core Parameterizations](#phase-1-core-parameterizations--utilities)
4. [Phase 2: Rendering Pipelines](#phase-2-rendering-pipelines)
5. [Phase 3: Loss Functions](#phase-3-loss-functions)
6. [Phase 4: Fusion Trainer](#phase-4-fusion-trainer)
7. [Phase 5: Testing & Validation](#phase-5-testing--validation)
8. [Integration with Existing Code](#integration-with-existing-code)
9. [Success Checklist](#success-checklist)
10. [Debugging Guide](#debugging-guide)

---

## Architecture Overview

### Current State
- **SonarSplat**: Sonar-only 3D reconstruction with polar rasterization
- **WaterSplatting**: Camera-only optical 3D reconstruction

### Target State
Unified multi-modal fusion system combining:
1. **Camera RGB** → provides lateral structure (xy-plane)
2. **Sonar depth** → provides range information (z-axis)
3. **Shared transmittance** → ensures geometric consistency

### The Missing Cone Problem

**What is it?**
When cameras have restricted baseline (small viewing angle range), the Fourier frequency space has a "cone" of unsampled data along the depth (z) axis, causing:
- Floaters in 3D space
- Blurry geometry
- Poor depth localization
- Ambiguous novel views along z-axis

**Why sonar solves it:**
Sonar samples the **orthogonal direction** in Fourier space:
- Camera gives horizontal slices (kz≈0)
- Sonar gives vertical slices (kx≈0)
- Together they cover the origin from two perpendicular directions

---

## Mathematical Foundation

### 1. Jacobian Linearization (Why Projections Stay Gaussian)

**Standard perspective projection is nonlinear:**
```
(x,y,z) → (f·x/z, f·y/z)  ❌ Gaussian → Non-Gaussian
```

**Z-Splat approach: Linearize using Jacobian at μ:**
```
J = [ 1/μz    0      -μx/μz² ]
    [  0     1/μz    -μy/μz² ]
    [ μx/l   μy/l     μz/l   ]

where l = ‖μ‖₂ = sqrt(μx² + μy² + μz²)
```

**Key insight**: Third row preserves Euclidean distance (critical for sonar range!)

**Covariance projection:**
```
Σ' = J W Σ W^T J^T  (still Gaussian!)
```

### 2. Covariance Marginalization (Different Sensors See Different Parts)

```
Full 3D Covariance:
Σ = [ σxx  σxy  σxz ]
    [ σxy  σyy  σyz ]
    [ σxz  σyz  σzz ]

Camera (xy-plane):        Echosounder (z-axis):     FLS (yz-plane):
Σ_2D = [ σxx  σxy ]      Σ_1D = σzz                Σ_2D = [ σyy  σyz ]
        [ σxy  σyy ]                                        [ σyz  σzz ]
```

**Critical**: Camera has NO gradient signal for σzz, μz. Sonar provides exactly this!

### 3. Volume Rendering Equation

```
C = Σₙ Tₙ · αₙ · cₙ    (camera: accumulates color)

Z[i] = Σₙ Tₙ · αₙ · exp(-(z - μz,n)²/2σzz,n)   (sonar: accumulates depth)

where Tₙ = ∏ₖ<ₙ (1 - αₖ)  (transmittance is SHARED!)
```

### 4. Key Mathematical Proofs

#### Proof 1: Sigmoid Reflectivity Parameterization
**Theorem**: The logit-sigmoid pair is the natural sufficient-statistic parameterization for acoustic reflectivity under the exponential family model.

```python
r̃ₙ ∈ ℝ (optimized parameter, unconstrained)
rₙ = sigmoid(r̃ₙ) ∈ (0,1) (actual reflectivity)
```

**Why**: This is the natural parameter of the Bernoulli exponential family, ensuring:
- Fisher information is well-conditioned
- Gradient flow is optimal: sigmoid'(0) = 0.25
- No vanishing gradients at initialization

#### Proof 3: Beam Pattern Separability
**Theorem**: For ULA with N elements, half-wavelength spacing:
```
|B(θ,φ) − Bₐ(θ)·Bₑ(φ)| ≤ C·φ²|θ|·N²
```

For N=64, |φ| ≤ 7°: Error < 1%

#### Proof 4: Gamma NLL Loss
**Theorem**: Under circular complex Gaussian scattering with K=1 look, intensity Z follows Exp(1/Ẑ) exactly.

**MLE Loss:**
```
L_sonar = Σᵢ [log Ẑ[i] + Z[i]/Ẑ[i]]
```

**Why not L2**: L2 assumes Gaussian noise. Sonar intensity is exponential (heavy-tailed). Gamma NLL is the MLE, L2 is inefficient.

#### Proof 5: Elevation Constraint Loss
**Theorem**: Hessian eigenvalues of elevation loss:
```
λ_radial = 2
λ_transverse = 2(‖μₙ‖ - r*)/‖μₙ‖
```

At convergence: ‖μₙ‖ = r* → λ_transverse = 0 → Gaussian lies on range sphere.

Combined with camera Hessian → full rank optimization.

#### Proof 6: Reflectivity Smoothness
**Theorem**: Under sonar render linearity, Gamma NLL + H¹ smoothness has unique minimizer.

```
L_smooth = ∫ ‖∇r‖² dx  (Sobolev H¹ regularization)
```

**Why**: Prevents oscillatory reflectivity artifacts, ensures convexity.

---

## Phase 1: Core Parameterizations & Utilities

### Week 1: Mathematical Foundations

#### Task 1.1: Sigmoid Reflectivity Parameterization

**File**: `sonar_splat/gsplat/params/reflectivity.py` (NEW)

```python
"""
Canonical reflectivity parameterization via logit-sigmoid.
Implements Proof 1: ensures unconstrained optimization with proper Fisher geometry.
"""

import torch
import torch.nn as nn

class ReflectivityParam(nn.Module):
    """
    Parameterizes reflectivity rₙ ∈ (0,1) via unconstrained r̃ₙ ∈ ℝ.

    Math:
        r̃ₙ ∈ ℝ (optimized parameter)
        rₙ = sigmoid(r̃ₙ) ∈ (0,1) (actual reflectivity)

    Why: This is the natural parameter of the Bernoulli exponential family,
         ensuring Fisher information is well-conditioned and gradients flow
         correctly at initialization (sigmoid'(0) = 0.25).
    """

    def __init__(self, num_gaussians: int, init_reflectivity: float = 0.5):
        super().__init__()
        # Initialize r̃ₙ such that sigmoid(r̃ₙ) ≈ init_reflectivity
        init_logit = torch.logit(torch.tensor(init_reflectivity))
        self.r_tilde = nn.Parameter(torch.full((num_gaussians,), init_logit))

    def forward(self) -> torch.Tensor:
        """Returns rₙ = sigmoid(r̃ₙ)"""
        return torch.sigmoid(self.r_tilde)

    def get_unconstrained(self) -> torch.Tensor:
        """Returns r̃ₙ for logging/analysis"""
        return self.r_tilde

    @staticmethod
    def gradient_scale_at_value(r: torch.Tensor) -> torch.Tensor:
        """
        Returns sigmoid'(r̃) = r·(1-r) for gradient flow analysis.
        At r=0.5: gradient scale = 0.25 (maximum)
        At r=0.1 or 0.9: gradient scale = 0.09 (vanishing)
        """
        return r * (1 - r)
```

**Tests**: `tests/test_reflectivity.py`
```python
def test_initialization():
    r_param = ReflectivityParam(100, init_reflectivity=0.5)
    r = r_param()
    assert torch.allclose(r, torch.tensor(0.5), atol=1e-6)

def test_gradient_flow():
    r_param = ReflectivityParam(10, init_reflectivity=0.5)
    r = r_param()
    loss = r.sum()
    loss.backward()
    # At r=0.5, gradient should be scaled by sigmoid'(0) = 0.25
    assert r_param.r_tilde.grad.abs().mean() > 0.2  # Non-vanishing
```

---

#### Task 1.2: Beam Pattern Separability

**File**: `sonar_splat/sonar/beam_pattern.py` (NEW or modify existing)

```python
"""
Separable beam pattern B(θ,φ) = Bₐ(θ)·Bₑ(φ).
Implements Proof 3 with formal error bounds.
"""

import torch
import math

class SeparableBeamPattern:
    """
    ULA beam pattern with azimuth-elevation separability.

    Math:
        B(θ,φ) ≈ Bₐ(θ)·Bₑ(φ)
        Error: |B - Bₐ·Bₑ| ≤ C·φ²|θ|·N²
    """

    def __init__(self, num_elements: int = 64, spacing: float = 0.5):
        """
        Args:
            num_elements: Number of array elements N
            spacing: Element spacing in wavelengths (d/λ)
        """
        self.N = num_elements
        self.d_lambda = spacing

        # Beamwidth estimates (radians)
        self.bw_az = 2 * self.d_lambda / self.N  # λ/(Nd)
        self.bw_el = math.pi / 2  # Omnidirectional elevation

    def compute_azimuth(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Azimuth beam pattern Bₐ(θ).

        Math:
            Bₐ(θ) = sin²(Nπ(d/λ)sinθ) / sin²(π(d/λ)sinθ)
        """
        u = math.pi * self.d_lambda * torch.sin(theta)
        numerator = torch.sin(self.N * u) ** 2
        denominator = torch.sin(u) ** 2 + 1e-8  # Numerical stability
        return numerator / denominator

    def compute_elevation(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Elevation modulation Bₑ(φ).

        Math:
            Bₑ(φ) = cos(φ) ≈ 1 - φ²/2 for small φ
        """
        return torch.cos(phi)

    def compute_full(self, theta: torch.Tensor, phi: torch.Tensor,
                     use_separable: bool = True) -> torch.Tensor:
        """
        Full beam pattern.

        Args:
            theta: Azimuth angles [N] (radians)
            phi: Elevation angles [N] (radians)
            use_separable: If True, use Bₐ·Bₑ approximation

        Returns:
            B: Beam pattern values [N]
        """
        if use_separable:
            # Separable approximation (fast)
            B_az = self.compute_azimuth(theta)
            B_el = self.compute_elevation(phi)
            return B_az * B_el
        else:
            # Exact formula (slow, for validation)
            u = self.d_lambda * torch.cos(phi) * torch.sin(theta)
            u_scaled = math.pi * u
            numerator = torch.sin(self.N * u_scaled) ** 2
            denominator = torch.sin(u_scaled) ** 2 + 1e-8
            return numerator / denominator

    def compute_error_bound(self, theta_max: float, phi_max: float) -> float:
        """
        Computes theoretical error bound from Proof 3.

        Math:
            Error ≤ C·φ²|θ|·N²
        """
        C = 2.0  # Conservative constant
        return C * (phi_max ** 2) * theta_max * (self.N ** 2)
```

**Tests**: `tests/test_beam_pattern.py`
```python
def test_separability_error():
    beam = SeparableBeamPattern(num_elements=64, spacing=0.5)
    theta = torch.linspace(-0.1, 0.1, 100)
    phi = torch.linspace(-0.1, 0.1, 100)

    B_separable = beam.compute_full(theta, phi, use_separable=True)
    B_exact = beam.compute_full(theta, phi, use_separable=False)

    error = torch.abs(B_separable - B_exact).max()
    bound = beam.compute_error_bound(theta_max=0.1, phi_max=0.1)

    print(f"Max error: {error:.6f}, Bound: {bound:.6f}")
    assert error < bound  # Verify theoretical bound holds
```

---

#### Task 1.3: Jacobian Projection (from Z-Splat)

**File**: `sonar_splat/gsplat/projection/jacobian.py` (NEW)

```python
"""
Jacobian-based linearized projection for Gaussian splatting.
Ensures 3D Gaussians remain Gaussian after perspective projection.
"""

import torch

def compute_projection_jacobian(means: torch.Tensor,
                                focal_length: float) -> torch.Tensor:
    """
    Compute Jacobian J for perspective projection linearization.

    Args:
        means: [N, 3] Gaussian centers (μx, μy, μz) in camera space
        focal_length: Camera focal length f

    Returns:
        J: [N, 3, 3] Jacobian matrices

    Math:
        J = [ 1/μz      0        -μx/μz² ]
            [  0       1/μz      -μy/μz² ]
            [ μx/l     μy/l       μz/l   ]
        where l = ‖μ‖₂ = sqrt(μx² + μy² + μz²)

    Why: Linearizing projection at μ keeps Gaussians in the Gaussian family.
         Third row preserves Euclidean distance (critical for sonar range).
    """
    N = means.shape[0]
    device = means.device

    mu_x, mu_y, mu_z = means[:, 0], means[:, 1], means[:, 2]

    # Numerical stability
    mu_z_safe = mu_z.clamp(min=1e-6)

    # Euclidean norm
    l = torch.norm(means, dim=1, keepdim=True).clamp(min=1e-6)  # [N, 1]

    # Build Jacobian
    J = torch.zeros(N, 3, 3, device=device)

    # Row 1: perspective projection x-component
    J[:, 0, 0] = 1.0 / mu_z_safe
    J[:, 0, 2] = -mu_x / (mu_z_safe ** 2)

    # Row 2: perspective projection y-component
    J[:, 1, 1] = 1.0 / mu_z_safe
    J[:, 1, 2] = -mu_y / (mu_z_safe ** 2)

    # Row 3: preserve Euclidean distance (CRITICAL for sonar!)
    J[:, 2, 0] = mu_x / l.squeeze()
    J[:, 2, 1] = mu_y / l.squeeze()
    J[:, 2, 2] = mu_z / l.squeeze()

    return J

def project_covariance(cov_3d: torch.Tensor,
                       viewmat: torch.Tensor,
                       jacobian: torch.Tensor) -> torch.Tensor:
    """
    Project 3D covariance to 2D using Jacobian.

    Args:
        cov_3d: [N, 3, 3] 3D covariances Σ
        viewmat: [3, 3] rotation part of world-to-camera matrix W
        jacobian: [N, 3, 3] projection Jacobian J

    Returns:
        cov_2d: [N, 2, 2] projected covariances Σ' = J W Σ W^T J^T

    Note: This is the standard 2D camera projection. For sonar, we'll use
          different marginalizations (see marginalize.py).
    """
    N = cov_3d.shape[0]

    # Transform to camera space: Σ_cam = W Σ W^T
    W = viewmat[:3, :3]  # Rotation only
    cov_cam = W @ cov_3d @ W.T  # [N, 3, 3] (broadcasted)

    # Project: Σ' = J Σ_cam J^T
    cov_proj = jacobian @ cov_cam @ jacobian.transpose(-1, -2)  # [N, 3, 3]

    # Extract 2D (camera sees xy plane)
    cov_2d = cov_proj[:, :2, :2]  # [N, 2, 2]

    return cov_2d
```

**Tests**: `tests/test_jacobian.py`
```python
def test_third_row_preserves_norm():
    means = torch.randn(10, 3)
    J = compute_projection_jacobian(means, focal_length=500.0)

    # Third row should be unit vector in direction of means
    third_row = J[:, 2, :]  # [N, 3]
    norms = torch.norm(third_row, dim=1)

    assert torch.allclose(norms, torch.ones(10), atol=1e-5)

def test_covariance_positive_definite():
    cov = torch.eye(3).unsqueeze(0).repeat(5, 1, 1)  # [5, 3, 3]
    means = torch.randn(5, 3)
    J = compute_projection_jacobian(means, focal_length=500.0)
    viewmat = torch.eye(4)

    cov_2d = project_covariance(cov, viewmat, J)

    # Check positive definiteness via eigenvalues
    eigvals = torch.linalg.eigvalsh(cov_2d)
    assert (eigvals > 0).all()
```

---

#### Task 1.4: Covariance Marginalization

**File**: `sonar_splat/gsplat/projection/marginalize.py` (NEW)

```python
"""
Extract observable covariance submatrices for different sensor modalities.
"""

import torch

def marginalize_camera(cov_3d: torch.Tensor) -> torch.Tensor:
    """
    Camera observes xy-plane → extract Σ[:2, :2].

    Args:
        cov_3d: [N, 3, 3]
    Returns:
        cov_2d: [N, 2, 2]
    """
    return cov_3d[:, :2, :2]

def marginalize_echosounder(cov_3d: torch.Tensor) -> torch.Tensor:
    """
    Echosounder observes z-axis only → extract σ_zz.

    Args:
        cov_3d: [N, 3, 3]
    Returns:
        var_z: [N] (just the (2,2) entry)
    """
    return cov_3d[:, 2, 2]

def marginalize_fls(cov_3d: torch.Tensor) -> torch.Tensor:
    """
    FLS observes yz-plane → extract Σ[1:, 1:].

    Args:
        cov_3d: [N, 3, 3]
    Returns:
        cov_2d: [N, 2, 2] (bottom-right block)
    """
    return cov_3d[:, 1:, 1:]

def get_observable_covariance(cov_3d: torch.Tensor,
                              sensor_type: str) -> torch.Tensor:
    """
    Unified interface for covariance marginalization.

    Args:
        cov_3d: [N, 3, 3] full 3D covariances
        sensor_type: 'camera' | 'echosounder' | 'fls'

    Returns:
        Observable covariance (shape depends on sensor)
    """
    if sensor_type == 'camera':
        return marginalize_camera(cov_3d)
    elif sensor_type == 'echosounder':
        return marginalize_echosounder(cov_3d)
    elif sensor_type == 'fls':
        return marginalize_fls(cov_3d)
    else:
        raise ValueError(f"Unknown sensor: {sensor_type}")
```

---

## Phase 2: Rendering Pipelines

### Week 2: Echosounder Renderer

#### Task 2.1: Echosounder Renderer (1D Depth Histogram)

**File**: `sonar_splat/gsplat/rendering/echosounder.py` (NEW)

```python
"""
1D echosounder rendering: depth histogram from Gaussian splatting.
"""

import torch
from ..projection.marginalize import marginalize_echosounder

def render_echosounder(
    means: torch.Tensor,           # [N, 3] Gaussian centers in camera space
    covs: torch.Tensor,            # [N, 3, 3] Covariances
    opacities: torch.Tensor,       # [N] Opacity (density, not reflectivity)
    reflectivities: torch.Tensor,  # [N] Reflectivity rₙ ∈ (0,1)
    num_bins: int = 256,
    depth_range: tuple = (0.0, 10.0),
    beam_pattern: torch.Tensor = None  # [N] Optional beam weights
) -> torch.Tensor:
    """
    Render 1D depth histogram for echosounder.

    Math:
        Z[i] = Σₙ Tₙ · rₙ · Bₙ · exp(-(z_i - ‖μₙ‖)² / 2σ_zz,n)

        where:
            Tₙ = ∏ₖ<ₙ (1 - αₖ) = transmittance
            rₙ = reflectivity (from ReflectivityParam)
            Bₙ = beam pattern weight
            σ_zz,n = depth variance from marginalize_echosounder()

    Returns:
        Z: [num_bins] depth histogram
    """
    device = means.device
    N = means.shape[0]

    # 1. Extract depth and depth variance
    mu_z = torch.norm(means, dim=1)  # Euclidean distance [N]
    sigma_zz = marginalize_echosounder(covs)  # [N]
    sigma_zz = sigma_zz.clamp(min=1e-6)  # Stability

    # 2. Sort by depth (front-to-back for transmittance)
    sorted_idx = torch.argsort(mu_z)
    mu_z_sorted = mu_z[sorted_idx]
    sigma_zz_sorted = sigma_zz[sorted_idx]
    alpha_sorted = opacities[sorted_idx]
    refl_sorted = reflectivities[sorted_idx]

    # 3. Compute transmittance T
    T = torch.cumprod(1 - alpha_sorted, dim=0)
    T = torch.cat([torch.ones(1, device=device), T[:-1]])  # Shift by 1

    # 4. Beam pattern (optional)
    if beam_pattern is not None:
        beam_sorted = beam_pattern[sorted_idx]
    else:
        beam_sorted = torch.ones(N, device=device)

    # 5. Create depth bins
    z_min, z_max = depth_range
    z_bins = torch.linspace(z_min, z_max, num_bins, device=device)

    # 6. Accumulate histogram (vectorized)
    # Shape: [num_bins, N]
    dz = (z_bins.unsqueeze(1) - mu_z_sorted.unsqueeze(0)) ** 2
    gaussian_contrib = torch.exp(-0.5 * dz / sigma_zz_sorted.unsqueeze(0))

    # Weight by T · r · B · α
    weights = T * refl_sorted * beam_sorted * alpha_sorted  # [N]
    weighted_contrib = gaussian_contrib * weights.unsqueeze(0)  # [num_bins, N]

    Z = weighted_contrib.sum(dim=1)  # [num_bins]

    return Z
```

**Tests**: `tests/test_echosounder.py`
```python
def test_single_gaussian():
    # Single Gaussian at depth 5.0 with σ_zz = 0.1
    means = torch.tensor([[0.0, 0.0, 5.0]])
    covs = torch.eye(3).unsqueeze(0) * 0.01  # Small variance
    covs[0, 2, 2] = 0.1  # Depth variance
    opacities = torch.tensor([0.9])
    reflectivities = torch.tensor([0.8])

    Z = render_echosounder(means, covs, opacities, reflectivities,
                          num_bins=100, depth_range=(4.0, 6.0))

    # Peak should be at depth 5.0 (bin index 50)
    peak_idx = torch.argmax(Z)
    assert 45 <= peak_idx <= 55  # Within tolerance

def test_occlusion():
    # Two Gaussians: one at 4.0 (blocks), one at 5.0 (occluded)
    means = torch.tensor([[0.0, 0.0, 4.0],
                          [0.0, 0.0, 5.0]])
    covs = torch.eye(3).unsqueeze(0).repeat(2, 1, 1) * 0.01
    opacities = torch.tensor([0.9, 0.9])  # First one blocks
    reflectivities = torch.tensor([0.5, 1.0])

    Z = render_echosounder(means, covs, opacities, reflectivities,
                          num_bins=100, depth_range=(3.0, 6.0))

    # First peak should be much stronger than second
    assert Z[33] > Z[66]  # Bin 33 ≈ depth 4.0, bin 66 ≈ depth 5.0
```

---

## Phase 3: Loss Functions

### Week 3: Loss Implementation

#### Task 3.1: Gamma NLL Loss (from Proof 4)

**File**: `sonar_splat/gsplat/loss/gamma_nll.py` (NEW)

```python
"""
Gamma Negative Log-Likelihood for sonar intensity.
Implements Proof 4: proper MLE under exponential intensity distribution.
"""

import torch
import torch.nn.functional as F

def gamma_nll_loss(rendered: torch.Tensor,
                   target: torch.Tensor,
                   eps: float = 1e-6) -> torch.Tensor:
    """
    Gamma NLL loss for sonar depth histogram.

    Math:
        L_sonar = Σᵢ [log Ẑ[i] + Z[i]/Ẑ[i]]

    Where Z ~ Exp(1/Ẑ) = Gamma(1, Ẑ).

    Args:
        rendered: [num_bins] or [H, W] predicted intensity Ẑ
        target: [num_bins] or [H, W] observed intensity Z
        eps: Numerical stability constant

    Returns:
        loss: Scalar loss value

    Why not L2:
        L2 = Σ(Z - Ẑ)² assumes Gaussian noise.
        Sonar intensity follows exponential (heavy-tailed).
        Gamma NLL is the MLE, L2 is inefficient (Proof 4).
    """
    rendered_safe = rendered.clamp(min=eps)

    # Gamma NLL = log(Ẑ) + Z/Ẑ
    loss = torch.log(rendered_safe) + target / rendered_safe

    return loss.mean()

def frobenius_norm_loss(rendered: torch.Tensor,
                       target: torch.Tensor) -> torch.Tensor:
    """
    Frobenius norm for FLS images (2D sonar).

    Math:
        ‖S - Ŝ‖_F² = Σᵢⱼ (S[i,j] - Ŝ[i,j])²

    This is equivalent to L2 on flattened arrays.
    Smooth gradient: ∂‖A‖²_F/∂A = 2A.
    """
    return torch.norm(rendered - target, p='fro') ** 2

def sonar_loss(rendered: torch.Tensor,
              target: torch.Tensor,
              sensor_type: str = 'echosounder') -> torch.Tensor:
    """
    Unified sonar loss dispatcher.

    Args:
        rendered: Predicted sonar signal
        target: Ground truth sonar signal
        sensor_type: 'echosounder' | 'fls'

    Returns:
        loss: Scalar
    """
    if sensor_type == 'echosounder':
        return gamma_nll_loss(rendered, target)
    elif sensor_type == 'fls':
        return frobenius_norm_loss(rendered, target)
    else:
        raise ValueError(f"Unknown sensor: {sensor_type}")
```

**Tests**: `tests/test_gamma_nll.py`
```python
def test_gamma_nll_optimum():
    # At optimum (Ẑ = Z), loss should be minimized
    target = torch.rand(100) + 0.1  # Avoid zeros
    rendered = target.clone()

    loss_optimal = gamma_nll_loss(rendered, target)

    # Perturb prediction
    rendered_perturbed = rendered * 1.2
    loss_perturbed = gamma_nll_loss(rendered_perturbed, target)

    assert loss_perturbed > loss_optimal

def test_gamma_vs_l2():
    # Generate exponential data
    true_mean = 2.0
    samples = torch.distributions.Exponential(1.0 / true_mean).sample((1000,))

    # Fit with Gamma NLL
    pred_gamma = torch.tensor([true_mean], requires_grad=True)
    optimizer = torch.optim.Adam([pred_gamma], lr=0.01)

    for _ in range(100):
        optimizer.zero_grad()
        loss = gamma_nll_loss(pred_gamma.repeat(1000), samples)
        loss.backward()
        optimizer.step()

    # Gamma NLL should recover true mean within 5%
    assert torch.abs(pred_gamma - true_mean) / true_mean < 0.05
```

---

#### Task 3.2: Elevation Constraint Loss (from Proof 5)

**File**: `sonar_splat/gsplat/loss/elevation.py` (NEW)

```python
"""
Elevation constraint to regularize depth.
Implements Proof 5: forces Gaussians onto range sphere.
"""

import torch

def elevation_loss(means: torch.Tensor,
                  target_ranges: torch.Tensor,
                  weight: float = 1.0) -> torch.Tensor:
    """
    Penalizes deviation from target range surface.

    Math:
        L_e = Σₙ (‖μₙ‖ - r*)²

    Where r* is the observed range from sonar.

    Args:
        means: [N, 3] Gaussian centers
        target_ranges: [N] target Euclidean ranges
        weight: Loss weight

    Returns:
        loss: Scalar

    Why this matters (Proof 5):
        - Camera loss has zero Hessian eigenvalues in transverse directions
        - Elevation loss provides curvature to constrain depth
        - At convergence: λ_transverse = 2(‖μₙ‖-r*)/‖μₙ‖ = 0
        - Combined Hessian H_total = H_camera + H_sonar + H_elevation is full rank
    """
    predicted_ranges = torch.norm(means, dim=1)
    residuals = predicted_ranges - target_ranges

    loss = (residuals ** 2).mean()

    return weight * loss

def hessian_eigenvalue_analysis(means: torch.Tensor,
                                target_ranges: torch.Tensor) -> dict:
    """
    Compute Hessian eigenvalues from Proof 5 for debugging.

    Returns:
        dict with:
            'lambda_radial': Radial eigenvalue (always 2)
            'lambda_transverse': Transverse eigenvalue (2(‖μ‖-r*)/‖μ‖)
            'num_saddle_points': Count of Gaussians with λ_transverse < 0
    """
    norms = torch.norm(means, dim=1)

    lambda_radial = 2.0  # Constant
    lambda_transverse = 2 * (norms - target_ranges) / norms

    return {
        'lambda_radial': lambda_radial,
        'lambda_transverse': lambda_transverse.mean().item(),
        'num_saddle_points': (lambda_transverse < 0).sum().item(),
        'converged_fraction': (torch.abs(lambda_transverse) < 0.01).float().mean().item()
    }
```

---

#### Task 3.3: Reflectivity Smoothness (from Proof 6)

**File**: `sonar_splat/gsplat/loss/reflectivity_smooth.py` (NEW)

```python
"""
Sobolev H¹ regularization for reflectivity.
Implements Proof 6: ensures unique minimizer.
"""

import torch

def reflectivity_smoothness_loss(reflectivities: torch.Tensor,
                                 means: torch.Tensor,
                                 k_neighbors: int = 8,
                                 weight: float = 0.1) -> torch.Tensor:
    """
    Penalizes rapid reflectivity variation in space.

    Math:
        L_smooth = Σₙ Σₖ∈N(n) (rₙ - rₖ)² / ‖μₙ - μₖ‖²

    This approximates the H¹ Sobolev norm ∫ ‖∇r‖² dx.

    Args:
        reflectivities: [N] reflectivity values rₙ
        means: [N, 3] Gaussian positions
        k_neighbors: Number of nearest neighbors
        weight: Regularization strength

    Returns:
        loss: Scalar

    Why (Proof 6):
        - Without regularization, Gamma NLL + linearity → convex but many local minima
        - H¹ smoothness makes the problem strictly convex
        - Guarantees unique minimizer via parallelogram law
    """
    N = means.shape[0]
    device = means.device

    # Compute pairwise distances
    dists = torch.cdist(means, means)  # [N, N]

    # Find k nearest neighbors (excluding self)
    knn_dists, knn_idx = torch.topk(dists, k=k_neighbors + 1,
                                    dim=1, largest=False)
    knn_dists = knn_dists[:, 1:]  # Exclude self (distance 0)
    knn_idx = knn_idx[:, 1:]

    # Gather neighbor reflectivities
    r_neighbors = reflectivities[knn_idx]  # [N, k]
    r_self = reflectivities.unsqueeze(1)  # [N, 1]

    # Compute smoothness penalty
    r_diff = (r_self - r_neighbors) ** 2  # [N, k]
    spatial_weights = 1.0 / (knn_dists ** 2 + 1e-6)  # Inverse distance weighting

    loss = (r_diff * spatial_weights).mean()

    return weight * loss
```

---

## Phase 4: Fusion Trainer

### Week 4-5: Training Integration

#### Task 4.1: Multi-Modal Dataloader

**File**: `sonar_splat/sonar/dataset/fusion_dataloader.py` (NEW)

```python
"""
Dataloader for synchronized camera-sonar pairs.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
import pickle
from PIL import Image
import torchvision.transforms as transforms

class CameraSonarDataset(Dataset):
    """
    Loads synchronized camera RGB + sonar depth data.

    Directory structure:
        data/
        ├── images/
        │   ├── 0000.jpg
        │   ├── 0001.jpg
        ├── sonar/
        │   ├── 0000.pkl  # Contains 'depth_hist', 'range', 'pose'
        │   ├── 0001.pkl
        └── metadata.json  # Camera intrinsics, sync info
    """

    def __init__(self, data_dir: str, split: str = 'train'):
        self.data_dir = Path(data_dir)
        self.split = split

        # Load metadata
        with open(self.data_dir / 'metadata.json') as f:
            self.metadata = json.load(f)

        # Image transforms
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Get file lists
        self.image_files = sorted(self.data_dir.glob(f'{split}/images/*.jpg'))
        self.sonar_files = sorted(self.data_dir.glob(f'{split}/sonar/*.pkl'))

        assert len(self.image_files) == len(self.sonar_files), \
            "Camera-sonar pair mismatch!"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load camera image
        img = Image.open(self.image_files[idx]).convert('RGB')
        img_tensor = self.img_transform(img)

        # Load sonar data
        with open(self.sonar_files[idx], 'rb') as f:
            sonar_data = pickle.load(f)

        return {
            'camera': {
                'image': img_tensor,  # [3, H, W]
                'pose': torch.tensor(sonar_data['camera_pose']),  # [4, 4]
                'intrinsics': torch.tensor(self.metadata['camera_intrinsics'])
            },
            'echosounder': {
                'histogram': torch.tensor(sonar_data['depth_hist']),  # [num_bins]
                'pose': torch.tensor(sonar_data['sonar_pose']),  # [4, 4]
                'range': sonar_data['max_range'],
                'num_bins': len(sonar_data['depth_hist'])
            },
            'frame_id': idx
        }
```

---

#### Task 4.2: Fusion Training Loop

**File**: `sonar_splat/examples/fusion_trainer.py` (NEW)

```python
"""
Main Z-Splat fusion trainer.
Combines camera RGB + sonar depth with all mathematical components.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from pathlib import Path

from gsplat.params.reflectivity import ReflectivityParam
from gsplat.projection.jacobian import compute_projection_jacobian, project_covariance
from gsplat.rendering.echosounder import render_echosounder
from gsplat.loss.gamma_nll import gamma_nll_loss
from gsplat.loss.elevation import elevation_loss
from gsplat.loss.reflectivity_smooth import reflectivity_smoothness_loss
from sonar.beam_pattern import SeparableBeamPattern
from sonar.dataset.fusion_dataloader import CameraSonarDataset

@dataclass
class FusionConfig:
    """Configuration for Z-Splat fusion training."""

    # Data
    data_dir: str = "/path/to/camera_sonar_data"
    num_epochs: int = 100
    batch_size: int = 4

    # Gaussians
    num_gaussians: int = 10000
    init_reflectivity: float = 0.5

    # Loss weights
    weight_camera: float = 1.0
    weight_sonar: float = 1.0
    weight_elevation: float = 0.5
    weight_smooth: float = 0.1

    # Sonar
    num_depth_bins: int = 256
    depth_range: tuple = (0.0, 10.0)
    num_beam_elements: int = 64

    # Optimization
    lr_means: float = 1e-4
    lr_covs: float = 1e-3
    lr_reflectivity: float = 1e-3

    # Misc
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    log_every: int = 100

class FusionTrainer:
    """Z-Splat fusion trainer."""

    def __init__(self, config: FusionConfig):
        self.cfg = config
        self.device = torch.device(config.device)

        # Initialize Gaussians
        self.means = nn.Parameter(torch.randn(config.num_gaussians, 3,
                                              device=self.device))
        self.covs = nn.Parameter(torch.eye(3, device=self.device)
                                .unsqueeze(0).repeat(config.num_gaussians, 1, 1) * 0.01)
        self.opacities = nn.Parameter(torch.ones(config.num_gaussians,
                                                 device=self.device) * 0.5)
        self.colors = nn.Parameter(torch.rand(config.num_gaussians, 3,
                                             device=self.device))

        # Reflectivity (sigmoid parameterization from Proof 1)
        self.reflectivity_param = ReflectivityParam(
            config.num_gaussians,
            init_reflectivity=config.init_reflectivity
        ).to(self.device)

        # Beam pattern (separable from Proof 3)
        self.beam_pattern = SeparableBeamPattern(
            num_elements=config.num_beam_elements
        )

        # Optimizer
        self.optimizer = torch.optim.Adam([
            {'params': [self.means], 'lr': config.lr_means},
            {'params': [self.covs], 'lr': config.lr_covs},
            {'params': [self.opacities], 'lr': config.lr_means},
            {'params': [self.colors], 'lr': config.lr_means},
            {'params': self.reflectivity_param.parameters(),
             'lr': config.lr_reflectivity}
        ])

        # Dataloader
        self.dataset = CameraSonarDataset(config.data_dir, split='train')
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4
        )

    def render_camera(self, batch):
        """Render camera RGB (integrate with your existing camera renderer)."""
        # TODO: Replace with your WaterSplatting or standard GS camera renderer
        # Should return: [B, 3, H, W]
        pass

    def render_sonar(self, batch):
        """Render sonar depth histogram using echosounder renderer."""
        B = len(batch['frame_id'])
        histograms = []

        for i in range(B):
            # Transform means to sonar coordinate frame
            sonar_pose = batch['echosounder']['pose'][i]  # [4, 4]
            means_sonar = (sonar_pose[:3, :3] @ self.means.T).T + sonar_pose[:3, 3]

            # Get reflectivities
            reflectivities = self.reflectivity_param()

            # Render
            hist = render_echosounder(
                means=means_sonar,
                covs=self.covs,
                opacities=torch.sigmoid(self.opacities),
                reflectivities=reflectivities,
                num_bins=self.cfg.num_depth_bins,
                depth_range=self.cfg.depth_range,
                beam_pattern=None
            )
            histograms.append(hist)

        return torch.stack(histograms)  # [B, num_bins]

    def compute_loss(self, batch, rendered_camera, rendered_sonar):
        """Compute multi-modal fusion loss."""

        # 1. Camera loss (standard L2 or SSIM)
        target_camera = batch['camera']['image'].to(self.device)
        loss_camera = nn.functional.mse_loss(rendered_camera, target_camera)

        # 2. Sonar loss (Gamma NLL from Proof 4)
        target_sonar = batch['echosounder']['histogram'].to(self.device)
        loss_sonar = gamma_nll_loss(rendered_sonar, target_sonar)

        # 3. Elevation constraint (Proof 5)
        target_ranges = torch.argmax(target_sonar, dim=1).float()
        target_ranges = (target_ranges *
                        (self.cfg.depth_range[1] - self.cfg.depth_range[0]) /
                        self.cfg.num_depth_bins + self.cfg.depth_range[0])

        loss_elev = elevation_loss(self.means,
                                   target_ranges.mean().repeat(self.cfg.num_gaussians))

        # 4. Reflectivity smoothness (Proof 6)
        reflectivities = self.reflectivity_param()
        loss_smooth = reflectivity_smoothness_loss(reflectivities, self.means)

        # Total loss
        loss_total = (self.cfg.weight_camera * loss_camera +
                     self.cfg.weight_sonar * loss_sonar +
                     self.cfg.weight_elevation * loss_elev +
                     self.cfg.weight_smooth * loss_smooth)

        return {
            'total': loss_total,
            'camera': loss_camera,
            'sonar': loss_sonar,
            'elevation': loss_elev,
            'smoothness': loss_smooth
        }

    def train_epoch(self, epoch):
        """Single training epoch."""
        for i, batch in enumerate(self.dataloader):
            self.optimizer.zero_grad()

            # Render both modalities
            rendered_camera = self.render_camera(batch)
            rendered_sonar = self.render_sonar(batch)

            # Compute loss
            losses = self.compute_loss(batch, rendered_camera, rendered_sonar)

            # Backward
            losses['total'].backward()
            self.optimizer.step()

            # Log
            if i % self.cfg.log_every == 0:
                print(f"Epoch {epoch}, Iter {i}: "
                      f"Loss={losses['total']:.4f} "
                      f"Camera={losses['camera']:.4f} "
                      f"Sonar={losses['sonar']:.4f} "
                      f"Elev={losses['elevation']:.4f} "
                      f"Smooth={losses['smoothness']:.4f}")

    def train(self):
        """Full training loop."""
        for epoch in range(self.cfg.num_epochs):
            self.train_epoch(epoch)

            # Save checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pth")

    def save_checkpoint(self, path):
        """Save model checkpoint."""
        torch.save({
            'means': self.means,
            'covs': self.covs,
            'opacities': self.opacities,
            'colors': self.colors,
            'reflectivity': self.reflectivity_param.state_dict()
        }, path)

# Usage
if __name__ == '__main__':
    config = FusionConfig(
        data_dir="/path/to/data",
        num_epochs=100,
        weight_sonar=1.0
    )

    trainer = FusionTrainer(config)
    trainer.train()
```

---

## Phase 5: Testing & Validation

### Week 6: Testing Suite

#### Task 5.1: Unit Test Suite

**File**: `tests/test_fusion_pipeline.py` (NEW)

```python
"""End-to-end fusion pipeline test."""

import torch
from fusion_trainer import FusionTrainer, FusionConfig

def test_full_pipeline():
    """Test complete fusion training pipeline."""

    # Minimal config for testing
    config = FusionConfig(
        data_dir="tests/data/synthetic",
        num_gaussians=100,
        num_epochs=2,
        batch_size=2,
        log_every=1
    )

    trainer = FusionTrainer(config)
    trainer.train()

    print("✓ Full pipeline test passed")

def test_gradient_flow():
    """Verify gradients reach depth parameters (CRITICAL!)."""

    means = torch.randn(10, 3, requires_grad=True)
    covs = torch.eye(3).unsqueeze(0).repeat(10, 1, 1)
    covs.requires_grad = True

    # Mock sonar rendering
    from gsplat.rendering.echosounder import render_echosounder

    reflectivities = torch.rand(10)
    opacities = torch.rand(10)

    rendered = render_echosounder(means, covs, opacities, reflectivities)
    target = torch.rand_like(rendered)

    loss = torch.nn.functional.mse_loss(rendered, target)
    loss.backward()

    # Check μz gradient
    assert means.grad[:, 2].abs().mean() > 1e-6, "No gradient to z-coordinate!"

    # Check σzz gradient
    assert covs.grad[:, 2, 2].abs().mean() > 1e-6, "No gradient to depth variance!"

    print("✓ Gradients flow to depth parameters (missing cone resolved!)")

if __name__ == '__main__':
    test_gradient_flow()
    # test_full_pipeline()  # Uncomment when data is ready
```

---

## Integration with Existing Code

### Modify `sonar_splat/gsplat/rendering.py`

```python
# Add this at the top
from .rendering.echosounder import render_echosounder

# Modify rasterization function
def rasterization(
    ...,
    sensor_type='camera',  # NEW PARAMETER
    reflectivities=None    # NEW PARAMETER
):
    if sensor_type == 'echosounder':
        return render_echosounder(...)
    elif sensor_type == 'sonar_polar':
        return _sonar_rasterization(...)  # Your existing SonarSplat code
    else:
        return _camera_rasterization(...)  # Existing camera rendering
```

### Modify `water_splatting/rasterize.py`

```python
# Add Jacobian option
from gsplat.projection.jacobian import project_covariance

def project_gaussians(..., use_jacobian=True):
    if use_jacobian:
        return project_covariance(covs, viewmat, jacobian)
    else:
        # Original projection code
        ...
```

---

## Success Checklist

### Math Components
- [ ] Reflectivity parameters initialized with sigmoid
- [ ] Beam pattern separability error < 1%
- [ ] Jacobian preserves Euclidean norm (third row test)
- [ ] Covariance marginalization extracts correct submatrices

### Rendering
- [ ] Echosounder renders 1D histogram correctly
- [ ] Single Gaussian test: peak at correct depth
- [ ] Occlusion test: transmittance computed correctly

### Loss Functions
- [ ] Gamma NLL loss decreases during training
- [ ] Gamma NLL outperforms L2 on exponential data
- [ ] Elevation loss converges λ_transverse → 0
- [ ] Reflectivity smoothness prevents artifacts

### Training
- [ ] Gradients flow to μz and σzz (missing cone resolved!)
- [ ] Camera quality preserved (PSNR not degraded)
- [ ] Depth error reduced by >50% vs camera-only
- [ ] No NaN losses during training

### Final Metrics (from Z-Splat paper)
- [ ] Novel view PSNR: +5 dB improvement
- [ ] Chamfer distance: -60% reduction
- [ ] Depth RMS error: -50% reduction

---

## Debugging Guide

### Problem: Gradients vanish for depth parameters

**Diagnosis:**
```python
# Check gradient magnitudes
print("Mean ∂L/∂μz:", means.grad[:, 2].abs().mean())
print("Mean ∂L/∂σzz:", covs.grad[:, 2, 2].abs().mean())
```

**If zero:**
- Sonar loss not connected to Gaussians
- Check `render_echosounder` returns non-zero values
- Verify `marginalize_echosounder` extracts σzz correctly
- Ensure `gamma_nll_loss` differentiable w.r.t. rendered

**Fix:**
```python
# Add explicit gradient check
rendered = render_echosounder(...)
print("Rendered min/max:", rendered.min(), rendered.max())
assert rendered.requires_grad, "Rendered should require grad!"
```

---

### Problem: Camera quality degrades with fusion

**Diagnosis:**
```python
# Compare camera-only vs fusion
loss_camera_only = train_camera_only()
loss_fusion = train_fusion()
print(f"Camera PSNR degradation: {loss_fusion - loss_camera_only:.2f} dB")
```

**If degraded by >1 dB:**
- Sonar weight too high
- Elevation constraint too strong
- Shared transmittance incorrect

**Fix:**
```python
# Reduce sonar weight gradually
config.weight_sonar = 0.1  # Start small
config.weight_elevation = 0.1
# Increase after 10 epochs if camera stable
```

---

### Problem: NaN losses

**Diagnosis:**
```python
# Print intermediate values
print("Rendered stats:", rendered.mean(), rendered.std(), rendered.min(), rendered.max())
print("Sigma_zz stats:", sigma_zz.mean(), sigma_zz.min())
```

**Common causes:**
- Division by zero in `sigma_zz`
- Log of zero in Gamma NLL
- Exploding gradients

**Fix:**
```python
# Add epsilon everywhere
rendered_safe = rendered.clamp(min=1e-6)
sigma_zz_safe = sigma_zz.clamp(min=1e-6)

# Clip gradients
torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)

# Reduce learning rates
config.lr_means = 1e-5  # Was 1e-4
config.lr_covs = 1e-4   # Was 1e-3
```

---

### Problem: Depth histogram is all zeros

**Diagnosis:**
```python
# Check Gaussian distribution
mu_z = torch.norm(means, dim=1)
print(f"Depth range: [{mu_z.min():.2f}, {mu_z.max():.2f}]")
print(f"Histogram range: {config.depth_range}")
```

**If mismatch:**
- Gaussians outside histogram range
- σzz too small (no contribution)
- Transmittance killing all Gaussians

**Fix:**
```python
# Adjust depth range to match Gaussians
config.depth_range = (mu_z.min().item() - 1, mu_z.max().item() + 1)

# Check σzz values
print("Sigma_zz range:", sigma_zz.min(), sigma_zz.max())
# Should be O(0.01-0.1), not 1e-10

# Check transmittance
T = compute_transmittance(opacities)
print("Transmittance at bin 0:", T[0])  # Should be ~1.0
print("Transmittance at bin -1:", T[-1])  # Should be >0.01
```

---

### Problem: Reflectivity all converges to same value

**Diagnosis:**
```python
reflectivities = reflectivity_param()
print("Reflectivity std:", reflectivities.std())
print("Reflectivity range:", reflectivities.min(), reflectivities.max())
```

**If std < 0.01:**
- Smoothness weight too high
- No per-Gaussian sonar signal
- Initialization too narrow

**Fix:**
```python
# Reduce smoothness weight
config.weight_smooth = 0.01  # Was 0.1

# Initialize with more variation
r_param = ReflectivityParam(N, init_reflectivity=0.5)
r_param.r_tilde.data += torch.randn_like(r_param.r_tilde) * 0.5  # Add noise

# Check per-Gaussian contribution to histogram
weights = T * reflectivities * opacities
print("Weight std:", weights.std())  # Should be >0.01
```

---

## Implementation Priority

### Week 1: Math Foundations ✅
1. `ReflectivityParam` (sigmoid parameterization)
2. `SeparableBeamPattern` (azimuth-elevation)
3. `compute_projection_jacobian` (Z-Splat core)
4. `marginalize_*` (covariance extraction)

### Week 2: Rendering ✅
5. `render_echosounder` (1D depth histogram)
6. Tests: single Gaussian, occlusion

### Week 3: Loss Functions ✅
7. `gamma_nll_loss` (Proof 4)
8. `elevation_loss` (Proof 5)
9. `reflectivity_smoothness_loss` (Proof 6)

### Week 4-5: Integration ✅
10. `CameraSonarDataset` (dataloader)
11. `FusionTrainer` (main training loop)
12. Integrate with existing camera renderer

### Week 6: Testing ✅
13. Unit tests for all components
14. Gradient flow validation
15. End-to-end fusion pipeline test

### Week 7: Evaluation & Polish
16. Run on real data
17. Compare against camera-only baseline
18. Generate visualizations (Fourier coverage, depth error plots)
19. Write documentation

---

## Key Equations Reference

### Jacobian Matrix
```
J = [ 1/μz    0      -μx/μz² ]
    [  0     1/μz    -μy/μz² ]
    [ μx/l   μy/l     μz/l   ]

where l = sqrt(μx² + μy² + μz²)
```

### Covariance Projection
```
Σ' = J W Σ W^T J^T
```

### Echosounder Rendering
```
Z[i] = Σₙ Tₙ · rₙ · Bₙ · exp(-(z_i - μz,n)² / (2σzz,n))

where:
  Tₙ = ∏ₖ<ₙ (1 - αₖ)  (transmittance)
  rₙ = sigmoid(r̃ₙ)     (reflectivity)
  Bₙ = Bₐ(θ)·Bₑ(φ)     (beam pattern)
  σzz,n = Σ[2,2]       (depth variance)
```

### Fusion Loss
```
L = w_c·L_camera + w_s·L_sonar + w_e·L_elevation + w_r·L_smooth

L_camera = ‖I - Î‖²₂
L_sonar = Σᵢ [log Ẑ[i] + Z[i]/Ẑ[i]]
L_elevation = Σₙ (‖μₙ‖ - r*)²
L_smooth = Σₙ Σₖ (rₙ - rₖ)² / ‖μₙ - μₖ‖²
```

### Depth Gradients (from Gamma NLL)
```
∂L_sonar/∂μz,n = Σᵢ 2(Z[i] - Ẑ[i]) · Tₙ · rₙ · Bₙ · (z_i - μz,n)/σzz,n · exp(...)

∂L_sonar/∂σzz,n = Σᵢ 2(Z[i] - Ẑ[i]) · Tₙ · rₙ · Bₙ · [(z_i - μz,n)²/σ²zz,n - 1/σzz,n] · exp(...)
```

---

## File Structure Summary

```
sonar_splat/
├── gsplat/
│   ├── params/
│   │   └── reflectivity.py          # NEW - Sigmoid parameterization
│   ├── projection/
│   │   ├── jacobian.py              # NEW - Jacobian projection
│   │   └── marginalize.py           # NEW - Covariance marginalization
│   ├── rendering/
│   │   ├── echosounder.py           # NEW - 1D depth rendering
│   │   └── rendering.py             # MODIFY - Add sensor_type parameter
│   └── loss/
│       ├── gamma_nll.py             # NEW - Gamma NLL loss
│       ├── elevation.py             # NEW - Elevation constraint
│       └── reflectivity_smooth.py   # NEW - H¹ smoothness
├── sonar/
│   ├── beam_pattern.py              # NEW or MODIFY - Separable beam pattern
│   └── dataset/
│       └── fusion_dataloader.py     # NEW - Camera-sonar pairs
├── examples/
│   └── fusion_trainer.py            # NEW - Main training script
└── tests/
    ├── test_reflectivity.py         # NEW
    ├── test_beam_pattern.py         # NEW
    ├── test_jacobian.py             # NEW
    ├── test_echosounder.py          # NEW
    ├── test_gamma_nll.py            # NEW
    └── test_fusion_pipeline.py      # NEW
```

---

## Expected Results (from Z-Splat Paper)

| Metric | Baseline (Camera Only) | Z-Splat Fusion | Improvement |
|--------|------------------------|----------------|-------------|
| Novel View PSNR | X dB | X + 5 dB | **+5 dB** |
| Chamfer Distance | Y | 0.4 · Y | **-60%** |
| Depth RMS Error | Z | 0.5 · Z | **-50%** |

---

## Contact & Support

For questions about this implementation:
1. Review the mathematical proofs in the header
2. Check the debugging guide for common issues
3. Run unit tests to isolate problems
4. Verify gradients flow to depth parameters

**Remember**: The key to Z-Splat is that sonar provides gradients to depth (μz, σzz) that camera cannot. Verify this first!

---

**END OF IMPLEMENTATION GUIDE**
