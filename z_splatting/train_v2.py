"""
train_v2.py — Z-Splat + Priyanshu's sonar physics math.

Changes vs train.py (baseline):
  1. GaussianModelV2: per-Gaussian reflectivity r_tilde (sigmoid)
  2. ULA beam pattern: modulates effective opacity per camera view
  3. RGB loss: L1+SSIM unchanged (camera noise model)
  4. Z loss: gamma NLL replacing L2 (sonar depth has multiplicative noise)
  5. Elevation constraint loss (annealed)
  6. Reflectivity spatial regularizer (kNN, 4096 subsample)
  7. NaN guards on gradients and parameters after each step
"""

import os
import json
import math
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene
from scene.gaussian_model_v2 import GaussianModelV2
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from gsplat import _sonar_rasterization
    SONAR_RASTERIZER_AVAILABLE = True
except ImportError:
    SONAR_RASTERIZER_AVAILABLE = False

try:
    from rl_loss_controller import RLLossController, make_controller
    RL_CONTROLLER_AVAILABLE = True
except ImportError:
    RL_CONTROLLER_AVAILABLE = False
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


# ======================================================================== #
# Priyanshu's sonar math (same formulas as sonar_simple_trainer_v2.py)     #
# ======================================================================== #

def compute_beam_pattern(means: torch.Tensor, viewmat: torch.Tensor,
                          n_elements: int, d: float, wavelength: float) -> torch.Tensor:
    """Per-Gaussian ULA beam weight in [0,1], shape (N,).
    viewmat: (4,4) world→camera row-major (= world_view_transform.T).
    """
    means_cam = (viewmat[:3, :3] @ means.T + viewmat[:3, 3:4]).T   # (N,3)
    r_n = torch.sqrt((means_cam ** 2).sum(dim=-1).clamp(min=1e-12))
    theta_n = torch.atan2(means_cam[:, 1], means_cam[:, 0])
    phi_n   = torch.asin((means_cam[:, 2] / r_n).clamp(-1.0, 1.0))

    u = (d / wavelength) * torch.sin(theta_n)
    # torch.sinc(x) = sin(pi*x)/(pi*x); well-defined and differentiable at x=0
    B_a = (torch.sinc(n_elements * u) / torch.sinc(u).clamp(min=1e-8)) ** 2
    B_e = torch.cos(phi_n) ** 2
    return (B_a * B_e).clamp(0.0, 1.0)


def gamma_nll_loss(S_measured: torch.Tensor, S_predicted: torch.Tensor,
                   eps: float = 0.01) -> torch.Tensor:
    """MLE loss for multiplicative exponential sonar noise."""
    S_hat = S_predicted.clamp(min=eps)
    return (S_measured / S_hat + torch.log(S_hat)).mean()


def elevation_constraint_loss(xyz: torch.Tensor, r_sonar: torch.Tensor,
                               cam_center: torch.Tensor = None) -> torch.Tensor:
    """L_e = mean_n(||mu_n - cam||^2 - r_sonar_n)^2  (spec Eq. 25).
    cam_center: sonar/camera position in world space (must be detached). Defaults to origin."""
    if cam_center is None:
        cam_center = torch.zeros(3, device=xyz.device)
    predicted_range = torch.sqrt(((xyz - cam_center) ** 2).sum(dim=-1).clamp(min=1e-12))
    return torch.mean((predicted_range - r_sonar) ** 2)


def reflectivity_regularizer_loss(r_n: torch.Tensor, r_neighbors: torch.Tensor,
                                   lambda_reg: float = 0.01) -> torch.Tensor:
    smoothness = torch.mean((r_n.unsqueeze(1) - r_neighbors) ** 2)
    r_bar = r_n.mean().detach()
    mean_reversion = lambda_reg * torch.mean((r_n - r_bar) ** 2)
    return smoothness + mean_reversion


def compute_z_density_diff(
    gaussians,
    eff_opacity: torch.Tensor,
    viewpoint_cam,
    height: int,
    width: int,
    h_res: int,
    w_res: int,
    n_bins: int = 200,
    depth_min: float = 0.0,
    depth_max: float = 8.0,
    depth_scale: float = 1.0,
) -> tuple:
    """Differentiable 1D depth histogram — replaces the non-existent CUDA z_density.

    The diff-gaussian-rasterization submodule in this repo does NOT implement
    z_density (it only returns color+radii). This function computes the same
    quantity entirely in Python using scatter_add, so gradients flow back
    through eff_opacity → r_tilde.

    Physical model
    --------------
    For each Gaussian, its sonar "range" is the Euclidean distance from the
    camera centre (sonar measures slant range, not projected z). This range is
    multiplied by depth_scale to convert from normalised world-space units to
    physical metres so it lines up with the GT bin_edges = linspace(0, 8, 201).

    Depth_scale derivation
    ----------------------
    Caller should pass  depth_scale = sonar_max_range / scene.cameras_extent.
    If cameras are at radius R_world from scene centre (in world units) and the
    sonar sees to R_sonar metres, then 1 world unit ≈ R_sonar / R_world metres.

    Histogram construction
    ----------------------
    Each Gaussian contributes its weight (eff_opacity) to the depth bin that
    matches its range, linearly interpolated between the floor and ceil bins.
    The bin index is non-differentiable (integer), but the weight is — so
    the gradient flows:  loss → z_density → scatter_add → eff_opacity → r_tilde.

    Returns
    -------
    z_density_h : (n_bins, h_res)  normalised depth histogram per height strip
    z_density_w : (n_bins, w_res)  normalised depth histogram per width  strip
    """
    device = eff_opacity.device

    # --- Gaussian positions in world space (geometry detached: Z loss trains
    #     r_tilde only, not xyz/scaling/rotation) ----------------------------
    xyz = gaussians.get_xyz.detach()          # (N, 3)

    # --- Sonar range = Euclidean distance from camera centre ---------------
    # camera_center is the camera position in world space
    cam_center = viewpoint_cam.camera_center.detach()   # (3,)
    range_world = torch.sqrt(((xyz - cam_center) ** 2).sum(-1).clamp(min=1e-12))  # (N,)
    range_m = range_world * depth_scale                  # (N,) in approximate metres

    # --- Project to camera space for height/width strip assignment ---------
    # viewmat = W2C row-major = world_view_transform^T
    viewmat = viewpoint_cam.world_view_transform.T       # (4,4)
    R = viewmat[:3, :3]                                  # (3,3)
    t = viewmat[:3,  3]                                  # (3,)
    means_cam = (R @ xyz.T).T + t                        # (N,3) camera-space

    # Pixel coordinates (approximate pinhole projection) — used only for
    # strip index, NOT for gradient, so depth_z is not in the grad graph
    depth_z = means_cam[:, 2].abs().clamp(min=1e-6)     # (N,) positive cam-z
    fov_x = viewpoint_cam.FoVx
    fov_y = viewpoint_cam.FoVy
    fx = 0.5 * width  / math.tan(0.5 * fov_x)
    fy = 0.5 * height / math.tan(0.5 * fov_y)
    pixel_x = means_cam[:, 0] / depth_z * fx + 0.5 * width   # (N,)
    pixel_y = means_cam[:, 1] / depth_z * fy + 0.5 * height  # (N,)

    # --- Strip indices (integer, non-differentiable) -----------------------
    h_strip = (pixel_y / height * h_res).long().clamp(0, h_res - 1)   # (N,)
    w_strip = (pixel_x / width  * w_res).long().clamp(0, w_res - 1)   # (N,)

    # --- Depth bin via linear interpolation --------------------------------
    # Soft histogram: each Gaussian splits its weight between the floor and
    # ceil depth bins proportionally to its fractional bin position.
    # This makes the histogram differentiable w.r.t. weights (not indices).
    depth_norm = (range_m - depth_min) / (depth_max - depth_min) * n_bins  # (N,)
    bin_lo = depth_norm.long().clamp(0, n_bins - 1)    # (N,) lower bin index
    bin_hi = (bin_lo + 1).clamp(0, n_bins - 1)         # (N,) upper bin index
    alpha  = (depth_norm - bin_lo.float()).clamp(0.0, 1.0)  # (N,) interp weight

    # --- Weights (gradient lives here) -------------------------------------
    weights = eff_opacity.squeeze(-1)                   # (N,) — r_tilde gradient flows

    # --- scatter_add into flat histogram buffers ---------------------------
    # Global flat index: bin_index * n_strips + strip_index
    # scatter_add is differentiable w.r.t. src (weights), not index.

    # Height histogram (n_bins, h_res)
    flat_lo_h = bin_lo * h_res + h_strip                # (N,) int64
    flat_hi_h = bin_hi * h_res + h_strip                # (N,) int64
    z_density_h = torch.zeros(n_bins * h_res, device=device)
    z_density_h = z_density_h.scatter_add(0, flat_lo_h, weights * (1.0 - alpha))
    z_density_h = z_density_h.scatter_add(0, flat_hi_h, weights * alpha)
    z_density_h = z_density_h.view(n_bins, h_res)       # (200, h_res)

    # Width histogram (n_bins, w_res)
    flat_lo_w = bin_lo * w_res + w_strip                # (N,) int64
    flat_hi_w = bin_hi * w_res + w_strip                # (N,) int64
    z_density_w = torch.zeros(n_bins * w_res, device=device)
    z_density_w = z_density_w.scatter_add(0, flat_lo_w, weights * (1.0 - alpha))
    z_density_w = z_density_w.scatter_add(0, flat_hi_w, weights * alpha)
    z_density_w = z_density_w.view(n_bins, w_res)       # (200, w_res)

    # --- Normalise per strip to [0,1] — same as GT normalisation ----------
    z_density_h = z_density_h / (z_density_h.max(dim=0, keepdim=True)[0] + 1e-10)
    z_density_w = z_density_w / (z_density_w.max(dim=0, keepdim=True)[0] + 1e-10)

    return z_density_h, z_density_w


# ======================================================================== #
# Sonar image render (proper L_s from spec — Eq. 28)                        #
# ======================================================================== #

class SonarDataCache:
    """Loads and caches sonar pkl images, matched to COLMAP views by stem name.

    AONeuS layout:
      sonar_data_dir/Data/{000..059}.pkl   — ImagingSonar + PoseSensor
      sonar_data_dir/Config.json           — RangeBins, AzimuthBins, Azimuth, RangeMax

    The pk image (256 range × 96 azimuth) is transposed to (96 azimuth × 256 range),
    normalised to [0,1], and thresholded identically to SonarSensorDataParser.
    """

    def __init__(self, sonar_data_dir: str, sonar_max_range: float,
                 img_threshold: float = 0.0):
        self.data_dir = sonar_data_dir
        self.sonar_max_range = sonar_max_range
        self._cache: dict = {}

        cfg_path = os.path.join(sonar_data_dir, "Config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)["agents"][0]["sensors"][-1]["configuration"]
        self.n_range   = cfg["RangeBins"]          # 256
        self.n_az      = cfg["AzimuthBins"]        # 96
        self.hfov_deg  = cfg["Azimuth"]            # 60.0
        self.hfov_rad  = math.radians(self.hfov_deg)
        self.img_threshold = img_threshold

        print(f"[SonarDataCache] dir={sonar_data_dir}")
        print(f"  sonar: {self.n_az} az × {self.n_range} rng bins, "
              f"hfov={self.hfov_deg}°, max_range={sonar_max_range:.2f} m")

    def _load(self, pkl_path: str) -> torch.Tensor:
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
        img = d["ImagingSonar"]
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        else:
            img = img.astype(np.float32)
        img = img.transpose(1, 0)           # (range,az) → (az,range) = (H,W)
        img = np.clip(img, 0.0, 1.0)
        img[img < self.img_threshold] = 0.0
        img[:, 0:10]  = 0.0                 # edge noise
        img[:, -10:]  = 0.0
        img[0:10,  :] = 0.0
        img[-10:,  :] = 0.0
        return torch.from_numpy(img).float().cuda()  # (n_az, n_range)

    def get(self, image_name: str):
        """Return sonar tensor (n_az, n_range) for the given COLMAP image name.
        Returns None if no matching pkl file exists."""
        stem = os.path.splitext(os.path.basename(image_name))[0]  # e.g. '000'
        if stem in self._cache:
            return self._cache[stem]
        pkl_path = os.path.join(self.data_dir, "Data", stem + ".pkl")
        if not os.path.exists(pkl_path):
            return None
        tensor = self._load(pkl_path)
        self._cache[stem] = tensor
        return tensor


def render_sonar_image(gaussians, viewpoint_cam, sonar_Ks: torch.Tensor,
                       n_az: int, n_range: int, max_range_wu: float,
                       n_array_elements: int, element_spacing: float,
                       wavelength: float, beam_alpha: float = 1.0) -> torch.Tensor:
    """Render a sonar intensity image via _sonar_rasterization (gsplat).

    Uses the COLMAP camera pose so Gaussian positions (in COLMAP world space)
    are correctly projected to (azimuth, range) bins.

    Returns render_powers: (n_az, n_range) tensor — same shape as sonar GT image.
    """
    if not SONAR_RASTERIZER_AVAILABLE:
        raise RuntimeError("gsplat._sonar_rasterization not available. "
                           "Install gsplat or use --sonar_data_dir '' to disable.")

    N = gaussians.get_xyz.shape[0]
    device = gaussians.get_xyz.device

    # World→camera (column-major) from COLMAP camera
    # world_view_transform is W2C in row-major; .T gives column-major W2C
    w2c = viewpoint_cam.world_view_transform.T.unsqueeze(0)   # (1, 4, 4)

    # Beam pattern × r_tilde × opacity — the effective sonar opacity per Gaussian
    viewmat = viewpoint_cam.world_view_transform.T             # (4, 4)
    beam_w_physics = compute_beam_pattern(gaussians.get_xyz.detach(), viewmat,
                                          n_array_elements, element_spacing, wavelength)
    beam_w = beam_alpha * beam_w_physics + (1.0 - beam_alpha)
    opacities = (gaussians.get_opacity.squeeze(-1)              # (N,) — ∇L_sonar reaches opacity ✓
                 * gaussians.get_r_tilde.squeeze(-1)           # (N,) — ∇L_sonar reaches r_tilde ✓
                 * beam_w)                                     # (N,) no_grad (beam is deterministic)

    # SH colour coefficients (same format as gsplat: [N, K, 3])
    colors = torch.cat([gaussians._features_dc,
                        gaussians._features_rest], dim=1)      # (N, K, 3)

    # sat_probability: not modelled in GaussianModelV2 — use neutral 0.5
    sat_prob = torch.full((N,), 0.5, device=device)

    Ks_batched = sonar_Ks.unsqueeze(0)                         # (1, 3, 3)

    render_powers, _ = _sonar_rasterization(
        means       = gaussians.get_xyz,
        quats       = gaussians.get_rotation,          # (N, 4) wxyz
        scales      = gaussians.get_scaling,           # (N, 3) already exp-activated
        opacities   = opacities,                       # (N,)
        sat_probability = sat_prob,
        colors      = colors,
        viewmats    = w2c,                             # (1, 4, 4)
        Ks          = Ks_batched,                      # (1, 3, 3)
        width       = n_range,
        height      = n_az,
        max_range   = max_range_wu,
        sh_degree   = gaussians.active_sh_degree,
        near_plane  = 0.01,
        far_plane   = max_range_wu * 1.2,
        sph         = True,
    )
    return render_powers.squeeze(0).squeeze(-1)  # → (n_az, n_range)


# ======================================================================== #
# Training                                                                   #
# ======================================================================== #

def training(dataset, opt, pipe, testing_iterations, saving_iterations,
             checkpoint_iterations, checkpoint, debug_from, args):

    # --- sonar physics constants ----------------------------------------
    speed_of_sound   = args.speed_of_sound
    bandwidth        = args.bandwidth
    n_array_elements = args.n_array_elements
    element_spacing  = args.element_spacing
    center_frequency = args.center_frequency
    wavelength       = speed_of_sound / center_frequency
    sigma_r          = speed_of_sound / (4.0 * bandwidth)
    print(f"sigma_r = {sigma_r * 100:.2f} cm  |  wavelength = {wavelength * 1000:.2f} mm")

    w_e              = args.w_e
    w_e_final        = args.w_e_final
    w_e_anneal_steps = args.w_e_anneal_steps
    beam_anneal_steps = getattr(args, 'beam_anneal_steps', 0)  # 0 = no annealing (full beam from start)
    refl_reg_weight  = args.reflectivity_reg_weight
    lambda_reg       = args.lambda_reg
    refl_reg_every   = args.reflectivity_reg_every
    z_loss_weight      = args.z_loss_weight        # kept for histogram-fallback compat
    camera_loss_weight = getattr(args, 'camera_loss_weight', 0.3)   # w_c in spec total loss

    # --- RL loss controller (optional) ----------------------------------
    rl_ctrl = None
    if RL_CONTROLLER_AVAILABLE and getattr(args, 'use_rl_controller', False):
        rl_ctrl = make_controller(args, max_steps=opt.iterations)
        print(f"[RLCtrl] Active — initial z_w={rl_ctrl.z_w:.4f}  cam_w={rl_ctrl.cam_w:.4f}")
    else:
        print(f"[RLCtrl] Disabled — fixed z_w={z_loss_weight}  cam_w={camera_loss_weight}")

    # --- model & scene --------------------------------------------------
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModelV2(dataset.sh_degree, r_tilde_lr=args.r_tilde_lr)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    # depth_scale converts world-space units → physical metres so that the
    # predicted depth histogram lines up with the GT bin_edges (0–8 m, 200 bins).
    #
    # Auto-calibration: use median nonzero sonar depth from depth.npy files (in
    # physical metres) divided by the mean camera-to-scene distance (in world
    # units).  This is robust to scenes where cameras_extent is NOT equal to the
    # camera-to-scene distance (e.g. AONeuS, where cameras orbit at z=-2.25 wu
    # but cameras_extent ≈ 0.2 wu — the spread only, not the distance).
    #
    # Explicit override: pass --depth_scale > 0 to bypass auto-calibration.
    if args.depth_scale > 0.0:
        depth_scale = args.depth_scale
        print(f"depth_scale = {depth_scale:.3f}  (explicit override via --depth_scale)")
    else:
        depth_dir = os.path.join(dataset.source_path, "depth")
        auto_scale = None
        if os.path.exists(depth_dir):
            depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.npy')])[:5]
            nz_all = []
            for fname in depth_files:
                d = np.load(os.path.join(depth_dir, fname))
                nz = d[d > 0].ravel()
                nz_all.extend(nz.tolist())
            if nz_all:
                median_depth_m = float(np.median(nz_all))
                cam_np = np.array([c.camera_center.detach().cpu().numpy()
                                   for c in scene.getTrainCameras()])
                avg_cam_dist_wu = float(np.linalg.norm(cam_np, axis=1).mean())
                if avg_cam_dist_wu > 0.1:
                    auto_scale = median_depth_m / avg_cam_dist_wu
                    print(f"depth_scale = {auto_scale:.3f}  "
                          f"(auto: median_depth={median_depth_m:.3f} m / "
                          f"avg_cam_dist={avg_cam_dist_wu:.3f} wu)")
        if auto_scale is None:
            auto_scale = args.sonar_max_range / max(scene.cameras_extent, 1e-6)
            print(f"depth_scale = {auto_scale:.3f}  "
                  f"(fallback: sonar_max_range={args.sonar_max_range} m / "
                  f"cameras_extent={scene.cameras_extent:.3f} wu)")
        depth_scale = auto_scale

    # --- sonar image render (L_s from spec Eq. 28) -------------------------
    # Initialise if --sonar_data_dir is provided and gsplat render is available.
    sonar_cache  = None
    sonar_Ks     = None
    n_az = n_range = 0
    max_range_wu = args.sonar_max_range / depth_scale   # metres → COLMAP world units

    if getattr(args, 'sonar_data_dir', '') and SONAR_RASTERIZER_AVAILABLE:
        sonar_cache = SonarDataCache(args.sonar_data_dir, args.sonar_max_range)
        n_az        = sonar_cache.n_az       # 96  (azimuth bins = render height)
        n_range     = sonar_cache.n_range    # 256 (range bins = render width)
        hfov_rad    = sonar_cache.hfov_rad   # 60° total field of view

        # --- Bug 3 startup verification: confirm pkl files exist and load correctly ---
        _data_subdir = os.path.join(args.sonar_data_dir, "Data")
        _pkl_dir = _data_subdir if os.path.isdir(_data_subdir) else args.sonar_data_dir
        _pkl_files = sorted([f for f in os.listdir(_pkl_dir) if f.endswith('.pkl')])
        if len(_pkl_files) == 0:
            raise RuntimeError(f"[SonarDataCache] No .pkl files found in {_pkl_dir}. "
                               f"Check --sonar_data_dir path.")
        print(f"[SonarDataCache] Found {len(_pkl_files)} sonar pkl files in {_pkl_dir}")
        with open(os.path.join(_pkl_dir, _pkl_files[0]), 'rb') as _f:
            _sample = pickle.load(_f)
        _sample_img = _sample["ImagingSonar"]
        print(f"[SonarDataCache] Sample {_pkl_files[0]}: "
              f"shape={_sample_img.shape}, dtype={_sample_img.dtype}, "
              f"max={float(_sample_img.max()):.4f}")
        if float(_sample_img.max()) <= 0.0:
            raise RuntimeError("[SonarDataCache] Sample sonar image is all zeros — "
                               "wrong path or corrupt data.")
        del _sample, _sample_img, _pkl_files
        # --- end startup verification ---

        # Sonar intrinsics in COLMAP world-unit space (same as _sonar_rasterization
        # expects — no physical-metre conversion; depth_scale handles the mapping).
        #   K[0,0]: bins per wu  (range axis)
        #   K[1,1]: bins per rad (azimuth axis)
        #   K[1,2]: azimuth centre bin
        rng_res_wu  = max_range_wu / n_range          # wu per range bin
        az_res_rad  = hfov_rad     / n_az             # rad per azimuth bin
        sonar_Ks = torch.tensor([
            [1.0 / rng_res_wu,        0.0,              0.0],
            [0.0,          1.0 / az_res_rad,   n_az / 2.0],
            [0.0,                     0.0,              1.0],
        ], dtype=torch.float32, device="cuda")
        print(f"[sonar render] max_range_wu={max_range_wu:.3f} wu, "
              f"K[0,0]={sonar_Ks[0,0].item():.1f} bins/wu, "
              f"K[1,1]={sonar_Ks[1,1].item():.1f} bins/rad")
    elif getattr(args, 'sonar_data_dir', '') and not SONAR_RASTERIZER_AVAILABLE:
        print("[WARNING] --sonar_data_dir given but gsplat._sonar_rasterization "
              "not available — falling back to Z-density histogram.")
    else:
        if not getattr(args, 'sonar_data_dir', ''):
            print("[WARNING] --sonar_data_dir not set. Using depth histogram proxy (weaker supervision).")

    # Change 1 sanity check: print active sonar supervision path at startup
    sonar_path_mode = "full_render" if sonar_cache is not None else "debug_histogram_proxy"
    print(f"SONAR_PATH={sonar_path_mode}")
    print(f"Loss weights: L_sonar(1.0) + w_c({camera_loss_weight:.2f})*L_camera "
          f"+ w_e({w_e:.1f}→{w_e_final:.1f}@{w_e_anneal_steps})*L_elev "
          f"[beam_anneal={beam_anneal_steps}] "
          f"+ w_r({refl_reg_weight})*L_reg")

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end   = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log_rgb = 0.0
    ema_loss_for_log_z   = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # kNN cache for reflectivity regularizer
    _knn_source  = None
    _knn_indices = None

    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # ---- ground-truth data ----------------------------------------
        gt_image      = viewpoint_cam.original_image
        gt_density_h  = viewpoint_cam.z_density_h if viewpoint_cam.z_density_h is not None else None
        gt_density_w  = viewpoint_cam.z_density_w if viewpoint_cam.z_density_w is not None else None
        height        = gt_image.shape[1]
        width         = gt_image.shape[2]

        # ---- beam-weighted effective opacity (sonar Z path only) --------
        # world_view_transform = W2C.T  →  .T gives W2C row-major
        viewmat = viewpoint_cam.world_view_transform.T   # (4,4)
        # Beam annealing: interpolate flat→physics over beam_anneal_steps.
        # Prevents near-zero opacities from a very narrow beam (large N) starving gradients.
        beam_alpha = 1.0 if beam_anneal_steps == 0 else min(1.0, iteration / max(beam_anneal_steps, 1))
        with torch.no_grad():
            beam_w_physics = compute_beam_pattern(
                gaussians.get_xyz, viewmat,
                n_array_elements, element_spacing, wavelength
            )  # (N,)
            beam_w = beam_alpha * beam_w_physics + (1.0 - beam_alpha)  # flat=1 when beam_alpha=0
        # Sonar effective opacity for histogram fallback: opacity × r_tilde × beam
        # Both opacity and r_tilde receive ∇L_sonar so the sonar loss can adapt
        # which Gaussians are acoustically visible (opacity) as well as their
        # reflectivity (r_tilde). Freezing opacity here was confirmed to sever the
        # gradient path in practice — r_tilde cannot learn if opacity is stuck at
        # the RGB-trained value when that value makes the Gaussian acoustically invisible.
        eff_opacity = (gaussians.get_opacity                        # (N,1) — ∇L_sonar reaches opacity ✓
                       * gaussians.get_r_tilde                      # (N,1) — ∇L_sonar reaches r_tilde ✓
                       * beam_w.unsqueeze(-1))                      # (N,1) — no_grad (beam deterministic)

        # ---- RGB render: standard opacity (camera noise model) ---------
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter      = render_pkg["visibility_filter"]
        radii                  = render_pkg["radii"]

        # ---- L_camera (geometric/photometric regularizer; sonar is primary) --
        Ll1  = l1_loss(image, gt_image)
        # Skip SSIM when camera_loss_weight=0 — avoids GPU crash if sonar NaN corrupts memory
        if camera_loss_weight > 0 and opt.lambda_dssim > 0:
            L_cam = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        else:
            L_cam = Ll1

        # ---- L_s: sonar image render (spec Eq. 28, primary sonar loss) ----
        # Prefer the proper sonar alpha-compositing render when sonar_data_dir
        # is given (uses _sonar_rasterization from gsplat, matches the paper spec
        # exactly: S_hat rendered from Gaussians, Gamma NLL vs measured S).
        # Falls back to Z-density histogram when no sonar images are available.
        ZL = torch.zeros(1, dtype=torch.float32, device="cuda")
        sonar_gt = None
        if sonar_cache is not None:
            sonar_gt = sonar_cache.get(viewpoint_cam.image_name)
        if sonar_gt is not None and opt.depth_loss:
            # Proper sonar render: S_hat via alpha-compositing with T_n transmittance,
            # beam pattern, and r_tilde reflectivity — exactly the spec formulation.
            sonar_pred = render_sonar_image(
                gaussians, viewpoint_cam, sonar_Ks,
                n_az, n_range, max_range_wu,
                n_array_elements, element_spacing, wavelength,
                beam_alpha=beam_alpha,
            )
            if torch.isnan(sonar_pred).any() or torch.isinf(sonar_pred).any():
                print(f"[WARNING] NaN/Inf in sonar_pred at iter {iteration} — skipping sonar loss", flush=True)
                ZL = torch.zeros(1, dtype=torch.float32, device="cuda")
            else:
                ZL = gamma_nll_loss(sonar_gt, sonar_pred)
        elif gt_density_h is not None and gt_density_w is not None and opt.depth_loss:
            # Fallback: Z-density histogram (no sonar images available)
            z_density_h, z_density_w = compute_z_density_diff(
                gaussians, eff_opacity, viewpoint_cam,
                height, width, dataset.h_res, dataset.w_res,
                n_bins=200, depth_min=0.0, depth_max=8.0,
                depth_scale=depth_scale,
            )
            ZL_h = gamma_nll_loss(gt_density_h.float(), z_density_h)
            ZL_w = gamma_nll_loss(gt_density_w.float(), z_density_w)
            ZL   = (ZL_h * dataset.h_res + ZL_w * dataset.w_res) / (dataset.h_res + dataset.w_res)

        # ---- elevation constraint (annealed) — anchored to sonar GT -----
        # Spec Change 5 (Eq. 25): L_e = sum_n (||mu_n|| - r_sonar_n)^2
        # r_sonar_n is the range bin of peak sonar return in a 3*sigma_r window
        # around each Gaussian's predicted range — from the actual sonar image.
        t_frac = min(iteration, w_e_anneal_steps) / w_e_anneal_steps
        w_e_cur = w_e * (w_e_final / w_e) ** t_frac
        xyz = gaussians.get_xyz
        cam_c = viewpoint_cam.camera_center.detach()   # sonar slant range measured from sensor
        if sonar_gt is not None:
            with torch.no_grad():
                # range profile: mean over azimuth → (n_range,)
                range_profile = sonar_gt.mean(dim=0)
                r_bin_center = (
                    torch.norm(xyz.detach() - cam_c, dim=-1) / max_range_wu * n_range
                ).long().clamp(0, n_range - 1)               # (N,) camera-relative slant range
                sigma_r_wu = sigma_r / depth_scale
                w_bins = max(1, int(3 * sigma_r_wu / max_range_wu * n_range))
                offsets   = torch.arange(-w_bins, w_bins + 1, device=xyz.device)
                win_idx   = (r_bin_center.unsqueeze(1) + offsets.unsqueeze(0)
                             ).clamp(0, n_range - 1)          # (N, 2w+1)
                windows   = range_profile[win_idx]             # (N, 2w+1)
                peak_off  = windows.argmax(dim=1) - w_bins     # (N,)
                peak_bin  = (r_bin_center + peak_off).clamp(0, n_range - 1).float()
                r_sonar   = peak_bin / n_range * max_range_wu  # COLMAP units (N,)
        else:
            # No sonar GT — elevation constraint has no anchor (no-op by design)
            with torch.no_grad():
                r_sonar = torch.norm(xyz.detach() - cam_c, dim=-1)  # camera-relative (no-op target)
        elev_loss = elevation_constraint_loss(xyz, r_sonar, cam_c)

        # ---- reflectivity regularizer (every N steps) ------------------
        if iteration % refl_reg_every == 0:
            with torch.no_grad():
                pts   = gaussians.get_xyz.detach()
                N_gs  = pts.shape[0]
                max_p = min(N_gs, 4096)
                perm  = torch.randperm(N_gs, device=pts.device)[:max_p]
                dists = torch.cdist(pts[perm], pts[perm])
                k_nn  = min(8, max_p - 1)
                _, idx = dists.topk(k_nn + 1, dim=-1, largest=False)
                _knn_source  = perm
                _knn_indices = perm[idx[:, 1:]]

        refl_reg = torch.tensor(0.0, device="cuda")
        if _knn_source is not None and _knn_source.max() < gaussians.get_r_tilde.shape[0]:
            r_n_all  = gaussians.get_r_tilde                        # (N,1)
            r_n_sub  = r_n_all[_knn_source]                         # (M,1)
            r_n_nbrs = r_n_all[_knn_indices]                        # (M,k,1)
            refl_reg = reflectivity_regularizer_loss(r_n_sub, r_n_nbrs, lambda_reg)

        # ---- RL controller: update dynamic weights each step ----------------
        if rl_ctrl is not None:
            _r_mean = gaussians.get_r_tilde.mean().item()
            z_loss_weight, camera_loss_weight = rl_ctrl.step(
                ZL.item(), L_cam.item(), _r_mean, iteration)

        # ---- Assemble total loss — spec: L = L_sonar + w_c*L_cam + w_e*L_elev + w_r*L_reg
        loss = (z_loss_weight * ZL
                + camera_loss_weight * L_cam
                + w_e_cur * elev_loss
                + refl_reg_weight * refl_reg)

        # ---- backward ----------------------------------------------------
        loss.backward()

        nan_grad = False
        for pg in gaussians.optimizer.param_groups:
            for param in pg["params"]:
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    param.grad.zero_()
                    nan_grad = True
        if nan_grad:
            print(f"[WARNING] NaN/Inf gradient zeroed at iter {iteration}", flush=True)

        # Gradient diagnostics — log gradient norms every 500 iterations.
        # r_tilde should receive nonzero gradient whenever ZL > 0.
        def _gn(p): return p.grad.norm().item() if (p.grad is not None) else 0.0
        if iteration % 500 == 0:
            print(f"[iter {iteration}] ∇norms: "
                  f"xyz={_gn(gaussians._xyz):.3e}  "
                  f"r_tilde={_gn(gaussians._r_tilde):.3e}  "
                  f"opacity={_gn(gaussians._opacity):.3e}  "
                  f"f_dc={_gn(gaussians._features_dc):.3e}  "
                  f"| r_tilde.mean={gaussians.get_r_tilde.mean().item():.4f}", flush=True)
        if iteration % 500 == 0 and sonar_gt is not None:
            print(f"[iter {iteration}] sonar_gt: shape={tuple(sonar_gt.shape)}  "
                  f"max={sonar_gt.max().item():.4f}  nonzero={sonar_gt.nonzero().shape[0]}", flush=True)

        # Clip r_tilde SEPARATELY from the RGB/geometry parameters.
        # Rationale: r_tilde receives gradient from the Z (gamma NLL) loss.
        # If clipped together, a large Z gradient reduces the RGB gradient budget,
        # which was observed to drop test PSNR by ~2 dB.  Independent clipping
        # lets both loss paths use their full max_norm=1.0 budget.
        _rgb_params = [p for pg in gaussians.optimizer.param_groups
                       if pg["name"] != "r_tilde" for p in pg["params"]]
        _zt_params  = [p for pg in gaussians.optimizer.param_groups
                       if pg["name"] == "r_tilde" for p in pg["params"]]
        torch.nn.utils.clip_grad_norm_(_rgb_params, max_norm=1.0)
        if _zt_params:
            torch.nn.utils.clip_grad_norm_(_zt_params, max_norm=1.0)

        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log_rgb = 0.4 * Ll1.item() + 0.6 * ema_loss_for_log_rgb
            ema_loss_for_log_z   = 0.4 * ZL.item()  + 0.6 * ema_loss_for_log_z
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "RGB": f"{ema_loss_for_log_rgb:.5f}",
                    "Z":   f"{ema_loss_for_log_z:.5f}",
                    "r":   f"{gaussians.get_r_tilde.mean().item():.3f}",
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            _last_test_psnr = training_report(
                tb_writer, iteration, Ll1, ZL, loss, l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations, scene, render, (pipe, background))

            # === Changes 9+11: per-component loss breakdown + sonar dual eval ===
            if iteration in testing_iterations:
                print(f"[ITER {iteration}] Loss breakdown | "
                      f"ZL={ZL.item():.5f}  "
                      f"L_cam={L_cam.item():.5f}  "
                      f"elev={elev_loss.item():.5f}  "
                      f"reg={refl_reg.item():.5f}  "
                      f"total={loss.item():.5f}", flush=True)

                if sonar_cache is not None:
                    sonar_nll_frames, sonar_pred_0, sonar_gt_0 = [], None, None
                    for _i, vc in enumerate(scene.getTestCameras()):
                        sg = sonar_cache.get(vc.image_name)
                        if sg is None:
                            continue
                        sp = render_sonar_image(
                            gaussians, vc, sonar_Ks, n_az, n_range, max_range_wu,
                            n_array_elements, element_spacing, wavelength,
                            beam_alpha=1.0,  # always use full beam at eval time
                        )
                        sonar_nll_frames.append(gamma_nll_loss(sg, sp).item())
                        if sonar_pred_0 is None:
                            sonar_pred_0, sonar_gt_0 = sp, sg
                    if sonar_nll_frames:
                        avg_snll = float(np.mean(sonar_nll_frames))
                        print(f"[ITER {iteration}] Sonar test Gamma-NLL: "
                              f"{avg_snll:.4f}  ({len(sonar_nll_frames)} frames)", flush=True)
                        if tb_writer:
                            tb_writer.add_scalar('test/sonar_gamma_nll', avg_snll, iteration)
                        # Feed sparse reward to RL controller (PSNR read from training_report return)
                        if rl_ctrl is not None and _last_test_psnr is not None:
                            rl_ctrl.checkpoint_reward(_last_test_psnr, avg_snll, iteration)
                    if sonar_pred_0 is not None:
                        out_npz = os.path.join(dataset.model_path,
                                               f"sonar_eval_iter_{iteration}.npz")
                        np.savez(out_npz,
                                 pred=sonar_pred_0.cpu().numpy(),
                                 gt=sonar_gt_0.cpu().numpy())
                        print(f"[ITER {iteration}] Saved {out_npz}", flush=True)
                    torch.cuda.empty_cache()

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005,
                                                scene.cameras_extent, size_threshold)

                if (iteration % opt.opacity_reset_interval == 0 or
                        (dataset.white_background and iteration == opt.densify_from_iter)):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

                # NaN parameter guard
                for param in [gaussians._xyz, gaussians._opacity, gaussians._r_tilde]:
                    if torch.isnan(param.data).any():
                        param.data = torch.nan_to_num(param.data, nan=0.0)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration),
                           scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    # Save RL policy after training so cross-run learning accumulates
    if rl_ctrl is not None:
        rl_ctrl.save(os.path.join(dataset.model_path, "rl_policy.pt"))


# ======================================================================== #
# Helpers                                                                    #
# ======================================================================== #

def prepare_output_and_logger(args):
    if not args.model_path:
        unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.model_path = os.path.join(args.model_path, timestamp)
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as f:
        f.write(str(Namespace(**vars(args))))
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    return tb_writer


def training_report(tb_writer, iteration, Ll1, ZL, loss, l1_loss, elapsed,
                    testing_iterations, scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/rgb_loss',   Ll1.item(),  iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/z_loss',     ZL.item(),   iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr_out = None   # returned so caller can pass to RL controller
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        print(f"\n[ITER {iteration}] Z-loss: {ZL.item():.6f}")
        validation_configs = (
            {'name': 'test',  'cameras': scene.getTestCameras()},
            {'name': 'train', 'cameras': [scene.getTrainCameras()[i % len(scene.getTrainCameras())]
                                           for i in range(5, 30, 5)]},
        )
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image    = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and idx < 5:
                        tb_writer.add_images(config['name'] + f"_view_{viewpoint.image_name}/render",
                                             image[None], global_step=iteration)
                    l1_test   += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test   /= len(config['cameras'])
                print(f"\n[ITER {iteration}] Evaluating {config['name']}: "
                      f"L1 {l1_test} PSNR {psnr_test}")
                if config['name'] == 'test':
                    test_psnr_out = float(psnr_test)
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss',  l1_test,   iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr',     psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
    return test_psnr_out


# ======================================================================== #
# Entry point                                                                #
# ======================================================================== #

if __name__ == "__main__":
    parser = ArgumentParser(description="train_v2: Z-Splat + sonar physics math")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip',    type=str, default="127.0.0.1")
    parser.add_argument('--port',  type=int, default=6009)
    parser.add_argument('--debug_from',   type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations",       nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations",       nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet",                 action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint",      type=str,  default=None)

    # Sonar physics
    parser.add_argument("--speed_of_sound",        type=float, default=1500.0)
    parser.add_argument("--bandwidth",             type=float, default=30_000.0)
    parser.add_argument("--n_array_elements",      type=int,   default=64)
    parser.add_argument("--element_spacing",       type=float, default=0.003)
    parser.add_argument("--center_frequency",      type=float, default=1.1e6)
    parser.add_argument("--r_tilde_lr",            type=float, default=0.01)
    parser.add_argument("--w_e",                   type=float, default=1.0)
    parser.add_argument("--w_e_final",             type=float, default=0.1)
    parser.add_argument("--w_e_anneal_steps",      type=int,   default=10_000)
    parser.add_argument("--beam_anneal_steps",     type=int,   default=0,
                        help="Anneal beam from flat (no directivity) → full physics ULA over N steps. "
                             "Use >0 for large arrays (N>64) where narrow beam starves gradients early. "
                             "0 = full physics beam from iter 0 (default, safe for N<=64).")
    parser.add_argument("--reflectivity_reg_weight", type=float, default=0.1)
    parser.add_argument("--lambda_reg",            type=float, default=0.01)
    parser.add_argument("--reflectivity_reg_every",type=int,   default=100)
    # Z density (depth histogram) loss
    # sonar_max_range: physical max range of the sonar in metres (used to
    #   convert world-space units to metres for depth histogram binning).
    #   For AONeuS dataset this is 5.0 m (see create_valid_z_splat_scene.py).
    # z_loss_weight: scalar multiplier on the Z gamma-NLL loss term.
    #   Set to 0 to disable Z loss entirely.  Start small (0.01–0.1) while
    #   tuning depth_scale; increase once r_tilde starts moving.
    parser.add_argument("--sonar_max_range",       type=float, default=5.0,
                        help="Sonar max range in metres (AONeuS=5.0)")
    parser.add_argument("--depth_scale",           type=float, default=0.0,
                        help="World→metres conversion factor.  0 = auto-calibrate from "
                             "depth.npy median + avg camera distance (recommended).  "
                             "Pass a positive value to override (AONeuS empirical ≈ 1.38).")
    parser.add_argument("--z_loss_weight",         type=float, default=0.1,
                        help="Weight on gamma-NLL Z histogram fallback loss (kept for compat; "
                             "not used in sonar-primary loss assembly when sonar_data_dir is set)")
    parser.add_argument("--camera_loss_weight",    type=float, default=0.3,
                        help="w_c: weight on camera L1+SSIM loss. Sonar loss has implicit weight "
                             "1.0 (primary). Spec default: 0.3.")
    parser.add_argument("--sonar_data_dir",        type=str,   default="",
                        help="Path to sonar pkl data dir (e.g. .../reduced_baseline_0.6x_sonar). "
                             "When set, enables proper sonar image render (L_s from spec Eq. 28) "
                             "instead of the Z-density histogram fallback. "
                             "Requires gsplat._sonar_rasterization.")
    # RL loss controller
    parser.add_argument("--use_rl_controller",     action="store_true", default=False,
                        help="Enable PPO-based adaptive loss weight controller. "
                             "Dynamically adjusts z_loss_weight and camera_loss_weight each "
                             "adapt step to balance sonar vs camera loss contributions. "
                             "Saves policy to <model_path>/rl_policy.pt for cross-run learning.")
    parser.add_argument("--rl_target_ratio",       type=float, default=1.0,
                        help="Target |ZL×z_w| / |L_cam×cam_w| ratio for RL balance reward. "
                             "1.0 = equal contributions. Lower = favour camera.")
    parser.add_argument("--rl_adapt_every",        type=int,   default=200,
                        help="How many training iters between RL weight updates.")
    parser.add_argument("--rl_auto_init_steps",    type=int,   default=100,
                        help="Iterations to observe loss scales before auto-calibrating "
                             "initial weights. Makes the controller dataset-agnostic.")
    parser.add_argument("--rl_policy_path",        type=str,   default="",
                        help="Shared policy path for cross-dataset learning. "
                             "If set, policy is loaded from and saved to this path instead "
                             "of <model_path>/rl_policy.pt. Allows one policy file to "
                             "accumulate experience across multiple datasets and runs.")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    test_iterations = range(5000, args.iterations + 1, 3000)
    args.test_iterations = [i for i in test_iterations if i < args.iterations]
    args.test_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(lp.extract(args), op.extract(args), pp.extract(args),
             args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint,
             args.debug_from, args)

    print("\nTraining complete.")
