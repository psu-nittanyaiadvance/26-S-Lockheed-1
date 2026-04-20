"""
GaussianModelV2: adds per-Gaussian reflectivity r_tilde (sigmoid activation)
on top of the base GaussianModel. Used by train_v2.py.
"""

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.gaussian_model import GaussianModel


class GaussianModelV2(GaussianModel):
    """GaussianModel extended with per-Gaussian reflectivity r_tilde."""

    def __init__(self, sh_degree: int, r_tilde_lr: float = 0.01):
        super().__init__(sh_degree)
        self._r_tilde = torch.empty(0)   # [N, 1], logit-space; sigmoid gives r in (0,1)
        self.r_tilde_lr = r_tilde_lr

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def get_r_tilde(self):
        return torch.sigmoid(self._r_tilde)   # [N, 1]

    # ------------------------------------------------------------------ #
    # Initialisation                                                        #
    # ------------------------------------------------------------------ #

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        super().create_from_pcd(pcd, spatial_lr_scale)
        N = self._xyz.shape[0]
        # Init to 0 → sigmoid(0) = 0.5 (neutral reflectivity)
        self._r_tilde = nn.Parameter(
            torch.zeros((N, 1), dtype=torch.float, device="cuda").requires_grad_(True)
        )

    # ------------------------------------------------------------------ #
    # Optimizer setup                                                       #
    # ------------------------------------------------------------------ #

    def training_setup(self, training_args):
        super().training_setup(training_args)
        # Inject r_tilde into the existing Adam optimizer
        self.optimizer.add_param_group(
            {'params': [self._r_tilde], 'lr': self.r_tilde_lr, "name": "r_tilde"}
        )

    # ------------------------------------------------------------------ #
    # Checkpoint                                                            #
    # ------------------------------------------------------------------ #

    def capture(self):
        base = super().capture()
        return base + (self._r_tilde,)

    def restore(self, model_args, training_args):
        *base_args, r_tilde = model_args
        super().restore(tuple(base_args), training_args)
        self._r_tilde = r_tilde
        # Re-add r_tilde param group (training_setup recreates optimizer)
        for pg in self.optimizer.param_groups:
            if pg["name"] == "r_tilde":
                break
        else:
            self.optimizer.add_param_group(
                {'params': [self._r_tilde], 'lr': self.r_tilde_lr, "name": "r_tilde"}
            )

    # ------------------------------------------------------------------ #
    # Pruning / densification                                               #
    # ------------------------------------------------------------------ #

    def prune_points(self, mask):
        valid_points_mask = ~mask
        # _prune_optimizer already iterates ALL param groups (including r_tilde)
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        self._xyz          = optimizable_tensors["xyz"]
        self._features_dc  = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity      = optimizable_tensors["opacity"]
        self._scaling      = optimizable_tensors["scaling"]
        self._rotation     = optimizable_tensors["rotation"]
        self._r_tilde      = optimizable_tensors["r_tilde"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom     = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest,
                               new_opacities, new_scaling, new_rotation, new_r_tilde=None):
        if new_r_tilde is None:
            new_r_tilde = torch.zeros((new_xyz.shape[0], 1), device="cuda")
        # Pass ALL tensors (including r_tilde) in one call so cat_tensors_to_optimizer
        # finds every param group key — avoids KeyError: 'r_tilde'
        d = {"xyz":     new_xyz,
             "f_dc":    new_features_dc,
             "f_rest":  new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation,
             "r_tilde": new_r_tilde}
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz          = optimizable_tensors["xyz"]
        self._features_dc  = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity      = optimizable_tensors["opacity"]
        self._scaling      = optimizable_tensors["scaling"]
        self._rotation     = optimizable_tensors["rotation"]
        self._r_tilde      = optimizable_tensors["r_tilde"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom     = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_r_tilde = self._r_tilde[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest,
                                    new_opacity, new_scaling, new_rotation, new_r_tilde)

        prune_filter = torch.cat((selected_pts_mask,
                                   torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_r_tilde = self._r_tilde[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest,
                                    new_opacities, new_scaling, new_rotation, new_r_tilde)

    # ------------------------------------------------------------------ #
    # PLY save / load                                                       #
    # ------------------------------------------------------------------ #

    def construct_list_of_attributes(self):
        l = super().construct_list_of_attributes()
        l.append('r_tilde')
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        r_tilde = self._r_tilde.detach().cpu().numpy()          # [N, 1]

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, r_tilde), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path):
        super().load_ply(path)
        plydata = PlyData.read(path)
        if 'r_tilde' in [p.name for p in plydata.elements[0].properties]:
            r_tilde = np.asarray(plydata.elements[0]["r_tilde"])[..., np.newaxis]
            self._r_tilde = nn.Parameter(
                torch.tensor(r_tilde, dtype=torch.float, device="cuda").requires_grad_(True)
            )
        else:
            # Checkpoint without r_tilde — init neutral
            N = self._xyz.shape[0]
            self._r_tilde = nn.Parameter(
                torch.zeros((N, 1), dtype=torch.float, device="cuda").requires_grad_(True)
            )
