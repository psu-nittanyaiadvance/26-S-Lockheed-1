import math
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


def gamma_nll_loss(
    Z_hat: Tensor,
    Z: Tensor,
    K_looks: int = 1,
    eps: float = 1e-6,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Gamma negative log-likelihood loss for coherent sonar imagery.

    Single-look (K=1): per_pixel = Z / (Z_hat + eps) + log(Z_hat + eps)
    K-look:            per_pixel = K*Z/(Z_hat+eps) + K*log(Z_hat+eps) - (K-1)*log(Z+eps)

    Args:
        Z_hat: predicted intensity [H, W] or [N]
        Z:     observed intensity  [H, W] or [N]
        K_looks: number of sonar looks (1 = single-look coherent)
        eps:   numerical stability constant
        mask:  optional boolean mask; loss computed only on True pixels

    Returns:
        scalar mean loss
    """
    Z_hat_s = Z_hat + eps
    Z_s = Z + eps

    if K_looks == 1:
        per_pixel = Z / Z_hat_s + torch.log(Z_hat_s)
    else:
        per_pixel = (
            K_looks * Z / Z_hat_s
            + K_looks * torch.log(Z_hat_s)
            - (K_looks - 1) * torch.log(Z_s)
        )

    if mask is not None:
        masked = per_pixel[mask]
        if masked.numel() == 0:
            return torch.tensor(0.0, device=per_pixel.device, dtype=per_pixel.dtype)
        return masked.mean()
    return per_pixel.mean()


def elevation_loss_metric(
    means: Tensor,
    sonar_image: Tensor,
    viewmat: Tensor,
    max_range: float,
) -> Tensor:
    """
    Soft elevation resolver: penalises Gaussians whose Euclidean range
    disagrees with the peak sonar return along the same azimuth bin.

    Args:
        means:       Gaussian centres in world frame   [N, 3]
        sonar_image: current sonar image (detached)   [H_az, W_range]
        viewmat:     world-to-sensor 4x4 transform    [4, 4]
        max_range:   maximum sensor range in metres

    Returns:
        scalar mean squared range error
    """
    N = means.shape[0]
    H_az, W_range = sonar_image.shape[:2]

    # 1. Project means into sensor frame
    ones = torch.ones(N, 1, device=means.device, dtype=means.dtype)
    means_h = torch.cat([means, ones], dim=1)          # [N, 4]
    local_means = (viewmat @ means_h.T).T[:, :3]       # [N, 3]

    # 2. Euclidean range
    r_n = local_means.norm(dim=-1).clamp(min=1e-7)     # [N]

    # 3. Azimuth angle
    theta_n = torch.atan2(local_means[:, 1], local_means[:, 0])  # [N]

    # 4. Map to azimuth pixel index
    az_idx = ((theta_n.detach() / math.pi + 0.5) * H_az).long().clamp(0, H_az - 1)  # [N]

    # 5. Peak range from sonar image along each azimuth bin
    # Precompute per-az peak to avoid allocating [N, W_range] tensor.
    with torch.no_grad():
        peak_per_az = sonar_image.argmax(dim=-1).float() / W_range * max_range  # [H_az]
        r_sonar_n = peak_per_az[az_idx]  # [N]

    # 6. Mean squared error between predicted and observed range
    return ((r_n - r_sonar_n) ** 2).mean()


def reflectivity_reg(
    r_n: Tensor,
    means: Tensor,
    k_neighbors: int = 5,
    lambda_mean: float = 0.01,
) -> Tensor:
    """
    Spatial smoothness regularizer for per-Gaussian reflectivity r_n.

    Encourages nearby Gaussians (in 3-D) to have similar reflectivity,
    and keeps the mean reflectivity from drifting away from 0.5.

    Args:
        r_n:         per-Gaussian reflectivity in (0,1)  [N]
        means:       Gaussian centres                    [N, 3]
        k_neighbors: number of nearest neighbours
        lambda_mean: weight for mean-reversion term

    Returns:
        scalar regularisation loss
    """
    N = means.shape[0]
    chunk_size = 256
    # Cap at 4096 Gaussians to keep cdist memory bounded regardless of N.
    max_n = 4096
    if N > max_n:
        sub_idx = torch.randperm(N, device=means.device)[:max_n]
        r_n = r_n[sub_idx]
        means = means[sub_idx]
        N = max_n

    with torch.no_grad():
        # Build kNN graph in chunks to avoid OOM
        knn_idx_list = []
        knn_dists2_list = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk = means[start:end]                          # [B, 3]
            dists2 = torch.cdist(chunk, means) ** 2          # [B, N]
            dists2[:, start:end].fill_diagonal_(float("inf"))
            knn_vals, knn_idx = dists2.topk(k_neighbors, largest=False, dim=-1)  # [B, k]
            knn_idx_list.append(knn_idx)
            knn_dists2_list.append(knn_vals)
        knn_idx = torch.cat(knn_idx_list, dim=0)       # [N, k]
        knn_dists2 = torch.cat(knn_dists2_list, dim=0) # [N, k]

    r_neighbors = r_n[knn_idx]                                         # [N, k]
    r_diffs_sq = (r_neighbors - r_n[:, None]) ** 2                     # [N, k]
    smoothness = (r_diffs_sq / knn_dists2.clamp(min=1e-8)).sum()

    mean_rev = lambda_mean * ((r_n - r_n.mean().detach()) ** 2).sum()

    return smoothness + mean_rev
