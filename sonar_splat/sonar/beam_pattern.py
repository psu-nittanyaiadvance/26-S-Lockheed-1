import math
import torch
from torch import Tensor


def ula_azimuth_gain(theta: Tensor, N_elements: int, d_over_lambda: float = 0.5) -> Tensor:
    """B_a(theta) = |sin(N*pi*d*sin(theta)) / (N*sin(pi*d*sin(theta)))|^2"""
    eps = 1e-8
    psi = math.pi * d_over_lambda * torch.sin(theta)
    numerator = torch.sin(N_elements * psi)
    denominator = N_elements * torch.sin(psi).clamp(min=eps)
    return torch.where(torch.abs(psi) < eps, torch.ones_like(theta), (numerator / denominator) ** 2)


def ula_elevation_gain(phi: Tensor) -> Tensor:
    """B_e(phi) = cos^2(phi)"""
    return torch.cos(phi) ** 2


def ula_beam_pattern(theta: Tensor, phi: Tensor, N_elements: int, d_over_lambda: float = 0.5) -> Tensor:
    """Full B(theta, phi) = B_a(theta) * B_e(phi). Returns [N] in [0,1]."""
    return ula_azimuth_gain(theta, N_elements, d_over_lambda) * ula_elevation_gain(phi)
