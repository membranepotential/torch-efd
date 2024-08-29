import math
import torch
from torch import Tensor

from .compute import normalize_efds


def scale_efds(efds: Tensor, scale: float | Tensor) -> Tensor:
    """Scale contour."""
    return efds * scale


def rotate_efds(efds: Tensor, angle: float | Tensor) -> Tensor:
    """Rotate contour."""
    angle = torch.as_tensor(angle, dtype=efds.dtype, device=efds.device)

    cospsi = torch.cos(angle)
    sinpsi = torch.sin(angle)
    rot_mat = torch.stack([cospsi, sinpsi, -sinpsi, cospsi], dim=-1)
    rot_mat = rot_mat.reshape((*efds.shape[:-2], 1, 2, 2))

    efds = efds.reshape((*efds.shape[:-1], 2, 2))
    efds = rot_mat @ efds
    return efds.flatten(-2)


def _gaussian_kernel_fft(
    length: int,
    sigma: float,
    device: torch.device | str = "cpu",
) -> Tensor:
    kernel = torch.arange(-length / 2, length / 2, device=device)
    kernel = kernel.pow_(2).div_(-2 * sigma**2).exp_()
    kernel /= sigma * math.sqrt(2 * math.pi)
    return torch.fft.fft(kernel)


def smooth_efds(efds: Tensor, sigma: float, normalize: bool = False) -> Tensor:
    """
    Smoothen contour using Gaussian kernel.

    :param efds: EFDs of shape (..., N, 4)
    :param sigma: Standard deviation of the Gaussian kernel
    :param normalize: If True, normalize the result EFDs
    :return: EFDs of smoothed contour
    """
    kernel = _gaussian_kernel_fft(efds.shape[-2], sigma, device=efds.device)

    efds = torch.view_as_complex(efds.reshape(*efds.shape[:-1], 2, 2))
    efds = efds * kernel.unsqueeze(-1)
    efds = torch.view_as_real(efds).flatten(-2)

    if normalize:
        efds = normalize_efds(efds)

    return efds
