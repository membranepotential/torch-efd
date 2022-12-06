import math
import torch
from torch import Tensor

from .compute import normalize_efd


def scale_efds(efds: Tensor, scale: float | Tensor) -> Tensor:
    """Scale elliptical fourier descriptors."""
    return efds * scale


def rotate_efds(efds: Tensor, radians: float | Tensor) -> Tensor:
    """Rotate elliptical fourier descriptors."""
    radians = torch.as_tensor(radians)

    efds = efds.reshape((*efds.shape[:-1], 2, 2))

    cospsi = torch.cos(radians)
    sinpsi = torch.sin(radians)
    rot_mat = torch.stack([cospsi, sinpsi, -sinpsi, cospsi], dim=-1)
    rot_mat = rot_mat.reshape((*efds.shape[:-3], 1, 2, 2))

    efds = rot_mat @ efds
    return efds.flatten(-2)


def _gaussian_kernel_fft(
    length: int,
    sigma: float,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    kernel = torch.arange(-length / 2, length / 2, device=device)
    sig2 = -2.0 * sigma**2
    kernel = kernel.pow_(2.0).div_(sig2).exp_().div_(sigma * math.sqrt(2 * math.pi))
    return torch.fft.fft(kernel)


def smooth_efds(efds: Tensor, sigma: float, normalize: bool = False) -> Tensor:
    kernel = -_gaussian_kernel_fft(efds.shape[-2], sigma, device=efds.device)

    efds = torch.view_as_complex(efds.reshape(*efds.shape[:-1], 2, 2))
    efds = efds * kernel.unsqueeze(-1)
    efds = torch.view_as_real(efds).flatten(-2)

    if normalize:
        efds = normalize_efd(efds)

    return efds
