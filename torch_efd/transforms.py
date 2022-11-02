import torch
from torch import Tensor


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
