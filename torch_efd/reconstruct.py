"""
Reconstruct contours and other quantities from elliptical fourier descriptors,
in PyTorch and fully diferentiable.

In part adapted from PyEFD (https://github.com/hbldh/pyefd).
"""

import torch
from torch import Tensor


def _get_orders_seq(efds: Tensor) -> Tensor:
    order = efds.shape[-2]
    return torch.arange(
        1,
        order + 1,
        dtype=efds.dtype,
        device=efds.device,
    ).mul_(2 * torch.pi)


def _maybe_normalize(tensor: Tensor, normed: bool) -> Tensor:
    if normed:
        tensor = tensor / torch.linalg.norm(tensor, dim=-1, keepdim=True)
    return tensor


def coerce_ts_vec(efds: Tensor, ts: Tensor | int) -> Tensor:
    if isinstance(ts, int):
        ts = torch.linspace(
            0.0,
            1.0,
            ts,
            dtype=efds.dtype,
            device=efds.device,
        )
    return ts


def reconstruct_efds(efds: Tensor, ts: Tensor | int) -> Tensor:
    """
    Reconstruct contour vertices from EFDs.

    :param efds: EFDs of shape (..., N, 4)
    :param ts: Parameter vector of shape (..., M) or number of vertices M
    :return: Reconstructed vertices of shape (..., M, 2)
    """
    ts = coerce_ts_vec(efds, ts)
    orders = _get_orders_seq(efds)
    phi = ts.unsqueeze(-2) * orders.unsqueeze(-1)

    efds = efds.transpose(-1, -2)
    efds_cos = efds[..., ::2, :] @ phi.cos()
    efds_sin = efds[..., 1::2, :] @ phi.sin()

    return (efds_cos + efds_sin).transpose(-2, -1)


def derive_tangent_efds(efds: Tensor) -> Tensor:
    """
    Derive EFDs of the contour tangential,
    which corresponds to the first derivative or velocity.

    :param efds: Elliptical fourier descriptors of shape (..., N, 4)
    :return: Tangential fourier descriptors of shape (..., N, 4)
    """
    mat = torch.tensor(
        [
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, -1],
            [0, 0, 1, 0],
        ],
        dtype=efds.dtype,
        device=efds.device,
    )
    orders = _get_orders_seq(efds).unsqueeze(-1)
    return efds.mul(orders) @ mat


def derive_normal_efds(efds: Tensor) -> Tensor:
    """
    Derive EFDs of the contour normals, which are orthogonal to the tangents.

    :param efds: Elliptical fourier descriptors of shape (..., N, 4)
    :return: Normal fourier descriptors of shape (..., N, 4)
    """
    mat = torch.tensor(
        [
            [0, 0, 0, -1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
        ],
        dtype=efds.dtype,
        device=efds.device,
    )
    orders = _get_orders_seq(efds).unsqueeze(-1)
    return efds.mul(orders) @ mat


def reconstruct_tangents(
    efds: Tensor,
    ts: Tensor | int,
    normed: bool = False,
) -> Tensor:
    """
    Reconstruct tangents of contour,
    which correspond to the first derivative or velocity.

    :param efds: EFDs of shape (..., N, 4)
    :param ts: Parameter vector of shape (..., M) or number of vertices M
    :param normed: If True, normalize tangents to length 1
    :return: Reconstructed tangents of shape (..., M, 2)
    """
    efds_tan = derive_tangent_efds(efds)
    tangents = reconstruct_efds(efds_tan, ts)
    return _maybe_normalize(tangents, normed)


def reconstruct_normals(
    efds: Tensor,
    ts: Tensor | int,
    normed: bool = False,
) -> Tensor:
    """
    Reconstruct normals of contour.

    :param efds: EFDs of shape (..., N, 4)
    :param ts: Parameter vector of shape (..., M) or number of vertices M
    :param normed: If True, normalize normals to length 1
    :return: Reconstructed normals of shape (..., M, 2)
    """
    efds_norm = derive_normal_efds(efds)
    normals = reconstruct_efds(efds_norm, ts)
    return _maybe_normalize(normals, normed)


def reconstruct_tangrads(
    efds: Tensor,
    ts: Tensor | int,
    normed: bool = False,
) -> Tensor:
    """
    Reconstruct the gradient of the tangents, which corresponds to curvature.

    :param efds: EFDs of shape (..., N, 4)
    :param ts: Parameter vector of shape (..., M) or number of vertices M
    :param normed: If True, normalize normals to length 1
    :return: Reconstructed normals of shape (..., M, 2)
    """
    efds_tan = derive_tangent_efds(efds)
    tangrads = reconstruct_tangents(efds_tan, ts)
    return _maybe_normalize(tangrads, normed)
