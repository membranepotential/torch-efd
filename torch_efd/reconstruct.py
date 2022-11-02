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
    )


def _maybe_normalize(tensor: Tensor, normed: bool) -> Tensor:
    if normed:
        tensor /= torch.norm(tensor, dim=-1, keepdim=True)
    return tensor


def reconstruct_contours(efds: Tensor, ts: Tensor | int) -> Tensor:
    if isinstance(ts, int):
        ts = torch.linspace(
            0.0,
            1.0,
            ts,
            dtype=efds.dtype,
            device=efds.device,
        )

    orders = _get_orders_seq(efds)

    phi = ts[..., None] * orders[None, ...] * 2 * torch.pi
    phi = torch.stack([phi.cos(), phi.sin()], dim=-1)
    phi = phi.unsqueeze(-1).expand(*phi.shape, 2)

    efds = efds.reshape(*efds.shape[:-1], 2, 2)
    return torch.einsum("...onm,...komn->...kn", efds, phi)


def derive_tangential_efds(efds: Tensor) -> Tensor:
    orders = _get_orders_seq(efds)
    factor = torch.tensor(
        [1.0, -1.0, 1.0, -1.0],
        dtype=efds.dtype,
        device=efds.device,
    )
    factor *= 2 * torch.pi
    return efds[..., [1, 0, 3, 2]] * orders[:, None] * factor


def reconstruct_tangents(
    efds: Tensor,
    ts: Tensor | int,
    normed: bool = False,
) -> Tensor:
    efds_tan = derive_tangential_efds(efds)
    tangents = reconstruct_contours(efds_tan, ts)
    return _maybe_normalize(tangents, normed)


def reconstruct_normals(efds: Tensor, ts: Tensor | int, normed: bool = False) -> Tensor:
    rot_mat = torch.tensor(
        [[0.0, 1.0], [-1.0, 0.0]], dtype=efds.dtype, device=efds.device
    )
    tangents = reconstruct_tangents(efds, ts)
    normals = tangents @ rot_mat
    return _maybe_normalize(normals, normed)


def reconstruct_tangrads(
    efds: Tensor, ts: Tensor | int, normed: bool = False
) -> Tensor:
    efds_tan = derive_tangential_efds(efds)
    tangrads = reconstruct_tangents(efds_tan, ts)
    return _maybe_normalize(tangrads, normed)


def compute_curvature(efds: Tensor, ts: Tensor | int, signed: bool = False) -> Tensor:
    efds_tan = derive_tangential_efds(efds)
    tangents = reconstruct_contours(efds_tan, ts)
    tangrads = reconstruct_tangents(efds_tan, ts)

    curvature = (
        tangents[..., 0] * tangrads[..., 1] - tangents[..., 1] * tangrads[..., 0]
    )
    curvature /= torch.norm(tangents, dim=-1).pow(3)

    if not signed:
        curvature = curvature.abs()

    return curvature
