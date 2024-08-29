"""
Compute and normalize elliptical fourier descriptors for a contour in PyTorch.
Adapted from PyEFD (https://github.com/hbldh/pyefd).
"""

import torch
from torch import Tensor


def _get_contour_edges(contour: Tensor) -> tuple[Tensor, Tensor]:
    dxy = contour.diff(dim=-2)
    lengths = torch.linalg.norm(dxy, dim=-1)
    return dxy, lengths


def _accumulate_contour_edge_lengths(lengths: Tensor) -> Tensor:
    # accumulate lengths to positions on contour
    cum_lens = lengths.cumsum(-1)

    # prepend zero to cumumlative lengths
    zero = torch.zeros(
        (*cum_lens.shape[:-1], 1),
        dtype=cum_lens.dtype,
        device=cum_lens.device,
    )
    cum_lens = torch.cat([zero, cum_lens], dim=-1)

    # normalize to [0, 1]
    cum_lens /= cum_lens[..., [-1]]

    return cum_lens


def _close_contour(contour: Tensor, tolerance: float = 1e-6) -> Tensor:
    """
    Close an open contour by appending the first point to the end.
    """
    endpt_dist = torch.linalg.norm(contour[..., 0, :] - contour[..., -1, :], dim=-1)
    if torch.any(endpt_dist > tolerance):
        contour = torch.cat([contour, contour[..., [0], :]], dim=-2)
    return contour


def compute_efds(
    contour: Tensor,
    order: int,
    normalize: bool = False,
) -> Tensor:
    """
    Compute elliptical fourier descriptors for a contour.

    :param contour: 2D contour of shape (N, 2) or batch of contours of shape (B, N, 2).
    :param order: Number of fourier coefficients to compute.
    :param normalize: If True, return phase, rotation and scale invariant efds.
    :return: Elliptical fourier descriptors as shape ([B], N, order, 4)
    """
    contour = _close_contour(contour)
    edges, edge_lengths = _get_contour_edges(contour)
    ts = _accumulate_contour_edge_lengths(edge_lengths)

    orders = torch.arange(
        1,
        order + 1,
        device=contour.device,
        dtype=contour.dtype,
    ).mul_(2 * torch.pi)

    contour_length = edge_lengths.sum(-1, keepdim=True)
    consts = 2 * contour_length / orders.pow(2)

    phi = ts.unsqueeze(-2).mul(orders.unsqueeze(-1))
    dphi = torch.stack(
        (
            phi.cos().diff(dim=-1),
            phi.sin().diff(dim=-1),
        ),
        dim=-1,
    )

    # o: orders, n: contour points, i: x and y dim, j: cos/sin coeffs
    efds = torch.einsum(
        "...o,...ni,...onj->...oij",
        consts,
        edges / edge_lengths.unsqueeze(-1),
        dphi,
    )
    efds = efds.flatten(-2)

    if normalize:
        efds = normalize_efds(efds)

    return efds


def normalize_phase(efds: Tensor) -> Tensor:
    """
    Normalize the EFDs to have zero phase shift from the first major axis.
    """
    efds = efds.reshape((*efds.shape[:-1], 2, 2))

    # compute the shift angle theta
    a = 2 * efds[..., 0, :, :].prod(dim=-1).sum(dim=-1)

    ones = torch.tensor([1, -1], device=efds.device)
    b = efds[..., 0, :, :].pow(2) * ones
    b = b.sum(dim=[-1, -2])

    theta = 0.5 * torch.arctan2(a, b).unsqueeze(-1)

    orders = 1 + torch.arange(efds.shape[-3], device=efds.device).unsqueeze(0)
    sintheta = torch.sin(orders * theta)
    costheta = torch.cos(orders * theta)

    rot_mat = torch.stack([costheta, -sintheta, sintheta, costheta], dim=-1)
    rot_mat = rot_mat.reshape_as(efds)

    efds = efds @ rot_mat
    efds = efds.flatten(-2)

    # normalize sign of EFDs
    sign = efds[..., 0, 0].sign()
    efds *= sign[..., None, None]

    return efds


def normalize_rotation(efds: Tensor) -> Tensor:
    """
    Normalize the coefficients to be rotation invariant
    by aligning the semi-major axis with the x-axis.
    """
    efds = efds.reshape((*efds.shape[:-1], 2, 2))
    psi = torch.arctan2(efds[..., 0, 1, 0], efds[..., 0, 0, 0])

    cospsi = torch.cos(psi)
    sinpsi = torch.sin(psi)
    rot_mat = torch.stack([cospsi, sinpsi, -sinpsi, cospsi], dim=-1)
    rot_mat = rot_mat.reshape((*efds.shape[:-3], 1, 2, 2))

    efds = rot_mat @ efds
    return efds.flatten(-2)


def normalize_scale(efds: Tensor) -> Tensor:
    """Normalize the scale of the EFDs."""
    return efds / torch.abs(efds[..., :1, :1])


def normalize_efds(efds: Tensor) -> Tensor:
    """Normalize phase, rotation and scale of EFDs."""
    efds = normalize_phase(efds)
    efds = normalize_rotation(efds)
    efds = normalize_scale(efds)
    return efds
