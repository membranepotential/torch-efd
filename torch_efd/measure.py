import torch
from torch import Tensor


from .reconstruct import (
    derive_tangent_efds,
    reconstruct_tangents,
    reconstruct_efds,
    coerce_ts_vec,
)


def _close_ts_vec(efds: Tensor, ts: Tensor | int) -> Tensor:
    if isinstance(ts, int):
        return coerce_ts_vec(efds, ts)

    zero, one = torch.tensor(
        [[0.0], [1.0]],
        dtype=ts.dtype,
        device=ts.device,
    )
    ts_closed = [ts]

    if not torch.isclose(ts[0], zero):
        ts_closed = [zero] + ts_closed

    if not torch.isclose(ts[-1], one):
        ts_closed = ts_closed + [one]

    return torch.cat(ts_closed).to(efds)


def compute_curvature(
    efds: Tensor,
    ts: Tensor | int,
    signed: bool = True,
) -> Tensor:
    """
    Compute (signed) curvature of a contour.
    The curvatures are invariant to contour parameterization.

    :param efds: EFDs of shape (..., N, 4)
    :param ts: Parameter vector of shape (..., M) or number of vertices M
    :param signed: If True, return signed curvature, else return magnitude
    :return: Curvature values of shape (..., M)
    """

    efds_tan = derive_tangent_efds(efds)
    tangents = reconstruct_efds(efds_tan, ts)
    tangrads = reconstruct_tangents(efds_tan, ts)

    curvature = (
        tangents[..., 0] * tangrads[..., 1] - tangents[..., 1] * tangrads[..., 0]
    )
    curvature /= torch.linalg.norm(tangents, dim=-1).pow(3)

    if not signed:
        curvature = curvature.abs()

    return curvature


def compute_speed(efds: Tensor, ts: Tensor | int) -> Tensor:
    """
    Compute the norms of the tangents which indicate
    the speed of travel along the contour.

    :param efds: EFDs of shape (..., N, 4)
    :param ts: Parameter vector of shape (..., M) or number of vertices M
    :return: Velocity values of shape (..., M)
    """
    efds_tan = reconstruct_tangents(efds, ts)
    return torch.linalg.norm(efds_tan, dim=-1)


def compute_arclens(efds: Tensor, ts: Tensor | int) -> Tensor:
    """
    Compute the length of each arc on the reconstructed contour.

    :param efds: EFDs of shape (..., N, 4)
    :param ts: Parameter vector of shape (..., M) or number of vertices M
    :return: Arc lengths of shape (..., M - 1)
    """
    ts = coerce_ts_vec(efds, ts)
    dts = torch.diff(ts)

    if torch.any(dts < 0):
        raise ValueError(
            "ts must be monotonically increasing for arclength computation"
        )

    speed = compute_speed(efds, ts)
    arc_speed = (speed[:-1] + speed[1:]) / 2
    return arc_speed.mul(dts)


def integrate_total_length(efds: Tensor, ts: Tensor | int = 100) -> Tensor:
    """Compute total length of the contour."""
    ts = _close_ts_vec(efds, ts)
    return compute_arclens(efds, ts).sum(dim=-1)


def compute_area(efds: Tensor, ts: Tensor | int = 100) -> Tensor:
    """Compute polygon area using the shoelace method."""
    coords = reconstruct_efds(efds, ts)
    x, y = coords[..., 0], coords[..., 1]
    left = x * torch.roll(y, 1, dims=-1)
    right = y * torch.roll(x, 1, dims=-1)
    return left.sub_(right).sum(dim=-1).abs_().div_(2)
