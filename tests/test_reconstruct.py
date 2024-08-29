# pyright: basic

import torch
from torch.testing import assert_close

from shapely.geometry import Polygon

from torch_efd import (
    compute_efds,
    reconstruct_efds,
    reconstruct_tangents,
    reconstruct_normals,
)


def make_valid_polygon(coords):
    if torch.isclose(coords[0], coords[-1]).all():
        coords = coords[1:]
    return Polygon(coords)


def jaccard_similarity(a, b):
    a = make_valid_polygon(a)
    b = make_valid_polygon(b)

    intrs = (a & b).area
    union = (a | b).area
    return intrs / union


def estimate_tans(coords):
    """Estimate tangents using central differences"""
    arc_tans = torch.diff(
        coords,
        prepend=coords[[-2]],
        dim=0,
    )
    arc_tans = arc_tans * (len(coords) - 1)
    return (arc_tans[:-1] + arc_tans[1:]) / 2


def test_reconstruct_contours(ellipse):
    efds = compute_efds(ellipse, order=20)
    recons = reconstruct_efds(efds, ts=50)
    assert jaccard_similarity(ellipse, recons) > 0.99


def test_reconstruct_tangents(ellipse):
    efds = compute_efds(ellipse, order=20)
    recons = reconstruct_efds(efds, ts=100)
    expected = estimate_tans(recons)
    tangents = reconstruct_tangents(efds, ts=100)

    error = torch.linalg.norm(tangents[:-1] - expected, dim=-1)
    assert torch.all(error < 1e-1)


def test_reconstruct_normals(ellipse):
    efds = compute_efds(ellipse, order=20)
    normals = reconstruct_normals(efds, ts=50, normed=True)
    assert_close(normals.norm(dim=-1), torch.ones(len(normals)))
