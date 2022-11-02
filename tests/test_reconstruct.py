import torch
from torch.testing import assert_close

from shapely.geometry import Polygon

from torch_efd import (
    compute_elliptic_fourier_descriptors,
    reconstruct_contours,
    reconstruct_normals,
    compute_curvature,
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


def test_reconstruct_contours(ellipse):
    efds = compute_elliptic_fourier_descriptors(ellipse, order=20)
    recons = reconstruct_contours(efds, ts=50)
    assert jaccard_similarity(ellipse, recons) > 0.99


def test_reconstruct_normals(ellipse):
    efds = compute_elliptic_fourier_descriptors(ellipse, order=20)
    normals = reconstruct_normals(efds, ts=50, normed=True)
    assert_close(normals.norm(dim=-1), torch.ones(len(normals)))


def test_compute_curvature(ellipse):
    efds = compute_elliptic_fourier_descriptors(ellipse, order=20)
    curvature = compute_curvature(efds, ts=50)
    assert (curvature > 0).all()
