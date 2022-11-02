import torch
from torch.testing import assert_close

from torch_efd.compute import _close_contour
from torch_efd import (
    compute_elliptic_fourier_descriptors,
    normalize_phase,
    normalize_rotation,
    normalize_scale,
)


def assert_close_batch(batch):
    expected = batch[[0]].expand(len(batch), *batch.shape[1:])
    assert_close(batch, expected)


def test_close_contour(ellipse):
    open_ellipse = ellipse[:-1]
    assert torch.norm(open_ellipse[0] - open_ellipse[-1], dim=-1) > 1e-6

    closed = _close_contour(open_ellipse)
    assert len(closed) == len(ellipse)
    assert torch.norm(closed[0] - closed[-1], dim=-1) < 1e-6

    assert torch.norm(ellipse[0] - ellipse[-1], dim=-1) < 1e-6
    not_closed = _close_contour(ellipse)
    assert len(not_closed) == len(ellipse)


def test_compute_efds(ellipse):
    efds = compute_elliptic_fourier_descriptors(ellipse, order=10)
    assert efds.shape == (10, 4)

    efds = compute_elliptic_fourier_descriptors(ellipse.unsqueeze(0), order=10)
    assert efds.shape == (1, 10, 4)


def test_normalize_phase(rolled_ellipses):
    efds = compute_elliptic_fourier_descriptors(rolled_ellipses, order=10)
    efds = normalize_phase(efds)
    assert_close_batch(efds)


def test_normalize_rotation(rotated_ellipses):
    efds = compute_elliptic_fourier_descriptors(rotated_ellipses, order=10)
    efds = normalize_rotation(efds)
    assert_close_batch(efds)


def test_normalize_scale(scaled_ellipses):
    efds = compute_elliptic_fourier_descriptors(scaled_ellipses, order=10)
    efds = normalize_scale(efds)
    assert_close_batch(efds)


def test_normalize(random_transformed_ellipses):
    efds = compute_elliptic_fourier_descriptors(
        random_transformed_ellipses,
        order=10,
        normalize=True,
    )
    assert_close_batch(efds)
