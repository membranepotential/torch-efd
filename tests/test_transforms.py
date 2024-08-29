# pyright: basic

import torch
from torch.testing import assert_close

from torch_efd import (
    compute_efds,
    scale_efds,
    rotate_efds,
    smooth_efds,
)


def test_scale_efds(ellipse):
    efds = compute_efds(ellipse, order=10)
    scaled = scale_efds(efds, 2)
    assert_close(scaled, 2 * efds)


def test_rotate_efds(ellipse):
    efds = compute_efds(ellipse, order=10)
    rotated = rotate_efds(efds, angle=2 * torch.pi)
    assert_close(rotated, efds)


def test_smooth_efds(ellipse):
    efds = compute_efds(ellipse, order=10)
    smoothed = smooth_efds(efds, sigma=0.1)
    assert efds.shape == smoothed.shape
