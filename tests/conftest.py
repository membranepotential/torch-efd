# pylint: disable=redefined-outer-name

import torch
from torch import Tensor

from shapely.geometry import Point, LinearRing
from shapely.affinity import scale, rotate

import pytest


def roll_coords(coords: Tensor, shift: int):
    coords = coords[:-1]
    coords = coords.roll(shift, dims=-2)
    coords = torch.cat([coords, coords[..., [0], :]], dim=-2)
    return coords


@pytest.fixture
def ellipse() -> Tensor:
    shape = Point(0, 0).buffer(1).boundary
    shape = scale(shape, 2.0, 1.0)
    shape = rotate(shape, 30)
    shape = shape.simplify(0.01)
    return torch.tensor(shape.coords)


@pytest.fixture
def rolled_ellipses(ellipse) -> Tensor:
    shifts = torch.linspace(0, 1, 5) * ellipse.shape[-2]
    shifts = shifts.long().tolist()
    return torch.stack([roll_coords(ellipse, shift) for shift in shifts])


@pytest.fixture
def rotated_ellipses(ellipse) -> Tensor:
    ellipse = LinearRing(ellipse)

    rotations = torch.linspace(0, 360, 20)
    ellipses = [rotate(ellipse, rotation) for rotation in rotations]
    return torch.stack([torch.Tensor(e.coords) for e in ellipses])


@pytest.fixture
def scaled_ellipses(ellipse) -> Tensor:
    scales = [0.1, 0.5, 1.0, 2.0, 5.0]
    return torch.stack([ellipse * s for s in scales])


@pytest.fixture
def random_transformed_ellipses(ellipse) -> Tensor:
    ellipse = LinearRing(ellipse)
    n_ellipses = 20

    rotations = torch.rand(n_ellipses) * 360
    scales = torch.exp(torch.randn(n_ellipses))
    shifts = torch.randint(len(ellipse.coords), size=(n_ellipses,))

    ellipses = [rotate(ellipse, rotation) for rotation in rotations]
    ellipses = [torch.tensor(e.coords) for e in ellipses]
    ellipses = [roll_coords(e, int(s)) for e, s in zip(ellipses, shifts)]
    ellipses = [e * s for e, s in zip(ellipses, scales)]

    return torch.stack(ellipses)
