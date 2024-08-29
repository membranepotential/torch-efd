# pyright: basic


from torch_efd import (
    compute_efds,
    compute_curvature,
    compute_speed,
    integrate_total_length,
    compute_area,
)


def test_compute_curvature(ellipse):
    efds = compute_efds(ellipse, order=20)
    curvature = compute_curvature(efds, ts=50)
    assert (curvature < 0).all()


def test_compute_speed(ellipse):
    efds = compute_efds(ellipse, order=20)
    speed = compute_speed(efds, ts=50)
    assert (speed > 0).all()


def test_integrate_total_length(ellipse):
    efds = compute_efds(ellipse, order=20)
    total_length = integrate_total_length(efds, ts=50)
    assert total_length > 0


def test_compute_area(ellipse):
    efds = compute_efds(ellipse, order=20)
    area = compute_area(efds)
    assert area > 0
