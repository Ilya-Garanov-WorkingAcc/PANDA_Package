"""Отступления от PANDA_Polygon там, где формулы источника дают нефизичный результат (05_polygon_pores.md §4)."""

import math

import pytest

from panda_md import SquarePore, TrianglePore

TRIANGLE = TrianglePore(side_nm=1.0, length_nm=2.0, layer_width_nm=1.0)
SQUARE = SquarePore(side_nm=1.0, length_nm=4.0, layer_width_nm=1.0)
NEGATIVE_AREA_ANGLE_DEG = 2.0 * math.degrees(math.atan(1.5))  # 1 + cos θ = ⅔ sin θ: S_LV droplet_w2c1 меняет знак


def test_triangle_corner_droplet_does_not_exist_where_its_free_surface_is_negative() -> None:
    droplet = TRIANGLE.interface("droplet_w2c1")
    assert droplet.is_feasible(theta=100.0, delta=0.0, phi=0.02)
    assert droplet.surface_terms(theta=100.0, delta=0.0, phi=0.02).liquid_liquid_area > 0.0
    for theta in (NEGATIVE_AREA_ANGLE_DEG + 0.1, 115.0, 119.0):
        assert not droplet.is_feasible(theta=theta, delta=0.0, phi=0.02), theta


@pytest.mark.parametrize("theta", [135.0, 134.999])
def test_square_corner_droplet_boundary_is_strict(theta: float) -> None:
    """Источник допускает droplet_w2c1 при θ = 3π/4, но площадь там бесконечна; здесь граница строгая."""
    droplet = SQUARE.interface("droplet_w2c1")
    assert droplet.is_feasible(theta=theta, delta=0.0, phi=0.02) == (theta < 135.0)
