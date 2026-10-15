"""Отступления от PANDA_Polygon там, где формулы источника дают нефизичный результат (05_polygon_pores.md §4)."""

import math

from panda_md import TrianglePore

TRIANGLE = TrianglePore(side_nm=1.0, length_nm=2.0, layer_width_nm=1.0)
NEGATIVE_AREA_ANGLE_DEG = 2.0 * math.degrees(math.atan(1.5))  # 1 + cos θ = ⅔ sin θ: S_LV droplet_w2c1 меняет знак


def test_triangle_corner_droplet_does_not_exist_where_its_free_surface_is_negative() -> None:
    droplet = TRIANGLE.interface("droplet_w2c1")
    assert droplet.is_feasible(theta=100.0, delta=0.0, phi=0.02)
    assert droplet.surface_terms(theta=100.0, delta=0.0, phi=0.02).liquid_liquid_area > 0.0
    for theta in (NEGATIVE_AREA_ANGLE_DEG + 0.1, 115.0, 119.0):
        assert not droplet.is_feasible(theta=theta, delta=0.0, phi=0.02), theta
