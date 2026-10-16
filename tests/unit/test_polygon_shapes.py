"""Отступления от PANDA_Polygon там, где формулы источника нефизичны (05_polygon_pores.md §4)."""

import pytest

from panda_md import SquarePore, TrianglePore
from panda_md.core.errors import UnknownShapeError

TRIANGLE = TrianglePore(side_nm=1.0, length_nm=2.0, layer_width_nm=1.0)
SQUARE = SquarePore(side_nm=1.0, length_nm=4.0, layer_width_nm=1.0)


@pytest.mark.parametrize("shape_name", ["droplet_w2c1", "capsule_w3c3"])
def test_triangle_shapes_with_wrong_free_surface_are_not_ported(shape_name: str) -> None:
    """Площадь свободной поверхности источника расходится с геометрией (проверка Монте-Карло, §4 п. 7–8)."""
    with pytest.raises(UnknownShapeError):
        TRIANGLE.interface(shape_name)


@pytest.mark.parametrize("theta", [135.0, 134.999])
def test_square_corner_droplet_boundary_is_strict(theta: float) -> None:
    """Условие источника допускает θ = 3π/4, а функция площади там форму уже не считает; здесь граница строгая."""
    droplet = SQUARE.interface("droplet_w2c1")
    assert droplet.is_feasible(theta=theta, delta=0.0, phi=0.02) == (theta < 135.0)
