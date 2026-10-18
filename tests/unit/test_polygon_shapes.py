"""Deviations from PANDA_Polygon where the source formulas are unphysical (05_polygon_pores.md §4)."""

import pytest

from panda_md import SquarePore, TrianglePore
from panda_md.core.errors import UnknownShapeError

TRIANGLE = TrianglePore(side_nm=1.0, length_nm=2.0, layer_width_nm=1.0)
SQUARE = SquarePore(side_nm=1.0, length_nm=4.0, layer_width_nm=1.0)


@pytest.mark.parametrize("shape_name", ["droplet_w2c1", "capsule_w3c3"])
def test_triangle_shapes_with_wrong_free_surface_are_not_ported(shape_name: str) -> None:
    """The free surface area of the source disagrees with the geometry (Monte Carlo check, §4 items 7–8)."""
    with pytest.raises(UnknownShapeError):
        TRIANGLE.interface(shape_name)


@pytest.mark.parametrize("theta", [135.0, 134.999])
def test_square_corner_droplet_boundary_is_strict(theta: float) -> None:
    """The source condition allows θ = 3π/4, but the area function fails there; here the bound is strict."""
    droplet = SQUARE.interface("droplet_w2c1")
    assert droplet.is_feasible(theta=theta, delta=0.0, phi=0.02) == (theta < 135.0)
