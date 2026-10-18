"""Triangular pore: a channel whose cross-section is an equilateral triangle with side a.

Conventions as in PANDA_Polygon (`docs/model/triangle.py`, commit 38ff491): a = 1, the profile runs along the
height, perpendicular to the base; z = 0 is the centroid (intersection of the bisectors), the base is at
z = −H/3, the apex at z = 2H/3,
H = √3 a / 2. The δ layer shifts all three walls inwards: accessible side a′ = 1 − 2√3 δ.
"""

import math
from typing import ClassVar

from panda_md.pores.polygon import PolygonGeometry, PolygonPore

SQRT3 = math.sqrt(3.0)
TRIANGLE_SECTION_AREA = SQRT3 / 4.0  # area of the original cross-section at a = 1
TRIANGLE_HEIGHT = SQRT3 / 2.0  # height of the original triangle at a = 1


def accessible_side(wall_layer: float) -> float:
    """a′ = 1 − 2√3 δ — side of the triangle accessible to the liquid."""
    return 1.0 - 2.0 * SQRT3 * wall_layer


def accessible_height(wall_layer: float) -> float:
    """H′ = √3 a′ / 2."""
    return 0.5 * SQRT3 * accessible_side(wall_layer)


def accessible_bottom(wall_layer: float) -> float:
    """Base of the accessible triangle: −H′ / 3."""
    return -accessible_height(wall_layer) / 3.0


def accessible_top(wall_layer: float) -> float:
    """Apex of the accessible triangle: 2 H′ / 3."""
    return 2.0 * accessible_height(wall_layer) / 3.0


class TriangleGeometry(PolygonGeometry):
    section_area: ClassVar[float] = TRIANGLE_SECTION_AREA

    def coordinate_bounds(self) -> tuple[float, float]:
        return (-TRIANGLE_HEIGHT / 3.0, 2.0 * TRIANGLE_HEIGHT / 3.0)


class TrianglePore(PolygonPore):
    name: ClassVar[str] = "triangle"
    geometry_class: ClassVar[type[PolygonGeometry]] = TriangleGeometry

    def geometry(self) -> TriangleGeometry:
        return TriangleGeometry(
            pore_length_ratio=self.length_nm / self.side_nm, layer_width_ratio=self.layer_width_nm / self.side_nm
        )
