"""Треугольная пора: канал с сечением — равносторонним треугольником со стороной a.

Соглашения — как в PANDA_Polygon (`docs/model/triangle.py`, коммит 38ff491): a = 1, профиль — по высоте,
перпендикулярно основанию; z = 0 — центроид (пересечение биссектрис), основание при z = −H/3, вершина при z = 2H/3,
H = √3 a / 2. Слой δ сдвигает все три стенки внутрь: доступная сторона a′ = 1 − 2√3 δ.
"""

import math
from typing import ClassVar

from panda_md.pores.polygon import PolygonGeometry, PolygonPore

SQRT3 = math.sqrt(3.0)
TRIANGLE_SECTION_AREA = SQRT3 / 4.0  # площадь исходного сечения при a = 1
TRIANGLE_HEIGHT = SQRT3 / 2.0  # высота исходного треугольника при a = 1


def accessible_side(wall_layer: float) -> float:
    """a′ = 1 − 2√3 δ — сторона треугольника, доступная жидкости."""
    return 1.0 - 2.0 * SQRT3 * wall_layer


def accessible_height(wall_layer: float) -> float:
    """H′ = √3 a′ / 2."""
    return 0.5 * SQRT3 * accessible_side(wall_layer)


def accessible_bottom(wall_layer: float) -> float:
    """Основание доступного треугольника: −H′ / 3."""
    return -accessible_height(wall_layer) / 3.0


def accessible_top(wall_layer: float) -> float:
    """Вершина доступного треугольника: 2 H′ / 3."""
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
