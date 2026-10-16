"""Квадратная пора: канал квадратного сечения со стороной a и длиной L вдоль оси.

Соглашения — как в PANDA_Polygon (`docs/model/square.py`, коммит 38ff491): a = 1, z = 0 в центре сечения,
профиль поперёк одной стенки, нормировка на слой шириной ly (в долях a), φ — относительно исходного сечения.
"""

from typing import ClassVar

from panda_md.pores.polygon import PolygonGeometry, PolygonPore

SQUARE_HALF_SIDE = 0.5  # стенки квадрата в безразмерной координате: z = ±0.5
SQUARE_SECTION_AREA = 1.0  # площадь исходного сечения при a = 1


def accessible_side(wall_layer: float) -> float:
    """a′ = 1 − 2δ — сторона квадрата, доступная жидкости."""
    return 1.0 - 2.0 * wall_layer


class SquareGeometry(PolygonGeometry):
    section_area: ClassVar[float] = SQUARE_SECTION_AREA

    def coordinate_bounds(self) -> tuple[float, float]:
        return (-SQUARE_HALF_SIDE, SQUARE_HALF_SIDE)


class SquarePore(PolygonPore):
    name: ClassVar[str] = "square"
    geometry_class: ClassVar[type[PolygonGeometry]] = SquareGeometry

    def geometry(self) -> SquareGeometry:
        return SquareGeometry(
            pore_length_ratio=self.length_nm / self.side_nm, layer_width_ratio=self.layer_width_nm / self.side_nm
        )
