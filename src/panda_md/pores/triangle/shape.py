"""Общее для форм треугольной поры: система отсчёта поры и функции угла, повторяющиеся в формулах."""

import math
from typing import ClassVar

from panda_md.core.enums import Frame, ParameterName
from panda_md.core.parameters import ParameterValues
from panda_md.core.shape import InterfaceShape
from panda_md.pores.triangle.pore import TriangleGeometry, accessible_side

THIRD_PI = math.pi / 3.0
NON_WETTING_ABOVE_TWO_THIRDS_PI = (120.0, 180.0)  # формулы выведены при θ > 2π/3
NON_WETTING_FULL = (90.0, 180.0)
NON_WETTING_BELOW_TWO_THIRDS_PI = (90.0, 120.0)  # формулы выведены при π/2 < θ < 2π/3


class TriangleShape(InterfaceShape[TriangleGeometry]):
    """Форма треугольной поры: формулы записаны в системе поры (0 — центроид сечения)."""

    frame: ClassVar[Frame] = Frame.PORE

    @staticmethod
    def side(values: ParameterValues) -> float:
        """a′ — доступная сторона треугольника."""
        return accessible_side(values[ParameterName.WALL_LAYER_THICKNESS])

    @staticmethod
    def supplementary_angle(values: ParameterValues) -> float:
        """θ′ = π − θ — угол через смачивающую жидкость (так записаны формулы PANDA_Polygon)."""
        return math.pi - values[ParameterName.CONTACT_ANGLE]


def corner_factor(supplementary_angle: float) -> float:
    """θ′ − π/3 − sin(2θ′ − π/3) + √3/2 — площадь скруглённого пустого угла, делённая на R²."""
    return supplementary_angle - THIRD_PI - math.sin(2.0 * supplementary_angle - THIRD_PI) + math.sqrt(3.0) / 2.0
