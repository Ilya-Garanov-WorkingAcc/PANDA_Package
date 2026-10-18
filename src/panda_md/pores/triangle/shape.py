"""Common code for the triangular pore shapes: pore frame of reference and angle functions repeated in formulas."""

import math
from typing import ClassVar

from panda_md.core.enums import Frame, ParameterName
from panda_md.core.parameters import ParameterValues
from panda_md.core.shape import InterfaceShape
from panda_md.pores.triangle.pore import TriangleGeometry, accessible_side

THIRD_PI = math.pi / 3.0
NON_WETTING_ABOVE_TWO_THIRDS_PI = (120.0, 180.0)  # formulas derived for θ > 2π/3
NON_WETTING_FULL = (90.0, 180.0)


class TriangleShape(InterfaceShape[TriangleGeometry]):
    """Triangular pore shape: formulas are written in the pore frame (0 — cross-section centroid)."""

    frame: ClassVar[Frame] = Frame.PORE

    @staticmethod
    def side(values: ParameterValues) -> float:
        """a′ — accessible side of the triangle."""
        return accessible_side(values[ParameterName.WALL_LAYER_THICKNESS])

    @staticmethod
    def supplementary_angle(values: ParameterValues) -> float:
        """θ′ = π − θ — angle through the wetting liquid (as the PANDA_Polygon formulas are written)."""
        return math.pi - values[ParameterName.CONTACT_ANGLE]


def corner_factor(supplementary_angle: float) -> float:
    """θ′ − π/3 − sin(2θ′ − π/3) + √3/2 — area of a rounded empty corner divided by R²."""
    return supplementary_angle - THIRD_PI - math.sin(2.0 * supplementary_angle - THIRD_PI) + math.sqrt(3.0) / 2.0
