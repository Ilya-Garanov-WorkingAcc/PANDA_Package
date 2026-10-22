"""Common code for the square pore shapes: pore frame of reference and the accessible side."""

import math
from typing import ClassVar

from panda_md.core.enums import Frame, ParameterName
from panda_md.core.parameters import ParameterValues
from panda_md.core.shape import InterfaceShape
from panda_md.pores.square.pore import SquareGeometry, accessible_side

THREE_QUARTERS_PI = 0.75 * math.pi
NON_WETTING_FULL = (90.0, 180.0)
NON_WETTING_BELOW_THREE_QUARTERS_PI = (90.0, 135.0)  # formulas derived for π/2 < θ < 3π/4
NON_WETTING_ABOVE_THREE_QUARTERS_PI = (135.0, 180.0)  # formulas derived for θ ≥ 3π/4


class SquareShape(InterfaceShape[SquareGeometry]):
    """Square pore shape: formulas are written in the pore frame (0 — cross-section centre)."""

    frame: ClassVar[Frame] = Frame.PORE

    @staticmethod
    def side(values: ParameterValues) -> float:
        """a′ — accessible side of the square."""
        return accessible_side(values[ParameterName.WALL_LAYER_THICKNESS])

    @classmethod
    def floor(cls, values: ParameterValues) -> float:
        """Lower accessible wall: −a′ / 2."""
        return -0.5 * cls.side(values)
