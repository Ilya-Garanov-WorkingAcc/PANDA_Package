"""Общая база форм щели: система отсчёта поры, доступная ширина 1 − 2δ, существование по интервалу φ."""

from typing import ClassVar

import numpy as np

from panda_md.core.enums import Frame, ParameterName
from panda_md.core.parameters import ParameterValues
from panda_md.core.shape import InterfaceShape
from panda_md.core.types import BoolArray, FloatArray
from panda_md.pores.slit.pore import SlitGeometry


class SlitShape(InterfaceShape[SlitGeometry]):
    """Форма щели. Существует, если доступная ширина положительна и φ строго внутри (φ_min, φ_max)."""

    frame: ClassVar[Frame] = Frame.PORE

    @staticmethod
    def accessible_width(values: ParameterValues) -> float:
        """1 − 2δ — ширина щели, доступная анализируемой жидкости."""
        return 1.0 - 2.0 * values[ParameterName.WALL_LAYER_THICKNESS]

    @staticmethod
    def inside_accessible(coordinate: FloatArray, values: ParameterValues) -> BoolArray:
        """|z| < 1/2 − δ — точки, доступные анализируемой жидкости."""
        half = 0.5 * SlitShape.accessible_width(values)
        return np.abs(coordinate) < half

    def feasible(self, geometry: SlitGeometry, values: ParameterValues) -> bool:
        if self.accessible_width(values) <= 0.0:
            return False
        lower, upper = self.phi_limits(geometry, values)
        return bool(lower < values[ParameterName.VOLUME_FRACTION] < upper)
