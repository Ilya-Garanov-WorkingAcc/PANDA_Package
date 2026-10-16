"""Общая база форм щели: система отсчёта поры, доступная ширина 1 − 2δ, существование по интервалу φ,
трёхмерная форма по профилю.

Профиль щели — доля площади сечения ячейки, занятая жидкостью на высоте z. Поэтому сечение восстанавливается
по профилю: круг (капля, пончик) площади ρ · S, круглое отверстие (перфорация) площади (1 − ρ) · S, полоса
вдоль короткой стороны (валик, червь) ширины ρ · l, всё сечение (слой); S = k l² — основание ячейки.
"""

import math
from enum import Enum
from typing import ClassVar

import numpy as np

from panda_md.core.enums import Frame, ParameterName
from panda_md.core.parameters import ParameterValues
from panda_md.core.shape import InterfaceShape
from panda_md.core.types import BoolArray, FloatArray
from panda_md.pores.slit.pore import SlitGeometry

FULL_DENSITY_THRESHOLD = 0.5  # профиль слоя — ступенька 0/1


class CrossSection(str, Enum):
    """Сечение формы щели плоскостью z = const."""

    DISK = "disk"  # круг с центром на вертикальной оси
    HOLE = "hole"  # всё доступное сечение, кроме круга на вертикальной оси
    BAND = "band"  # полоса вдоль короткой стороны, симметричная относительно x = 0
    FULL = "full"  # всё сечение


class SlitShape(InterfaceShape[SlitGeometry]):
    """Форма щели. Существует, если доступная ширина положительна и φ строго внутри (φ_min, φ_max)."""

    frame: ClassVar[Frame] = Frame.PORE
    cross_section: ClassVar[CrossSection]

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

    def contains(self, points: FloatArray, geometry: SlitGeometry, values: ParameterValues) -> BoolArray:
        """Точка в жидкости, если она внутри сечения на своей высоте (сечение — по профилю, см. модуль)."""
        length = geometry.pore_length_ratio
        width = geometry.short_side_ratio()
        along_length = points[:, 0] - length * np.round(points[:, 0] / length)
        along_width = points[:, 1] - width * np.round(points[:, 1] / width)
        density = self.profile(points[:, 2], geometry, values)
        base = geometry.base_area()
        if self.cross_section is CrossSection.BAND:
            inside: BoolArray = np.abs(along_length) < 0.5 * density * length
            return inside
        radius_squared = along_length**2 + along_width**2
        if self.cross_section is CrossSection.DISK:
            disk: BoolArray = radius_squared < density * base / math.pi
            return disk
        if self.cross_section is CrossSection.HOLE:
            hole_radius_squared = (1.0 - density) * base / math.pi
            liquid: BoolArray = self.inside_accessible(points[:, 2], values) & (radius_squared >= hole_radius_squared)
            return liquid
        full: BoolArray = density >= FULL_DENSITY_THRESHOLD
        return full
