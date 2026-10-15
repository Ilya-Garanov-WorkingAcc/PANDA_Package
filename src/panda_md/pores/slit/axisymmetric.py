"""Осесимметричные формы щели (капля, пончик, перфорация) в ячейке с основанием L × kL.

Формулы таких форм записаны для квадратной ячейки (Semenchuk et al., 2025). Если форма не касается боковых
границ ячейки, она «видит» только короткую сторону kL: ячейка L × kL эквивалентна квадратной ячейке kL × kL,
в которой та же жидкость занимает другую долю объёма. Подкласс задаёт формулы квадратной ячейки и правило
пересчёта φ; профиль, границы φ и центр масс в ячейке L × kL выводятся отсюда. При k = 1 пересчёт тождественный.
Проверка — формулы Shistko et al. (2025) при θ = 180°, δ = 0 (tests/unit/test_slit_rectangular_cell.py).
"""

from abc import abstractmethod
from typing import ClassVar

import numpy as np

from panda_md.core.enums import Frame, ParameterName
from panda_md.core.parameters import ParameterValues
from panda_md.core.shape import InterfaceShape
from panda_md.core.types import BoolArray, FloatArray
from panda_md.pores.slit.pore import SlitGeometry


class AxisymmetricSlitShape(InterfaceShape[SlitGeometry]):
    """База осесимметричной формы щели: формулы квадратной ячейки + пересчёт на ячейку L × kL."""

    frame: ClassVar[Frame] = Frame.PORE

    @abstractmethod
    def square_profile(self, coordinate: FloatArray, square: SlitGeometry, values: ParameterValues) -> FloatArray:
        """ρ/ρ_bulk в квадратной ячейке со стороной l (k = 1)."""

    @abstractmethod
    def square_phi_limits(self, square: SlitGeometry, values: ParameterValues) -> tuple[float, float]:
        """Интервал φ в квадратной ячейке."""

    def square_center_of_mass(self, square: SlitGeometry, values: ParameterValues) -> float:
        return InterfaceShape.center_of_mass(self, square, values)

    def equivalent_volume_fraction(self, geometry: SlitGeometry, values: ParameterValues) -> float:
        """φ в эквивалентной квадратной ячейке: объём жидкости сохраняется, площадь основания k l² → (k l)²."""
        return values[ParameterName.VOLUME_FRACTION] / geometry.base_aspect_ratio

    def volume_fraction_from_equivalent(
        self, geometry: SlitGeometry, values: ParameterValues, equivalent: float
    ) -> float:
        """Обратный пересчёт: φ в ячейке L × kL по φ эквивалентной квадратной ячейки."""
        return equivalent * geometry.base_aspect_ratio

    def profile_from_equivalent(
        self, geometry: SlitGeometry, values: ParameterValues, coordinate: FloatArray, square_density: FloatArray
    ) -> FloatArray:
        """ρ в ячейке L × kL: то же сечение жидкости, но основание в k раз меньше квадратного (kl)²/(k l²) = k."""
        return geometry.base_aspect_ratio * square_density

    def square_values(self, geometry: SlitGeometry, values: ParameterValues) -> dict[str, float]:
        converted = dict(values)
        converted[ParameterName.VOLUME_FRACTION] = self.equivalent_volume_fraction(geometry, values)
        return converted

    def profile(self, coordinate: FloatArray, geometry: SlitGeometry, values: ParameterValues) -> FloatArray:
        square_density = self.square_profile(
            coordinate, geometry.equivalent_square(), self.square_values(geometry, values)
        )
        return self.profile_from_equivalent(geometry, values, coordinate, square_density)

    def phi_limits(self, geometry: SlitGeometry, values: ParameterValues) -> tuple[float, float]:
        lower, upper = self.square_phi_limits(geometry.equivalent_square(), self.square_values(geometry, values))
        return (
            self.volume_fraction_from_equivalent(geometry, values, lower),
            self.volume_fraction_from_equivalent(geometry, values, upper),
        )

    def center_of_mass(self, geometry: SlitGeometry, values: ParameterValues) -> float:
        return self.square_center_of_mass(geometry.equivalent_square(), self.square_values(geometry, values))

    def feasible(self, geometry: SlitGeometry, values: ParameterValues) -> bool:
        if self.accessible_width(values) <= 0.0:
            return False
        lower, upper = self.phi_limits(geometry, values)
        return bool(lower < values[ParameterName.VOLUME_FRACTION] < upper)

    @staticmethod
    def accessible_width(values: ParameterValues) -> float:
        """1 − 2δ — ширина щели, доступная анализируемой жидкости."""
        return 1.0 - 2.0 * values[ParameterName.WALL_LAYER_THICKNESS]

    @staticmethod
    def inside_accessible(coordinate: FloatArray, values: ParameterValues) -> BoolArray:
        """|z| < 1/2 − δ — точки, доступные анализируемой жидкости."""
        half = 0.5 * (1.0 - 2.0 * values[ParameterName.WALL_LAYER_THICKNESS])
        return np.abs(coordinate) < half
