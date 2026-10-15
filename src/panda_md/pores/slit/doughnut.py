"""Пончик (doughnut) в щели: перемычка между стенками, осесимметричная вокруг вертикальной оси.

Формулы: Semenchuk et al., Colloids Surf. A 708 (2025) 135994, уравнения (6)–(9), с поправкой на слой δ
по старой PANDA (`interface/alpha.py`). Ячейка L × kL — через эквивалентную квадратную ячейку (axisymmetric.py).
"""

import math
from typing import ClassVar

import numpy as np

from panda_md.core.enums import ParameterName, Regime
from panda_md.core.parameters import CONTACT_ANGLE, VOLUME_FRACTION, WALL_LAYER_THICKNESS, Parameter, ParameterValues
from panda_md.core.types import FloatArray
from panda_md.pores.slit.axisymmetric import AxisymmetricSlitShape
from panda_md.pores.slit.pore import SlitGeometry, SlitPore


@SlitPore.register
class SlitDoughnut(AxisymmetricSlitShape):
    name: ClassVar[str] = "doughnut"
    parameters: ClassVar[tuple[Parameter, ...]] = (CONTACT_ANGLE, WALL_LAYER_THICKNESS, VOLUME_FRACTION)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: (90.0, 180.0)}

    def contact_spot_diameter(self, square: SlitGeometry, values: ParameterValues) -> float:
        """d_dou — диаметр пятна контакта, уравнение (6) с учётом δ (alpha.py)."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        width = self.accessible_width(values)
        length_ratio = square.pore_length_ratio
        cos_angle = math.cos(contact_angle)
        tan_angle = math.tan(contact_angle)
        arc_term = math.pi - 2.0 * contact_angle
        discriminant = (
            width**2 * arc_term**2 / cos_angle**4
            - (12.0 - 4.0 * arc_term * tan_angle) * width**2 / cos_angle**2
            + 64.0 * values[ParameterName.VOLUME_FRACTION] * length_ratio**2 / math.pi / width
            + 4.0 * width**2 / 3.0
        )
        return (
            0.25 * math.sqrt(max(discriminant, 0.0)) - 0.5 * width * tan_angle + 0.25 * width * arc_term / cos_angle**2
        )

    def square_phi_limits(self, square: SlitGeometry, values: ParameterValues) -> tuple[float, float]:
        """φ_min — пятно контакта исчезает; φ_max — пончик касается своего образа (уравнение (8) с δ, alpha.py)."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        width = self.accessible_width(values)
        length_ratio = square.pore_length_ratio
        cos_angle = math.cos(contact_angle)
        tan_angle = math.tan(contact_angle)
        arc_term = math.pi - 2.0 * contact_angle
        lower = (
            math.pi
            * width**3
            / length_ratio**2
            * (tan_angle**2 / 4.0 + 1.0 / 6.0 - arc_term * tan_angle / cos_angle**2 / 8.0)
        )
        upper = (
            math.pi
            * width
            / 16.0
            / length_ratio**2
            * (
                width**2 * tan_angle**2
                - 2.0 * width**2 * arc_term / cos_angle**3
                - (width**2 / 3.0 - 4.0 * length_ratio**2)
                - 4.0 * width * length_ratio * tan_angle
                + width * (7.0 * width - 2.0 * arc_term * length_ratio) / cos_angle**2
                - 4.0 * width * (width * tan_angle - 2.0 * length_ratio) / cos_angle
            )
        )
        return lower, upper

    def square_center_of_mass(self, square: SlitGeometry, values: ParameterValues) -> float:
        return 0.0  # пончик симметричен относительно середины щели

    def square_profile(self, coordinate: FloatArray, square: SlitGeometry, values: ParameterValues) -> FloatArray:
        """ρ/ρ_bulk = π y² / l², y = (d + √((1−2δ)²/cos²θ − 4z²) + (1−2δ) tan θ) / 2 — уравнение (9) с δ."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        width = self.accessible_width(values)
        meniscus = np.sqrt(np.maximum(width**2 / math.cos(contact_angle) ** 2 - 4.0 * coordinate**2, 0.0))
        radius = 0.5 * (self.contact_spot_diameter(square, values) + meniscus + width * math.tan(contact_angle))
        density = math.pi * radius**2 / square.pore_length_ratio**2
        return np.where(self.inside_accessible(coordinate, values), density, 0.0)
