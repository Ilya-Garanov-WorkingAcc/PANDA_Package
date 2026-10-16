"""Перфорация (perforation) в щели: слой жидкости между стенками с осесимметричным отверстием.

Формулы: Semenchuk et al., Colloids Surf. A 708 (2025) 135994, уравнения (19)–(22), с поправкой на слой δ
по старой PANDA (`interface/alpha.py`). Ячейка L × kL — через эквивалентную квадратную ячейку: сохраняется
объём отверстия, а не жидкости (axisymmetric.py).
"""

import math
from typing import ClassVar

import numpy as np

from panda_md.core.centering import SymmetryAxis
from panda_md.core.enums import AxisLocator, AxisOrientation, ParameterName, Regime
from panda_md.core.parameters import CONTACT_ANGLE, VOLUME_FRACTION, WALL_LAYER_THICKNESS, Parameter, ParameterValues
from panda_md.core.types import FloatArray
from panda_md.pores.slit.axisymmetric import AxisymmetricSlitShape
from panda_md.pores.slit.pore import SlitGeometry, SlitPore
from panda_md.pores.slit.shape import CrossSection


@SlitPore.register
class SlitPerforation(AxisymmetricSlitShape):
    name: ClassVar[str] = "perforation"
    cross_section: ClassVar[CrossSection] = CrossSection.HOLE
    symmetry_axis: ClassVar[SymmetryAxis | None] = SymmetryAxis(
        orientation=AxisOrientation.VERTICAL, locate=AxisLocator.VOID_CENTER
    )
    parameters: ClassVar[tuple[Parameter, ...]] = (CONTACT_ANGLE, WALL_LAYER_THICKNESS, VOLUME_FRACTION)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: (90.0, 180.0)}

    def equivalent_volume_fraction(self, geometry: SlitGeometry, values: ParameterValues) -> float:
        """Объём отверстия ((1−2δ) − φ) k l² сохраняется в квадратной ячейке со стороной k l."""
        width = self.accessible_width(values)
        return width - (width - values[ParameterName.VOLUME_FRACTION]) / geometry.base_aspect_ratio

    def volume_fraction_from_equivalent(
        self, geometry: SlitGeometry, values: ParameterValues, equivalent: float
    ) -> float:
        width = self.accessible_width(values)
        return width - geometry.base_aspect_ratio * (width - equivalent)

    def profile_from_equivalent(
        self, geometry: SlitGeometry, values: ParameterValues, coordinate: FloatArray, square_density: FloatArray
    ) -> FloatArray:
        """Доля сечения, занятая отверстием, в k раз меньше квадратной: ρ = 1 − k (1 − ρ_кв) внутри щели."""
        density = 1.0 - geometry.base_aspect_ratio * (1.0 - square_density)
        return np.where(self.inside_accessible(coordinate, values), density, 0.0)

    def _discriminant(self, square: SlitGeometry, values: ParameterValues) -> float:
        """Подкоренное выражение уравнения (19); < 0 — отверстия с такой φ нет."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        width = self.accessible_width(values)
        cos_angle = math.cos(contact_angle)
        arc_term = math.pi - 2.0 * contact_angle
        return (
            width**2 * arc_term**2 / cos_angle**4
            - (12.0 - 4.0 * arc_term * math.tan(contact_angle)) * width**2 / cos_angle**2
            + 64.0 * (1.0 - values[ParameterName.VOLUME_FRACTION] / width) * square.pore_length_ratio**2 / math.pi
            + 4.0 * width**2 / 3.0
        )

    def hole_diameter(self, square: SlitGeometry, values: ParameterValues) -> float:
        """d_perf — диаметр отверстия у стенки, уравнение (19) с учётом δ (alpha.py)."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        width = self.accessible_width(values)
        cos_angle = math.cos(contact_angle)
        tan_angle = math.tan(contact_angle)
        arc_term = math.pi - 2.0 * contact_angle
        discriminant = self._discriminant(square, values)
        return (
            0.25 * math.sqrt(max(discriminant, 0.0)) + 0.5 * width * tan_angle - 0.25 * width * arc_term / cos_angle**2
        )

    def square_phi_limits(self, square: SlitGeometry, values: ParameterValues) -> tuple[float, float]:
        """φ_min — отверстие касается своего образа; φ_max — отверстие закрывается.
        Диаметр отверстия у стенки d(φ) = ¼√D(φ) + c, D линейно убывает с φ, поэтому d ≥ c. Если c > l, отверстие
        не помещается в ячейку ни при какой φ — перфорации нет (интервал пуст). Иначе корень формулы φ_min настоящий.
        Формула (21) получена возведением в квадрат и в узкой ячейке (c > l) даёт посторонний корень с отверстием
        шире ячейки и ρ < 0 (итоговый аудит, O12).
        """
        lower, upper = self._formula_phi_limits(square, values)
        if upper <= 0.0 or self._smallest_hole_diameter(values) > square.pore_length_ratio:
            return max(upper, 0.0), max(upper, 0.0)  # пустой интервал: перфорации в такой ячейке нет
        return max(lower, 0.0), upper

    def _smallest_hole_diameter(self, values: ParameterValues) -> float:
        """c — диаметр отверстия у стенки при D = 0: меньше он не бывает ни при какой φ (уравнение (19))."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        width = self.accessible_width(values)
        arc_term = math.pi - 2.0 * contact_angle
        return 0.5 * width * math.tan(contact_angle) - 0.25 * width * arc_term / math.cos(contact_angle) ** 2

    def _formula_phi_limits(self, square: SlitGeometry, values: ParameterValues) -> tuple[float, float]:
        """Уравнение (21) с δ (alpha.py): φ_min — отверстие касается образа, φ_max — закрывается."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        width = self.accessible_width(values)
        length_ratio = square.pore_length_ratio
        cos_angle = math.cos(contact_angle)
        tan_angle = math.tan(contact_angle)
        arc_term = math.pi - 2.0 * contact_angle
        lower = width * (1.0 - math.pi / 4.0) + math.pi * width**2 / 8.0 / length_ratio**2 * (
            2.0 * width / 3.0
            + 2.0 * length_ratio * tan_angle
            - (2.0 * width + arc_term * length_ratio - width * arc_term * tan_angle) / cos_angle**2
        )
        upper = width - math.pi * width**3 / (16.0 * length_ratio**2 * cos_angle**3) * (
            7.0 * cos_angle - math.cos(3.0 * contact_angle) / 3.0 - 2.0 * (arc_term + math.sin(2.0 * contact_angle))
        )
        return lower, upper

    def square_center_of_mass(self, square: SlitGeometry, values: ParameterValues) -> float:
        return 0.0  # перфорация симметрична относительно середины щели

    def square_profile(self, coordinate: FloatArray, square: SlitGeometry, values: ParameterValues) -> FloatArray:
        """ρ/ρ_bulk = 1 − π y² / l², y = (d − √((1−2δ)²/cos²θ − 4z²) − (1−2δ) tan θ) / 2 — уравнение (22) с δ."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        width = self.accessible_width(values)
        meniscus = np.sqrt(np.maximum(width**2 / math.cos(contact_angle) ** 2 - 4.0 * coordinate**2, 0.0))
        hole_radius = 0.5 * (self.hole_diameter(square, values) - meniscus - width * math.tan(contact_angle))
        density = 1.0 - math.pi * hole_radius**2 / square.pore_length_ratio**2
        return np.where(self.inside_accessible(coordinate, values), density, 0.0)
