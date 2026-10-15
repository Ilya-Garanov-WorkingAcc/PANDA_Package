"""Капля (droplet) в щели: сферический сегмент на нижней стенке.

Формулы: Semenchuk et al., Colloids Surf. A 708 (2025) 135994, уравнения (1), (3)–(5); границы существования —
по старой PANDA (`interface/alpha.py`, ветка θ ≥ 90°). Профиль записан в системе поры: капля стоит на слое δ
над нижней стенкой z = −1/2.
"""

import math
from typing import ClassVar

import numpy as np

from panda_md.core.enums import Frame, ParameterName, Regime
from panda_md.core.parameters import CONTACT_ANGLE, VOLUME_FRACTION, WALL_LAYER_THICKNESS, Parameter, ParameterValues
from panda_md.core.shape import InterfaceShape
from panda_md.core.types import FloatArray
from panda_md.pores.common import spherical_cap_centroid_offset, spherical_cap_volume_factor
from panda_md.pores.slit.pore import SLIT_HALF_WIDTH, SlitGeometry, SlitPore


@SlitPore.register
class SlitDroplet(InterfaceShape[SlitGeometry]):
    name: ClassVar[str] = "droplet"
    parameters: ClassVar[tuple[Parameter, ...]] = (CONTACT_ANGLE, WALL_LAYER_THICKNESS, VOLUME_FRACTION)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: (90.0, 180.0)}
    frame: ClassVar[Frame] = Frame.PORE

    def sphere_radius(self, geometry: SlitGeometry, values: ParameterValues) -> float:
        """R из π R³ (2 − 3 cos θ + cos³ θ) / 3 = φ l², уравнение (1)."""
        volume = values[ParameterName.VOLUME_FRACTION] * geometry.pore_length_ratio**2
        return float(
            np.cbrt(3.0 * volume / (math.pi * spherical_cap_volume_factor(values[ParameterName.CONTACT_ANGLE])))
        )

    def sphere_center(self, geometry: SlitGeometry, values: ParameterValues) -> float:
        floor = -SLIT_HALF_WIDTH + values[ParameterName.WALL_LAYER_THICKNESS]
        return floor - self.sphere_radius(geometry, values) * math.cos(values[ParameterName.CONTACT_ANGLE])

    def phi_limits(self, geometry: SlitGeometry, values: ParameterValues) -> tuple[float, float]:
        """φ ≤ min(капля не выше доступной щели, пятно не шире ячейки) — уравнение (3) и alpha.py."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        accessible_width = 1.0 - 2.0 * values[ParameterName.WALL_LAYER_THICKNESS]
        length_ratio = geometry.pore_length_ratio
        by_height = (
            math.pi
            * accessible_width**3
            * (2.0 + math.cos(contact_angle))
            / (3.0 * length_ratio**2 * (1.0 - math.cos(contact_angle)))
        )
        by_length = math.pi * length_ratio * (2.0 + math.cos(contact_angle)) * math.sin(0.5 * contact_angle) ** 4 / 6.0
        return 0.0, min(by_height, by_length)

    def feasible(self, geometry: SlitGeometry, values: ParameterValues) -> bool:
        if 1.0 - 2.0 * values[ParameterName.WALL_LAYER_THICKNESS] <= 0.0:
            return False
        lower, upper = self.phi_limits(geometry, values)
        return lower < values[ParameterName.VOLUME_FRACTION] <= upper

    def center_of_mass(self, geometry: SlitGeometry, values: ParameterValues) -> float:
        radius = self.sphere_radius(geometry, values)
        return self.sphere_center(geometry, values) + spherical_cap_centroid_offset(
            radius, values[ParameterName.CONTACT_ANGLE]
        )

    def profile(self, coordinate: FloatArray, geometry: SlitGeometry, values: ParameterValues) -> FloatArray:
        """ρ/ρ_bulk = π (R² − (z − z_c)²) / l² между опорой капли и её верхушкой, уравнение (5)."""
        radius = self.sphere_radius(geometry, values)
        center = self.sphere_center(geometry, values)
        floor = -SLIT_HALF_WIDTH + values[ParameterName.WALL_LAYER_THICKNESS]
        inside = (coordinate > floor) & (coordinate < center + radius)  # строгие границы, как в alpha.py
        section_area = math.pi * np.maximum(radius**2 - (coordinate - center) ** 2, 0.0)
        return np.where(inside, section_area / geometry.pore_length_ratio**2, 0.0)
