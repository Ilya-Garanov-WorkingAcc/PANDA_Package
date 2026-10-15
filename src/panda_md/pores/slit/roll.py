"""Валик (roll) в щели: перемычка между стенками, однородная вдоль одной оси.

Формулы: Semenchuk et al., Colloids Surf. A 725 (2025) 137586, уравнения (1)–(4);
при δ = 0 совпадают с Semenchuk et al., Colloids Surf. A 708 (2025) 135994, уравнения (15)–(18).
"""

import math
from typing import ClassVar

import numpy as np

from panda_md.core.enums import Frame, ParameterName, Regime
from panda_md.core.parameters import CONTACT_ANGLE, VOLUME_FRACTION, WALL_LAYER_THICKNESS, Parameter, ParameterValues
from panda_md.core.shape import InterfaceShape, SurfaceTerms
from panda_md.core.types import FloatArray
from panda_md.pores.slit.pore import SLIT_HALF_WIDTH, SlitGeometry, SlitPore


@SlitPore.register
class SlitRoll(InterfaceShape[SlitGeometry]):
    name: ClassVar[str] = "roll"
    parameters: ClassVar[tuple[Parameter, ...]] = (CONTACT_ANGLE, WALL_LAYER_THICKNESS, VOLUME_FRACTION)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: (90.0, 180.0)}
    frame: ClassVar[Frame] = Frame.PORE

    def contact_spot_width(self, geometry: SlitGeometry, values: ParameterValues) -> float:
        """d_roll — ширина пятна контакта, уравнение (1)."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        accessible_width = 1.0 - 2.0 * values[ParameterName.WALL_LAYER_THICKNESS]
        length_ratio = geometry.pore_length_ratio
        return (
            values[ParameterName.VOLUME_FRACTION] * length_ratio / accessible_width
            + accessible_width * (math.pi - 2.0 * contact_angle) / (4.0 * math.cos(contact_angle) ** 2)
            - 0.5 * accessible_width * math.tan(contact_angle)
        )

    def phi_limits(self, geometry: SlitGeometry, values: ParameterValues) -> tuple[float, float]:
        """φ_min и φ_max, уравнения (3) и (4)."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        accessible_width = 1.0 - 2.0 * values[ParameterName.WALL_LAYER_THICKNESS]
        length_ratio = geometry.pore_length_ratio
        cos_sq = math.cos(contact_angle) ** 2
        lower = (
            accessible_width**2
            / (2.0 * length_ratio)
            * (math.tan(contact_angle) - (math.pi - 2.0 * contact_angle) / (2.0 * cos_sq))
        )
        upper = accessible_width + accessible_width**2 / length_ratio * (
            1.0 / math.cos(contact_angle)
            - (math.pi - 2.0 * contact_angle) / (4.0 * cos_sq)
            - 0.5 * math.tan(contact_angle)
        )
        return lower, upper

    def feasible(self, geometry: SlitGeometry, values: ParameterValues) -> bool:
        if 1.0 - 2.0 * values[ParameterName.WALL_LAYER_THICKNESS] <= 0.0:
            return False
        lower, upper = self.phi_limits(geometry, values)
        return lower < values[ParameterName.VOLUME_FRACTION] < upper

    def center_of_mass(self, geometry: SlitGeometry, values: ParameterValues) -> float:
        return 0.0  # валик симметричен относительно середины щели

    def surface_terms(self, geometry: SlitGeometry, values: ParameterValues) -> SurfaceTerms:
        """Две дуги мениска R(2θ − π) · 2 = (1−2δ)(π − 2θ)/cos θ и два пятна контакта 2d — на длину валика k l.
        При θ = 180°, δ = 0 сумма — kl (2d + π), уравнение (21) Shistko et al. (2025).
        """
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        accessible_width = 1.0 - 2.0 * values[ParameterName.WALL_LAYER_THICKNESS]
        length = geometry.short_side_ratio()
        return SurfaceTerms(
            liquid_liquid_area=length * accessible_width * (math.pi - 2.0 * contact_angle) / math.cos(contact_angle),
            liquid_solid_area=2.0 * self.contact_spot_width(geometry, values) * length,
        )

    def profile(self, coordinate: FloatArray, geometry: SlitGeometry, values: ParameterValues) -> FloatArray:
        """ρ/ρ_bulk, уравнение (2): (d + √((1−2δ)²/cos²θ − 4z²) + (1−2δ) tan θ) / l при |z| < 1/2 − δ."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        wall_layer = values[ParameterName.WALL_LAYER_THICKNESS]
        accessible_width = 1.0 - 2.0 * wall_layer
        inside = np.abs(coordinate) < SLIT_HALF_WIDTH - wall_layer
        meniscus = np.sqrt(np.maximum(accessible_width**2 / math.cos(contact_angle) ** 2 - 4.0 * coordinate**2, 0.0))
        density = (
            self.contact_spot_width(geometry, values) + meniscus + accessible_width * math.tan(contact_angle)
        ) / geometry.pore_length_ratio
        return np.where(inside, density, 0.0)
