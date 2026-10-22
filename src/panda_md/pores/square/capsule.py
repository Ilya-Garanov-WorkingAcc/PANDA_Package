"""Capsules in a square pore. Formulas: PANDA_Polygon `square.py` (commit 38ff491).

capsule_w4c4 is not ported: in the source ∫ρ dz ≠ φ (0.986 φ at δ = 0 and δ = 0.03).
"""

import math
from typing import ClassVar

import numpy as np

from panda_md.core.enums import ParameterName, Regime
from panda_md.core.parameters import CONTACT_ANGLE, VOLUME_FRACTION, WALL_LAYER_THICKNESS, Parameter, ParameterValues
from panda_md.core.shape import SurfaceTerms
from panda_md.core.types import FloatArray
from panda_md.pores.common import circular_segment_area
from panda_md.pores.square.pore import SquareGeometry, SquarePore
from panda_md.pores.square.shape import NON_WETTING_ABOVE_THREE_QUARTERS_PI, THREE_QUARTERS_PI, SquareShape


@SquarePore.register
class SquareCapsuleFourWalls(SquareShape):
    """capsule_w4c0: a bridge touching four walls, the corners are empty; menisci at the ends (3π/4 ≤ θ ≤ π)."""

    name: ClassVar[str] = "capsule_w4c0"
    parameters: ClassVar[tuple[Parameter, ...]] = (CONTACT_ANGLE, WALL_LAYER_THICKNESS, VOLUME_FRACTION)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: NON_WETTING_ABOVE_THREE_QUARTERS_PI}

    def meniscus_radius(self, values: ParameterValues) -> float:
        """r = a′ / (2 cos θ′), θ′ = π − θ."""
        return self.side(values) / (2.0 * math.cos(math.pi - values[ParameterName.CONTACT_ANGLE]))

    def section_area(self, values: ParameterValues) -> float:
        """S_cross = r² (π − 4θ′ + 2 sin 2θ′) — cross-section of the insertion (S_cross_capsule_w4c0_square)."""
        supplementary_angle = math.pi - values[ParameterName.CONTACT_ANGLE]
        return self.meniscus_radius(values) ** 2 * (
            math.pi - 4.0 * supplementary_angle + 2.0 * math.sin(2.0 * supplementary_angle)
        )

    def insertion_length(self, geometry: SquareGeometry, values: ParameterValues) -> float:
        """l̃ = (φ l + 4π r³ / 3 − 2π a′ r² + π a′³ / 6) / S_cross (l_tilde_capsule_w4c0_square)."""
        side = self.side(values)
        radius = self.meniscus_radius(values)
        numerator = (
            values[ParameterName.VOLUME_FRACTION] * geometry.pore_length_ratio
            + 4.0 / 3.0 * math.pi * radius**3
            - 2.0 * math.pi * side * radius**2
            + math.pi / 6.0 * side**3
        )
        return numerator / self.section_area(values)

    def feasible(self, geometry: SquareGeometry, values: ParameterValues) -> bool:
        """cond_capsule_w4c0_square: 3π/4 ≤ θ ≤ π, l̃ ≥ 0, 2r + l̃ ≤ l."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        if self.side(values) <= 0.0 or not THREE_QUARTERS_PI <= contact_angle <= math.pi:
            return False
        if self.section_area(values) <= 0.0 or values[ParameterName.VOLUME_FRACTION] <= 0.0:
            return False
        insertion = self.insertion_length(geometry, values)
        return insertion >= 0.0 and 2.0 * self.meniscus_radius(values) + insertion <= geometry.pore_length_ratio

    def surface_terms(self, geometry: SquareGeometry, values: ParameterValues) -> SurfaceTerms:
        """S_LV = 4π r (a′ − r) + 2 r (π − 4θ′) l̃, S_SL = 4π r² − π a′² + 8 r sin θ′ · l̃."""
        side = self.side(values)
        radius = self.meniscus_radius(values)
        supplementary_angle = math.pi - values[ParameterName.CONTACT_ANGLE]
        insertion = self.insertion_length(geometry, values)
        liquid_liquid = 4.0 * math.pi * radius * (side - radius)
        liquid_liquid += insertion * 2.0 * radius * (math.pi - 4.0 * supplementary_angle)
        liquid_solid = 4.0 * math.pi * radius**2 - math.pi * side**2
        liquid_solid += insertion * 8.0 * radius * math.sin(supplementary_angle)
        return SurfaceTerms(liquid_liquid_area=liquid_liquid, liquid_solid_area=liquid_solid)

    def center_of_mass(self, geometry: SquareGeometry, values: ParameterValues) -> float:
        return 0.0  # the capsule is symmetric about the cross-section centre

    def profile(self, coordinate: FloatArray, geometry: SquareGeometry, values: ParameterValues) -> FloatArray:
        """The cross-section at height z is a "stadium" 2 l̃ r_z + π r_z² with segments cut off by the side walls."""
        floor = self.floor(values)
        half_side = -floor
        radius = self.meniscus_radius(values)
        insertion = self.insertion_length(geometry, values)
        inside = (coordinate >= floor) & (coordinate <= half_side)
        section_radius = np.sqrt(np.maximum(0.0, radius**2 - coordinate**2))
        unclipped = 2.0 * insertion * section_radius + np.pi * section_radius**2
        clipped = (
            2.0 * insertion * half_side
            + np.pi * section_radius**2
            - 2.0 * circular_segment_area(section_radius, half_side)
        )
        section = np.where(section_radius <= half_side, unclipped, clipped)
        return np.where(inside, section / geometry.layer_area(), 0.0)
