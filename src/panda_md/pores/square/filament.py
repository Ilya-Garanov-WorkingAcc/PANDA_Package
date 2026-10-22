"""Filaments in a square pore. Formulas: PANDA_Polygon `square.py` (commit 38ff491)."""

import math
from typing import ClassVar

import numpy as np

from panda_md.core.enums import ParameterName, Regime
from panda_md.core.parameters import CONTACT_ANGLE, VOLUME_FRACTION, WALL_LAYER_THICKNESS, Parameter, ParameterValues
from panda_md.core.shape import SurfaceTerms
from panda_md.core.types import FloatArray
from panda_md.pores.common import circular_segment_centroid_offset, circular_segment_factor
from panda_md.pores.square.pore import SquareGeometry, SquarePore, accessible_side
from panda_md.pores.square.shape import SquareShape


@SquarePore.register
class SquareFilamentOneWall(SquareShape):
    """filament_w1c0: a cylindrical segment along the pore axis on the lower wall, the corners are free."""

    name: ClassVar[str] = "filament_w1c0"
    parameters: ClassVar[tuple[Parameter, ...]] = (CONTACT_ANGLE, WALL_LAYER_THICKNESS, VOLUME_FRACTION)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {
        Regime.WETTING: (0.0, 90.0),
        Regime.NON_WETTING: (90.0, 180.0),
    }

    def arc_radius(self, geometry: SquareGeometry, values: ParameterValues) -> float:
        """R = √(φ / (θ − sin θ cos θ)): the segment area equals φ · a²."""
        return math.sqrt(
            values[ParameterName.VOLUME_FRACTION] / circular_segment_factor(values[ParameterName.CONTACT_ANGLE])
        )

    def arc_center(self, geometry: SquareGeometry, values: ParameterValues) -> float:
        floor = -0.5 * accessible_side(values[ParameterName.WALL_LAYER_THICKNESS])
        return floor - self.arc_radius(geometry, values) * math.cos(values[ParameterName.CONTACT_ANGLE])

    def feasible(self, geometry: SquareGeometry, values: ParameterValues) -> bool:
        """cond_filament_w1c0_square: φ > 0 and the chord 2R is shorter than the accessible side a′."""
        side = accessible_side(values[ParameterName.WALL_LAYER_THICKNESS])
        if side <= 0.0 or circular_segment_factor(values[ParameterName.CONTACT_ANGLE]) <= 0.0:
            return False
        if values[ParameterName.VOLUME_FRACTION] <= 0.0:
            return False
        return 2.0 * self.arc_radius(geometry, values) < side

    def center_of_mass(self, geometry: SquareGeometry, values: ParameterValues) -> float:
        """Segment centroid: 4 R sin³θ / (3 (2θ − sin 2θ)) from the circle centre."""
        radius = self.arc_radius(geometry, values)
        offset = circular_segment_centroid_offset(radius, values[ParameterName.CONTACT_ANGLE])
        return self.arc_center(geometry, values) + offset

    def surface_terms(self, geometry: SquareGeometry, values: ParameterValues) -> SurfaceTerms:
        """S_LV = 2 R θ l, S_SL = 2 R sin θ l (area_filament_w1c0_square)."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        radius = self.arc_radius(geometry, values)
        return SurfaceTerms(
            liquid_liquid_area=2.0 * radius * contact_angle * geometry.pore_length_ratio,
            liquid_solid_area=2.0 * radius * math.sin(contact_angle) * geometry.pore_length_ratio,
        )

    def profile(self, coordinate: FloatArray, geometry: SquareGeometry, values: ParameterValues) -> FloatArray:
        """ρ/ρ_bulk = 2 √(R² − (z − z_c)²) / ly (rho_filament_w1c0_square)."""
        radius = self.arc_radius(geometry, values)
        center = self.arc_center(geometry, values)
        floor = -0.5 * accessible_side(values[ParameterName.WALL_LAYER_THICKNESS])
        inside = (coordinate >= floor) & (coordinate <= center + radius)
        chord = 2.0 * np.sqrt(np.maximum(radius**2 - (coordinate - center) ** 2, 0.0))
        return np.where(inside, chord / geometry.layer_width_ratio, 0.0)


@SquarePore.register
class SquareFilamentNoWall(SquareShape):
    """filament_w0c0: a circular cylinder along the channel axis that touches no walls — no contact angle."""

    name: ClassVar[str] = "filament_w0c0"
    parameters: ClassVar[tuple[Parameter, ...]] = (WALL_LAYER_THICKNESS, VOLUME_FRACTION)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {}
    default_optimize: ClassVar[tuple[str, ...]] = (ParameterName.VOLUME_FRACTION.value,)

    @staticmethod
    def cylinder_radius(values: ParameterValues) -> float:
        """π r² = φ — the disk area equals the fraction φ of the cross-section."""
        return math.sqrt(values[ParameterName.VOLUME_FRACTION] / math.pi)

    def feasible(self, geometry: SquareGeometry, values: ParameterValues) -> bool:
        """0 < φ < π a′² / 4 — the disk is inscribed in the accessible square (cond_filament_w0c0_square)."""
        side = self.side(values)
        volume_fraction = values[ParameterName.VOLUME_FRACTION]
        return side > 0.0 and 0.0 < volume_fraction < math.pi * side**2 / 4.0

    def surface_terms(self, geometry: SquareGeometry, values: ParameterValues) -> SurfaceTerms:
        """S_LV = 2π r l, S_SL = 0."""
        liquid_liquid = 2.0 * math.pi * self.cylinder_radius(values) * geometry.pore_length_ratio
        return SurfaceTerms(liquid_liquid_area=liquid_liquid, liquid_solid_area=0.0)

    def center_of_mass(self, geometry: SquareGeometry, values: ParameterValues) -> float:
        return 0.0  # the cylinder axis is at the cross-section centre

    def profile(self, coordinate: FloatArray, geometry: SquareGeometry, values: ParameterValues) -> FloatArray:
        """ρ/ρ_bulk = 2 √(r² − z²) / ly for |z| ≤ r."""
        radius = self.cylinder_radius(values)
        chord = 2.0 * np.sqrt(np.maximum(0.0, radius**2 - coordinate**2))
        return np.where(np.abs(coordinate) <= radius, chord / geometry.layer_width_ratio, 0.0)


@SquarePore.register
class SquareFilamentStraight(SquareShape):
    """filament_w0c0_straight: a cord of square cross-section b × b along the channel axis, touching no walls."""

    name: ClassVar[str] = "filament_w0c0_straight"
    parameters: ClassVar[tuple[Parameter, ...]] = (WALL_LAYER_THICKNESS, VOLUME_FRACTION)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {}
    default_optimize: ClassVar[tuple[str, ...]] = (ParameterName.VOLUME_FRACTION.value,)

    @staticmethod
    def cord_side(values: ParameterValues) -> float:
        """b² = φ."""
        return math.sqrt(values[ParameterName.VOLUME_FRACTION])

    def feasible(self, geometry: SquareGeometry, values: ParameterValues) -> bool:
        """0 < φ < a′² (cond_filament_w0c0_straight_square)."""
        side = self.side(values)
        volume_fraction = values[ParameterName.VOLUME_FRACTION]
        return side > 0.0 and 0.0 < volume_fraction < side**2

    def surface_terms(self, geometry: SquareGeometry, values: ParameterValues) -> SurfaceTerms:
        """S_LV = 4 b l, S_SL = 0."""
        liquid_liquid = 4.0 * self.cord_side(values) * geometry.pore_length_ratio
        return SurfaceTerms(liquid_liquid_area=liquid_liquid, liquid_solid_area=0.0)

    def center_of_mass(self, geometry: SquareGeometry, values: ParameterValues) -> float:
        return 0.0  # the cord is symmetric about the cross-section centre

    def profile(self, coordinate: FloatArray, geometry: SquareGeometry, values: ParameterValues) -> FloatArray:
        """ρ/ρ_bulk = b / ly for |z| ≤ b / 2."""
        cord_side = self.cord_side(values)
        return np.where(np.abs(coordinate) <= cord_side / 2.0, cord_side / geometry.layer_width_ratio, 0.0)
