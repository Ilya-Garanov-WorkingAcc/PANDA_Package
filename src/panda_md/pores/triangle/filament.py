"""Filaments in a triangular pore: uniform along the channel axis, so the profile does not depend on l.

Formulas: PANDA_Polygon `docs/model/triangle.py` (commit 38ff491), ported verbatim. The shape name
`filament_w<N>c<M>`: touches N walls, occupies M corners. The source derives a density profile only for
filament_w0c0 and filament_w3c0; the other filaments take part in the stability diagram but not in the fit.
"""

import math
from abc import abstractmethod
from typing import ClassVar

import numpy as np

from panda_md.core.enums import ParameterName, Regime
from panda_md.core.parameters import CONTACT_ANGLE, VOLUME_FRACTION, WALL_LAYER_THICKNESS, Parameter, ParameterValues
from panda_md.core.shape import SurfaceTerms
from panda_md.core.types import FloatArray
from panda_md.pores.common import circular_segment_factor
from panda_md.pores.triangle.pore import SQRT3, TRIANGLE_SECTION_AREA, TriangleGeometry, TrianglePore, accessible_height
from panda_md.pores.triangle.shape import (
    NON_WETTING_ABOVE_TWO_THIRDS_PI,
    NON_WETTING_FULL,
    THIRD_PI,
    TriangleShape,
    corner_factor,
)

ANGLE_SHAPE_PARAMETERS = (CONTACT_ANGLE, WALL_LAYER_THICKNESS, VOLUME_FRACTION)
NO_ANGLE_PARAMETERS = (WALL_LAYER_THICKNESS, VOLUME_FRACTION)


class TriangleFilament(TriangleShape):
    """Filament: the surface areas are the cross-section perimeters multiplied by the channel length l."""

    @abstractmethod
    def perimeters(self, geometry: TriangleGeometry, values: ParameterValues) -> tuple[float, float]:
        """(P_LV, P_SL) — lengths of the liquid–liquid and liquid–solid boundaries in the cross-section."""

    def surface_terms(self, geometry: TriangleGeometry, values: ParameterValues) -> SurfaceTerms:
        liquid_liquid, liquid_solid = self.perimeters(geometry, values)
        length = geometry.pore_length_ratio
        return SurfaceTerms(liquid_liquid_area=liquid_liquid * length, liquid_solid_area=liquid_solid * length)


@TrianglePore.register
class TriangleFilamentNoWall(TriangleFilament):
    """filament_w0c0: a circular cylinder along the channel axis that touches no walls — no contact angle."""

    name: ClassVar[str] = "filament_w0c0"
    parameters: ClassVar[tuple[Parameter, ...]] = NO_ANGLE_PARAMETERS
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {}
    default_optimize: ClassVar[tuple[str, ...]] = (ParameterName.VOLUME_FRACTION.value,)

    def cylinder_radius(self, values: ParameterValues) -> float:
        """π r² = φ √3 / 4 — the disk area equals the fraction φ of the cross-section."""
        return math.sqrt(values[ParameterName.VOLUME_FRACTION] * SQRT3 / (4.0 * math.pi))

    def feasible(self, geometry: TriangleGeometry, values: ParameterValues) -> bool:
        """0 < φ ≤ π a′² / (3√3) — the disk is inscribed in the accessible triangle."""
        side = self.side(values)
        if side <= 0.0:
            return False
        volume_fraction = values[ParameterName.VOLUME_FRACTION]
        return 0.0 < volume_fraction <= math.pi * side**2 / (3.0 * SQRT3)

    def perimeters(self, geometry: TriangleGeometry, values: ParameterValues) -> tuple[float, float]:
        return 2.0 * math.pi * self.cylinder_radius(values), 0.0

    def center_of_mass(self, geometry: TriangleGeometry, values: ParameterValues) -> float:
        return 0.0  # the cylinder axis is at the centroid

    def profile(self, coordinate: FloatArray, geometry: TriangleGeometry, values: ParameterValues) -> FloatArray:
        """ρ/ρ_bulk = 2 √(r² − z²) / ly for |z| ≤ r."""
        radius = self.cylinder_radius(values)
        chord = 2.0 * np.sqrt(np.maximum(0.0, radius**2 - coordinate**2))
        return np.where(np.abs(coordinate) <= radius, chord / geometry.layer_width_ratio, 0.0)


@TrianglePore.register
class TriangleFilamentStraight(TriangleFilament):
    """filament_w0c0_straight: a straight cord along the channel axis that touches no walls — no contact angle."""

    name: ClassVar[str] = "filament_w0c0_straight"
    parameters: ClassVar[tuple[Parameter, ...]] = NO_ANGLE_PARAMETERS
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {}
    default_optimize: ClassVar[tuple[str, ...]] = (ParameterName.VOLUME_FRACTION.value,)

    def feasible(self, geometry: TriangleGeometry, values: ParameterValues) -> bool:
        """0 < φ and √φ ≤ a′."""
        side = self.side(values)
        volume_fraction = values[ParameterName.VOLUME_FRACTION]
        return side > 0.0 and volume_fraction > 0.0 and math.sqrt(volume_fraction) <= side

    def perimeters(self, geometry: TriangleGeometry, values: ParameterValues) -> tuple[float, float]:
        """P_LV = 3√φ (area_filament_w0c0_straight_triangle), S_SL = 0."""
        return 3.0 * math.sqrt(values[ParameterName.VOLUME_FRACTION]), 0.0


@TrianglePore.register
class TriangleFilamentOneWall(TriangleFilament):
    """filament_w1c0: a cylindrical segment on the base, the corners are free."""

    name: ClassVar[str] = "filament_w1c0"
    parameters: ClassVar[tuple[Parameter, ...]] = ANGLE_SHAPE_PARAMETERS
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {
        Regime.WETTING: (0.0, 90.0),
        Regime.NON_WETTING: NON_WETTING_FULL,
    }

    def arc_radius(self, values: ParameterValues) -> float:
        """R² (θ − sin θ cos θ) = φ √3 / 4."""
        segment = circular_segment_factor(values[ParameterName.CONTACT_ANGLE])
        return math.sqrt(values[ParameterName.VOLUME_FRACTION] * SQRT3 / (4.0 * segment))

    def feasible(self, geometry: TriangleGeometry, values: ParameterValues) -> bool:
        """The chord fits on the base and the arc does not touch the side walls (cond_filament_w1c0_triangle)."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        side = self.side(values)
        if circular_segment_factor(contact_angle) <= 0.0 or side <= 0.0:
            return False
        if values[ParameterName.VOLUME_FRACTION] <= 0.0:
            return False
        radius = self.arc_radius(values)
        fits_base = 2.0 * radius * math.sin(contact_angle) <= side
        # distance to the side wall as a quadratic function of height z above the base: max < 0 — no contact
        quadratic = -4.0 / 3.0
        linear = side / SQRT3 - 2.0 * radius * math.cos(contact_angle)
        constant = (radius * math.sin(contact_angle)) ** 2 - 0.25 * side**2
        top = radius * (1.0 - math.cos(contact_angle))
        largest = max(constant, quadratic * top**2 + linear * top + constant)
        vertex = -linear / (2.0 * quadratic)
        if 0.0 <= vertex <= top:
            largest = max(largest, quadratic * vertex**2 + linear * vertex + constant)
        return bool(fits_base and largest < 0.0)

    def perimeters(self, geometry: TriangleGeometry, values: ParameterValues) -> tuple[float, float]:
        """P_LV = 2Rθ, P_SL = 2R sin θ."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        radius = self.arc_radius(values)
        return 2.0 * radius * contact_angle, 2.0 * radius * math.sin(contact_angle)


@TrianglePore.register
class TriangleFilamentTwoWalls(TriangleFilament):
    """filament_w2c0: liquid in the corner between two walls without filling the corner (θ > 2π/3)."""

    name: ClassVar[str] = "filament_w2c0"
    parameters: ClassVar[tuple[Parameter, ...]] = ANGLE_SHAPE_PARAMETERS
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: NON_WETTING_ABOVE_TWO_THIRDS_PI}

    def arc_radius(self, values: ParameterValues) -> float:
        """R² (π − 2θ′ + sin 2θ′) = φ √3 / 4."""
        supplementary = self.supplementary_angle(values)
        factor = math.pi - 2.0 * supplementary + math.sin(2.0 * supplementary)
        return math.sqrt(values[ParameterName.VOLUME_FRACTION] * TRIANGLE_SECTION_AREA / factor)

    def feasible(self, geometry: TriangleGeometry, values: ParameterValues) -> bool:
        side = self.side(values)
        supplementary = self.supplementary_angle(values)
        if side <= 0.0 or values[ParameterName.CONTACT_ANGLE] <= 2.0 * THIRD_PI:
            return False
        if math.pi - 2.0 * supplementary + math.sin(2.0 * supplementary) <= 0.0:
            return False
        volume_fraction = values[ParameterName.VOLUME_FRACTION]
        if not 0.0 < volume_fraction < side**2:
            return False
        radius = self.arc_radius(values)
        chord_fits = 2.0 * radius * math.sin(THIRD_PI + supplementary) <= side
        stays_below_top = radius * (2.0 * math.cos(supplementary) + 1.0) <= SQRT3 / 2.0 * side
        return chord_fits and stays_below_top

    def perimeters(self, geometry: TriangleGeometry, values: ParameterValues) -> tuple[float, float]:
        """P_LV = 2R(π − 2θ′), P_SL = 4R sin θ′."""
        supplementary = self.supplementary_angle(values)
        radius = self.arc_radius(values)
        return 2.0 * radius * (math.pi - 2.0 * supplementary), 4.0 * radius * math.sin(supplementary)


@TrianglePore.register
class TriangleFilamentTwoWallsOneCorner(TriangleFilament):
    """filament_w2c1: liquid fills the corner between two walls (θ > π/2)."""

    name: ClassVar[str] = "filament_w2c1"
    parameters: ClassVar[tuple[Parameter, ...]] = ANGLE_SHAPE_PARAMETERS
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: NON_WETTING_FULL}

    @staticmethod
    def area_factor(supplementary: float) -> float:
        """2π/3 − θ′ + √3/2 + sin(π/3 + 2θ′) — cross-section area divided by R²."""
        return 2.0 * THIRD_PI - supplementary + SQRT3 / 2.0 + math.sin(THIRD_PI + 2.0 * supplementary)

    def arc_radius(self, values: ParameterValues) -> float:
        factor = self.area_factor(self.supplementary_angle(values))
        return math.sqrt(values[ParameterName.VOLUME_FRACTION] * TRIANGLE_SECTION_AREA / factor)

    def contact_length(self, values: ParameterValues) -> float:
        """x = 2R sin(π/3 + θ′) — patch length on each of the two walls."""
        return 2.0 * self.arc_radius(values) * math.sin(THIRD_PI + self.supplementary_angle(values))

    def feasible(self, geometry: TriangleGeometry, values: ParameterValues) -> bool:
        """As in the source: only the lower bound θ > π/2 is checked; φ < 1 (cond_filament_w2c1_triangle)."""
        side = self.side(values)
        supplementary = self.supplementary_angle(values)
        if side <= 0.0 or values[ParameterName.CONTACT_ANGLE] <= math.pi / 2.0:
            return False
        if self.area_factor(supplementary) <= 0.0:
            return False
        if not 0.0 < values[ParameterName.VOLUME_FRACTION] < 1.0:
            return False
        radius = self.arc_radius(values)
        fits_wall = self.contact_length(values) <= side
        stays_below_top = radius * (2.0 * math.cos(supplementary) + 1.0) <= SQRT3 / 2.0 * side
        return fits_wall and stays_below_top

    def perimeters(self, geometry: TriangleGeometry, values: ParameterValues) -> tuple[float, float]:
        """P_LV = R(4π/3 − 2θ′), P_SL = 2x."""
        radius = self.arc_radius(values)
        return radius * (4.0 * THIRD_PI - 2.0 * self.supplementary_angle(values)), 2.0 * self.contact_length(values)


class TriangleFilamentEmptyCorners(TriangleFilament):
    """Filaments occupying the middle of the cross-section with empty rounded corners (filament_w3c0, w3c1, w3c2)."""

    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: NON_WETTING_ABOVE_TWO_THIRDS_PI}
    empty_corner_count: ClassVar[int]  # number of empty rounded corners
    corner_area_scale: ClassVar[float]  # √3/12, √3/8 or √3/4 — divisor of the empty corner area in the source
    contact_limit_fraction: ClassVar[float]  # x ≤ (fraction) · a′

    def arc_radius(self, values: ParameterValues) -> float:
        """R² · corner_factor(θ′) = (a′² − φ) · corner_area_scale."""
        empty_area = max(0.0, self.side(values) ** 2 - values[ParameterName.VOLUME_FRACTION])
        return math.sqrt(empty_area * self.corner_area_scale / corner_factor(self.supplementary_angle(values)))

    def corner_contact(self, values: ParameterValues) -> float:
        """x = 2R sin(π/3 − θ′) — distance from the vertex to the contact point."""
        return 2.0 * self.arc_radius(values) * math.sin(THIRD_PI - self.supplementary_angle(values))

    def feasible(self, geometry: TriangleGeometry, values: ParameterValues) -> bool:
        side = self.side(values)
        if side <= 0.0 or values[ParameterName.CONTACT_ANGLE] <= 2.0 * THIRD_PI:
            return False
        if corner_factor(self.supplementary_angle(values)) <= 0.0:
            return False
        volume_fraction = values[ParameterName.VOLUME_FRACTION]
        if not 0.0 < volume_fraction < side**2 or self.arc_radius(values) <= 0.0:
            return False
        return self.corner_contact(values) <= self.contact_limit_fraction * side

    def perimeters(self, geometry: TriangleGeometry, values: ParameterValues) -> tuple[float, float]:
        """P_LV = 2N R(π/3 − θ′), P_SL = 3a′ − 2N x (N — number of empty corners)."""
        corners = self.empty_corner_count
        radius = self.arc_radius(values)
        liquid_liquid = 2.0 * corners * radius * (THIRD_PI - self.supplementary_angle(values))
        return liquid_liquid, 3.0 * self.side(values) - 2.0 * corners * self.corner_contact(values)


@TrianglePore.register
class TriangleFilamentThreeWalls(TriangleFilamentEmptyCorners):
    """filament_w3c0: the middle of the cross-section is occupied, all three corners are empty."""

    name: ClassVar[str] = "filament_w3c0"
    parameters: ClassVar[tuple[Parameter, ...]] = ANGLE_SHAPE_PARAMETERS
    empty_corner_count: ClassVar[int] = 3
    corner_area_scale: ClassVar[float] = SQRT3 / 12.0
    contact_limit_fraction: ClassVar[float] = 0.5

    def profile(self, coordinate: FloatArray, geometry: TriangleGeometry, values: ParameterValues) -> FloatArray:
        """Three segments: lower rounded corners, full triangle width, upper rounded corner."""
        supplementary = self.supplementary_angle(values)
        side = self.side(values)
        height = accessible_height(values[ParameterName.WALL_LAYER_THICKNESS])
        radius = self.arc_radius(values)
        bottom, top = -height / 3.0, 2.0 * height / 3.0
        lower_center = bottom + radius * math.cos(supplementary)  # circle centres of the lower corners
        upper_center = top - 2.0 * radius * math.cos(supplementary)  # circle centre of the upper corner
        vertex_to_contact = 2.0 * radius * math.sin(THIRD_PI - supplementary)
        lower_width = side - 2.0 * SQRT3 * radius * math.cos(supplementary)
        lower_end = bottom + vertex_to_contact * SQRT3 / 2.0
        upper_start = top - vertex_to_contact * SQRT3 / 2.0
        upper_end = upper_center + radius

        width = np.zeros_like(coordinate, dtype=np.float64)
        in_lower = (coordinate >= bottom) & (coordinate <= lower_end)
        width[in_lower] = lower_width + 2.0 * np.sqrt(
            np.maximum(0.0, radius**2 - (coordinate[in_lower] - lower_center) ** 2)
        )
        in_middle = (coordinate > lower_end) & (coordinate <= upper_start)
        width[in_middle] = side - 2.0 / SQRT3 * (coordinate[in_middle] - bottom)
        in_upper = (coordinate > upper_start) & (coordinate <= upper_end)
        width[in_upper] = 2.0 * np.sqrt(np.maximum(0.0, radius**2 - (coordinate[in_upper] - upper_center) ** 2))
        return width / geometry.layer_width_ratio


@TrianglePore.register
class TriangleFilamentThreeWallsTwoCorners(TriangleFilamentEmptyCorners):
    """filament_w3c2: touches three walls, two corners occupied, one empty (source formulas)."""

    name: ClassVar[str] = "filament_w3c2"
    parameters: ClassVar[tuple[Parameter, ...]] = ANGLE_SHAPE_PARAMETERS
    empty_corner_count: ClassVar[int] = 2
    corner_area_scale: ClassVar[float] = SQRT3 / 8.0
    contact_limit_fraction: ClassVar[float] = 0.5


@TrianglePore.register
class TriangleFilamentThreeWallsOneCorner(TriangleFilamentEmptyCorners):
    """filament_w3c1: touches three walls, one corner occupied (source formulas)."""

    name: ClassVar[str] = "filament_w3c1"
    parameters: ClassVar[tuple[Parameter, ...]] = ANGLE_SHAPE_PARAMETERS
    empty_corner_count: ClassVar[int] = 1
    corner_area_scale: ClassVar[float] = SQRT3 / 4.0
    contact_limit_fraction: ClassVar[float] = 1.0
