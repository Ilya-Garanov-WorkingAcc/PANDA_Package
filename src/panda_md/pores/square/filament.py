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
        """0 < φ < π a′² / 4 — the disk is inscribed in the accessible square."""
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
    """filament_w0c0_square: a cord of square cross-section b × b along the channel axis, touching no walls."""

    name: ClassVar[str] = "filament_w0c0_square"
    parameters: ClassVar[tuple[Parameter, ...]] = (WALL_LAYER_THICKNESS, VOLUME_FRACTION)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {}
    default_optimize: ClassVar[tuple[str, ...]] = (ParameterName.VOLUME_FRACTION.value,)

    @staticmethod
    def cord_side(values: ParameterValues) -> float:
        """b² = φ."""
        return math.sqrt(values[ParameterName.VOLUME_FRACTION])

    def feasible(self, geometry: SquareGeometry, values: ParameterValues) -> bool:
        """0 < φ < a′²."""
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


class SquareFilamentEmptyCorners(SquareShape):
    """Square filled except for congruent circular regions at its corners."""

    parameters: ClassVar[tuple[Parameter, ...]] = (CONTACT_ANGLE, WALL_LAYER_THICKNESS, VOLUME_FRACTION)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: (135.0, 180.0)}
    lower_corners: ClassVar[int] = 0
    upper_corners: ClassVar[int]

    def corner_geometry(self, values: ParameterValues) -> tuple[float, float, float]:
        excess = values[ParameterName.CONTACT_ANGLE] - 3.0 * math.pi / 4.0
        # Empty area per r² = sin²(excess) - (excess - sin(excess) cos(excess)).
        factor = math.sin(excess) ** 2 - circular_segment_factor(excess)
        missing = self.side(values) ** 2 - values[ParameterName.VOLUME_FRACTION]
        radius = math.sqrt(missing / ((self.lower_corners + self.upper_corners) * factor))
        contact = math.sqrt(2.0) * radius * math.sin(excess)
        return radius, contact, 2.0 * excess

    def feasible(self, geometry: SquareGeometry, values: ParameterValues) -> bool:
        side = self.side(values)
        if (
            side <= 0.0
            or values[ParameterName.CONTACT_ANGLE] <= 3.0 * math.pi / 4.0
            or not 0.0 < values[ParameterName.VOLUME_FRACTION] < side**2
        ):
            return False
        _, contact, _ = self.corner_geometry(values)
        # In the prescribed orientation, at most two empty corners share a wall.
        max_on_wall = 2 if self.upper_corners > 1 else 1
        return contact * max_on_wall < side

    def profile(self, coordinate: FloatArray, geometry: SquareGeometry, values: ParameterValues) -> FloatArray:
        side = self.side(values)
        radius, contact, _ = self.corner_geometry(values)
        offset = -radius * math.cos(values[ParameterName.CONTACT_ANGLE])
        lower = coordinate + side / 2.0
        upper = side / 2.0 - coordinate
        # Rationalize offset - sqrt(...) to avoid cancellation near 135°.
        lower_cut = (
            (contact - lower)
            * (2.0 * offset - contact - lower)
            / (offset + np.sqrt(np.maximum(radius**2 - (lower - offset) ** 2, 0.0)))
        )
        upper_cut = (
            (contact - upper)
            * (2.0 * offset - contact - upper)
            / (offset + np.sqrt(np.maximum(radius**2 - (upper - offset) ** 2, 0.0)))
        )
        width = side - self.lower_corners * np.where(lower < contact, lower_cut, 0.0)
        width -= self.upper_corners * np.where(upper < contact, upper_cut, 0.0)
        return np.where((lower >= 0.0) & (upper >= 0.0), width / geometry.layer_width_ratio, 0.0)

    def surface_terms(self, geometry: SquareGeometry, values: ParameterValues) -> SurfaceTerms:
        radius, contact, angle = self.corner_geometry(values)
        count = self.upper_corners + self.lower_corners
        length = geometry.pore_length_ratio
        return SurfaceTerms(
            liquid_liquid_area=count * radius * angle * length,
            liquid_solid_area=(4.0 * self.side(values) - 2.0 * count * contact) * length,
        )


@SquarePore.register
class SquareFilamentFourWalls(SquareFilamentEmptyCorners):
    name: ClassVar[str] = "filament_w4c0"
    lower_corners: ClassVar[int] = 2
    upper_corners: ClassVar[int] = 2


@SquarePore.register
class SquareFilamentFourWallsOneCorner(SquareFilamentEmptyCorners):
    name: ClassVar[str] = "filament_w4c1"
    lower_corners: ClassVar[int] = 1
    upper_corners: ClassVar[int] = 2


@SquarePore.register
class SquareFilamentFourWallsTwoCorners(SquareFilamentEmptyCorners):
    name: ClassVar[str] = "filament_w4c2"
    upper_corners: ClassVar[int] = 2


@SquarePore.register
class SquareFilamentFourWallsThreeCorners(SquareFilamentEmptyCorners):
    name: ClassVar[str] = "filament_w4c3"
    upper_corners: ClassVar[int] = 1


class SquareFilamentAcrossWalls(SquareShape):
    """Extruded rectangle with circular ends spanning two opposite walls."""

    parameters: ClassVar[tuple[Parameter, ...]] = (CONTACT_ANGLE, WALL_LAYER_THICKNESS, VOLUME_FRACTION)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: (90.0, 180.0)}
    end_count: ClassVar[int]

    def dimensions(self, values: ParameterValues) -> tuple[float, float, float, float]:
        side = self.side(values)
        half_angle = values[ParameterName.CONTACT_ANGLE] - math.pi / 2.0
        radius = side / (2.0 * math.sin(half_angle))
        height = (
            values[ParameterName.VOLUME_FRACTION] - self.end_count * radius**2 * circular_segment_factor(half_angle)
        ) / side
        sagitta = 2.0 * radius * math.sin(half_angle / 2.0) ** 2
        return radius, half_angle, height, sagitta

    def feasible(self, geometry: SquareGeometry, values: ParameterValues) -> bool:
        if self.side(values) <= 0.0 or values[ParameterName.CONTACT_ANGLE] <= math.pi / 2.0:
            return False
        _, _, height, sagitta = self.dimensions(values)
        return height > 0.0 and height + self.end_count * sagitta < self.side(values)

    def profile(self, coordinate: FloatArray, geometry: SquareGeometry, values: ParameterValues) -> FloatArray:
        radius, _, height, sagitta = self.dimensions(values)
        side = self.side(values)
        # Distance above the top chord. For two ends reflect across the centre.
        if self.end_count == 2:
            distance = np.abs(coordinate) - height / 2.0
            inside = distance <= sagitta
        else:
            distance = coordinate + side / 2.0 - height
            inside = (coordinate >= -side / 2.0) & (distance <= sagitta)
        radial_offset = radius - sagitta + np.maximum(distance, 0.0)
        width = 2.0 * np.sqrt(np.maximum((radius - radial_offset) * (radius + radial_offset), 0.0))
        return np.where(inside, np.where(distance <= 0.0, side, width) / geometry.layer_width_ratio, 0.0)

    def surface_terms(self, geometry: SquareGeometry, values: ParameterValues) -> SurfaceTerms:
        radius, half_angle, height, _ = self.dimensions(values)
        return SurfaceTerms(
            liquid_liquid_area=2.0 * self.end_count * radius * half_angle * geometry.pore_length_ratio,
            liquid_solid_area=(2.0 * height + (2 - self.end_count) * self.side(values)) * geometry.pore_length_ratio,
        )


@SquarePore.register
class SquareFilamentThreeWallsTwoCorners(SquareFilamentAcrossWalls):
    name: ClassVar[str] = "filament_w3c2"
    end_count: ClassVar[int] = 1


@SquarePore.register
class SquareFilamentOppositeWalls(SquareFilamentAcrossWalls):
    name: ClassVar[str] = "filament_w2c0_opp"
    end_count: ClassVar[int] = 2


@SquarePore.register
class SquareFilamentTwoWallsOneCorner(SquareShape):
    """Circular disk clipped by two adjacent walls, extruded along the pore."""

    name: ClassVar[str] = "filament_w2c1"
    parameters: ClassVar[tuple[Parameter, ...]] = (CONTACT_ANGLE, WALL_LAYER_THICKNESS, VOLUME_FRACTION)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {
        Regime.WETTING: (45.0, 90.0),
        Regime.NON_WETTING: (90.0, 135.0),
    }

    def dimensions(self, values: ParameterValues) -> tuple[float, float, float]:
        angle = values[ParameterName.CONTACT_ANGLE]
        cosine = math.cos(angle)
        excess = angle - math.pi / 4.0
        factor = circular_segment_factor(excess) + math.sin(excess) ** 2
        radius = math.sqrt(values[ParameterName.VOLUME_FRACTION] / factor)
        return radius, -radius * cosine, radius * (math.sin(angle) - cosine)

    def feasible(self, geometry: SquareGeometry, values: ParameterValues) -> bool:
        side = self.side(values)
        if (
            side <= 0.0
            or values[ParameterName.VOLUME_FRACTION] <= 0.0
            or not math.pi / 4.0 < values[ParameterName.CONTACT_ANGLE] < 3.0 * math.pi / 4.0
        ):
            return False
        radius, offset, _ = self.dimensions(values)
        return radius + offset < side

    def profile(self, coordinate: FloatArray, geometry: SquareGeometry, values: ParameterValues) -> FloatArray:
        radius, offset, _ = self.dimensions(values)
        height = coordinate - self.floor(values)
        half_chord = np.sqrt(np.maximum(radius**2 - (height - offset) ** 2, 0.0))
        width = np.clip(half_chord + offset, 0.0, 2.0 * half_chord)
        return np.where(height >= 0.0, width / geometry.layer_width_ratio, 0.0)

    def surface_terms(self, geometry: SquareGeometry, values: ParameterValues) -> SurfaceTerms:
        radius, _, contact = self.dimensions(values)
        arc = 2.0 * values[ParameterName.CONTACT_ANGLE] - math.pi / 2.0
        return SurfaceTerms(
            liquid_liquid_area=radius * arc * geometry.pore_length_ratio,
            liquid_solid_area=2.0 * contact * geometry.pore_length_ratio,
        )
