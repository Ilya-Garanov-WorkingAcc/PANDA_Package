"""Droplets in a triangular pore. Formulas: PANDA_Polygon `docs/model/triangle.py` (commit 38ff491)."""

import math
from functools import lru_cache
from typing import ClassVar

import numpy as np
from scipy.integrate import quad

from panda_md.core.enums import ParameterName, Regime
from panda_md.core.parameters import CONTACT_ANGLE, VOLUME_FRACTION, WALL_LAYER_THICKNESS, Parameter, ParameterValues
from panda_md.core.shape import SurfaceTerms
from panda_md.core.types import FloatArray
from panda_md.pores.common import circular_segment_area, spherical_cap_centroid_offset
from panda_md.pores.triangle.pore import SQRT3, TRIANGLE_SECTION_AREA, TriangleGeometry, TrianglePore, accessible_bottom
from panda_md.pores.triangle.shape import (
    NON_WETTING_ABOVE_TWO_THIRDS_PI,
    NON_WETTING_FULL,
    THIRD_PI,
    TriangleShape,
)

DROPLET_PARAMETERS = (CONTACT_ANGLE, WALL_LAYER_THICKNESS, VOLUME_FRACTION)
FIXED_VOLUME_TOLERANCE = 0.015  # droplet_w3c0: volume set by geometry, φ allowed within 1.5 % (source)


def liquid_volume(geometry: TriangleGeometry, values: ParameterValues) -> float:
    """V = φ · (√3/4) · l — liquid volume in a channel of length l."""
    return values[ParameterName.VOLUME_FRACTION] * TRIANGLE_SECTION_AREA * geometry.pore_length_ratio


def accessible_height_of(side: float) -> float:
    return SQRT3 * side / 2.0


@TrianglePore.register
class TriangleDropletOneWall(TriangleShape):
    """droplet_w1c0: a spherical cap on the base that does not touch the side walls."""

    name: ClassVar[str] = "droplet_w1c0"
    parameters: ClassVar[tuple[Parameter, ...]] = DROPLET_PARAMETERS
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: NON_WETTING_FULL}

    @staticmethod
    def volume_factor(contact_angle: float) -> float:
        """π/3 (2 − 3 cos θ + cos³ θ): cap volume = R³ · (this factor)."""
        cosine = math.cos(contact_angle)
        return math.pi / 3.0 * (2.0 - 3.0 * cosine + cosine**3)

    def sphere_radius(self, geometry: TriangleGeometry, values: ParameterValues) -> float:
        return float(np.cbrt(liquid_volume(geometry, values) / self.volume_factor(values[ParameterName.CONTACT_ANGLE])))

    def sphere_center(self, geometry: TriangleGeometry, values: ParameterValues) -> float:
        bottom = accessible_bottom(values[ParameterName.WALL_LAYER_THICKNESS])
        return bottom - self.sphere_radius(geometry, values) * math.cos(values[ParameterName.CONTACT_ANGLE])

    def feasible(self, geometry: TriangleGeometry, values: ParameterValues) -> bool:
        """Sphere shorter than the channel, patch on the base, cap not touching the side walls (cond_droplet_w1c0)."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        side = self.side(values)
        if side <= 0.0 or contact_angle < math.pi / 2.0 or self.volume_factor(contact_angle) <= 0.0:
            return False
        if not 0.0 < values[ParameterName.VOLUME_FRACTION] < 1.0:
            return False
        radius = self.sphere_radius(geometry, values)
        fits_length = 2.0 * radius <= geometry.pore_length_ratio
        fits_base = 2.0 * radius * math.sin(contact_angle) <= side
        # distance to the side wall as a quadratic function of height above the base: max < 0 — no contact
        quadratic = 4.0 / 3.0
        linear = side / SQRT3 - 2.0 * radius * math.cos(contact_angle)
        constant = -0.25 * side**2 + (radius * math.sin(contact_angle)) ** 2
        top = radius * (1.0 - math.cos(contact_angle))
        largest = max(constant, -quadratic * top**2 + linear * top + constant)
        vertex = linear / (2.0 * quadratic)
        if 0.0 <= vertex <= top:
            largest = max(largest, -quadratic * vertex**2 + linear * vertex + constant)
        return bool(fits_length and fits_base and largest < 0.0)

    def surface_terms(self, geometry: TriangleGeometry, values: ParameterValues) -> SurfaceTerms:
        """S_LV = 2π R² (1 − cos θ), S_SL = π R² sin² θ."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        radius = self.sphere_radius(geometry, values)
        return SurfaceTerms(
            liquid_liquid_area=2.0 * math.pi * radius**2 * (1.0 - math.cos(contact_angle)),
            liquid_solid_area=math.pi * radius**2 * math.sin(contact_angle) ** 2,
        )

    def center_of_mass(self, geometry: TriangleGeometry, values: ParameterValues) -> float:
        radius = self.sphere_radius(geometry, values)
        return self.sphere_center(geometry, values) + spherical_cap_centroid_offset(
            radius, values[ParameterName.CONTACT_ANGLE]
        )

    def profile(self, coordinate: FloatArray, geometry: TriangleGeometry, values: ParameterValues) -> FloatArray:
        """ρ/ρ_bulk = π (R² − (z − z_c)²) / (l · ly) between the base and the top of the droplet."""
        radius = self.sphere_radius(geometry, values)
        center = self.sphere_center(geometry, values)
        bottom = accessible_bottom(values[ParameterName.WALL_LAYER_THICKNESS])
        inside = (coordinate >= bottom) & (coordinate <= center + radius)
        section_area = math.pi * np.maximum(0.0, radius**2 - (coordinate - center) ** 2)
        return np.where(inside, section_area / geometry.layer_area(), 0.0)


@TrianglePore.register
class TriangleDropletTwoWalls(TriangleShape):
    """droplet_w2c0: a droplet in the corner between two walls without filling the corner (θ > 2π/3)."""

    name: ClassVar[str] = "droplet_w2c0"
    parameters: ClassVar[tuple[Parameter, ...]] = DROPLET_PARAMETERS
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: NON_WETTING_ABOVE_TWO_THIRDS_PI}

    @staticmethod
    def volume_factor(contact_angle: float) -> float:
        """2π/3 (−cos θ)(3 − cos² θ)."""
        cosine = math.cos(contact_angle)
        return 2.0 * math.pi / 3.0 * (-cosine) * (3.0 - cosine**2)

    def sphere_radius(self, geometry: TriangleGeometry, values: ParameterValues) -> float:
        return float(np.cbrt(liquid_volume(geometry, values) / self.volume_factor(values[ParameterName.CONTACT_ANGLE])))

    def feasible(self, geometry: TriangleGeometry, values: ParameterValues) -> bool:
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        side = self.side(values)
        if side <= 0.0 or contact_angle <= 2.0 * THIRD_PI or self.volume_factor(contact_angle) <= 0.0:
            return False
        if not 0.0 < values[ParameterName.VOLUME_FRACTION] < 1.0:
            return False
        radius = self.sphere_radius(geometry, values)
        fits_length = 2.0 * radius <= geometry.pore_length_ratio
        misses_third_wall = radius * (1.0 - 2.0 * math.cos(contact_angle)) <= accessible_height_of(side)
        fits_width = radius * (math.sin(contact_angle) + SQRT3 * math.cos(contact_angle)) <= side
        return fits_length and misses_third_wall and fits_width

    def surface_terms(self, geometry: TriangleGeometry, values: ParameterValues) -> SurfaceTerms:
        """S_LV = −4π R² cos θ, S_SL = 2π R² sin² θ."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        radius = self.sphere_radius(geometry, values)
        return SurfaceTerms(
            liquid_liquid_area=-4.0 * math.pi * radius**2 * math.cos(contact_angle),
            liquid_solid_area=2.0 * math.pi * radius**2 * math.sin(contact_angle) ** 2,
        )

    def profile(self, coordinate: FloatArray, geometry: TriangleGeometry, values: ParameterValues) -> FloatArray:
        radius = self.sphere_radius(geometry, values)
        floor = accessible_bottom(values[ParameterName.WALL_LAYER_THICKNESS])
        cosine = -math.cos(values[ParameterName.CONTACT_ANGLE])
        return radius**2 * wedge_slice((coordinate - floor) / radius, cosine) / geometry.layer_area()


@TrianglePore.register
class TriangleDropletThreeWalls(TriangleShape):
    """droplet_w3c0: the droplet touches three walls; its size is set by the cross-section, φ within 1.5 % (source)."""

    name: ClassVar[str] = "droplet_w3c0"
    parameters: ClassVar[tuple[Parameter, ...]] = DROPLET_PARAMETERS
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: NON_WETTING_ABOVE_TWO_THIRDS_PI}

    def sphere_radius(self, values: ParameterValues) -> float:
        """R = −a′ / (2√3 cos θ) — a sphere touching three walls at angle θ."""
        return -self.side(values) / (2.0 * SQRT3 * math.cos(values[ParameterName.CONTACT_ANGLE]))

    def droplet_volume(self, values: ParameterValues) -> float:
        cosine = math.cos(values[ParameterName.CONTACT_ANGLE])
        return math.pi * self.sphere_radius(values) ** 3 / 3.0 * (3.0 * cosine**3 - 9.0 * cosine - 2.0)

    def feasible(self, geometry: TriangleGeometry, values: ParameterValues) -> bool:
        side = self.side(values)
        if side <= 0.0 or values[ParameterName.CONTACT_ANGLE] <= 2.0 * THIRD_PI:
            return False
        radius = self.sphere_radius(values)
        if radius <= 0.0 or not 0.0 < values[ParameterName.VOLUME_FRACTION] < 1.0:
            return False
        fixed_volume = self.droplet_volume(values)
        mismatch = abs(liquid_volume(geometry, values) - fixed_volume) / max(abs(fixed_volume), 1e-12)
        return 2.0 * radius <= geometry.pore_length_ratio and mismatch < FIXED_VOLUME_TOLERANCE

    def surface_terms(self, geometry: TriangleGeometry, values: ParameterValues) -> SurfaceTerms:
        """S_LV = −2π R² (1 + 3 cos θ), S_SL = 3π (R sin θ)²."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        radius = self.sphere_radius(values)
        return SurfaceTerms(
            liquid_liquid_area=-2.0 * math.pi * radius**2 * (1.0 + 3.0 * math.cos(contact_angle)),
            liquid_solid_area=3.0 * math.pi * (radius * math.sin(contact_angle)) ** 2,
        )


def wedge_slice(height: FloatArray, cosine: float) -> FloatArray:
    """Slice area of a unit sphere cut by the base and one inclined wall."""
    radius = np.sqrt(np.maximum(0.0, 1.0 - (height - cosine) ** 2))
    distance = (3.0 * cosine - height) / SQRT3
    area = math.pi * radius**2 - circular_segment_area(radius, distance)
    return np.where(height >= 0.0, np.maximum(area, 0.0), 0.0)


@lru_cache(maxsize=128)
def wedge_surface_factors(contact_angle: float) -> tuple[float, float]:
    """Free sphere area and contact area of both faces, for unit radius."""
    cosine = -math.cos(contact_angle)
    sine = math.sin(contact_angle)
    cutoff = 1.5 * cosine + SQRT3 / 2.0 * sine

    def azimuth_span(height: float) -> float:
        radius = math.sqrt(max(0.0, 1.0 - (height - cosine) ** 2))
        distance = (3.0 * cosine - height) / SQRT3
        return 2.0 * math.acos(max(-1.0, min(1.0, -distance / radius)))

    free = quad(azimuth_span, 0.0, cutoff, epsabs=1e-10)[0]
    distance = SQRT3 * cosine
    contact = 2.0 * (math.pi * sine**2 - float(circular_segment_area(np.array(sine), distance)))
    return free, contact


@TrianglePore.register
class TriangleDropletTwoWallsOneCorner(TriangleDropletTwoWalls):
    """Sphere intersecting two adjacent walls and wetting their common edge."""

    name: ClassVar[str] = "droplet_w2c1"
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: (90.0, 120.0)}

    @staticmethod
    def volume_factor(contact_angle: float) -> float:
        free, contact = wedge_surface_factors(contact_angle)
        return (free - math.cos(contact_angle) * contact) / 3.0

    def feasible(self, geometry: TriangleGeometry, values: ParameterValues) -> bool:
        side = self.side(values)
        angle = values[ParameterName.CONTACT_ANGLE]
        if (
            side <= 0.0
            or not math.pi / 2.0 < angle < 2.0 * THIRD_PI
            or not 0.0 < values[ParameterName.VOLUME_FRACTION] < side**2
        ):
            return False
        radius = self.sphere_radius(geometry, values)
        return 2.0 * radius <= geometry.pore_length_ratio and (
            radius * (1.0 - 2.0 * math.cos(angle)) < accessible_height_of(side)
        )

    def surface_terms(self, geometry: TriangleGeometry, values: ParameterValues) -> SurfaceTerms:
        radius = self.sphere_radius(geometry, values)
        free, contact = wedge_surface_factors(values[ParameterName.CONTACT_ANGLE])
        return SurfaceTerms(liquid_liquid_area=free * radius**2, liquid_solid_area=contact * radius**2)
