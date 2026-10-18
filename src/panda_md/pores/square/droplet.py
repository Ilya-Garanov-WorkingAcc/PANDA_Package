"""Droplets in a square pore. Formulas: PANDA_Polygon `square.py` (commit 38ff491)."""

import math
from abc import abstractmethod
from typing import ClassVar

import numpy as np

from panda_md.core.enums import ParameterName, Regime
from panda_md.core.parameters import CONTACT_ANGLE, VOLUME_FRACTION, WALL_LAYER_THICKNESS, Parameter, ParameterValues
from panda_md.core.shape import REGIME_BOUNDARY_DEG, SurfaceTerms
from panda_md.core.types import FloatArray
from panda_md.pores.common import circular_segment_area, spherical_cap_centroid_offset, spherical_cap_volume_factor
from panda_md.pores.square.pore import SquareGeometry, SquarePore, accessible_side
from panda_md.pores.square.shape import (
    NON_WETTING_ABOVE_THREE_QUARTERS_PI,
    NON_WETTING_BELOW_THREE_QUARTERS_PI,
    THREE_QUARTERS_PI,
    SquareShape,
)


@SquarePore.register
class SquareDropletOneWall(SquareShape):
    """droplet_w1c0: a spherical cap on the lower wall that touches no other walls or corners."""

    name: ClassVar[str] = "droplet_w1c0"
    parameters: ClassVar[tuple[Parameter, ...]] = (CONTACT_ANGLE, WALL_LAYER_THICKNESS, VOLUME_FRACTION)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: (90.0, 180.0)}

    def sphere_radius(self, geometry: SquareGeometry, values: ParameterValues) -> float:
        """R from π R³ (2 − 3 cos θ + cos³ θ) / 3 = φ l (r_droplet_w1c0_square)."""
        volume = values[ParameterName.VOLUME_FRACTION] * geometry.pore_length_ratio
        return float(
            np.cbrt(3.0 * volume / (math.pi * spherical_cap_volume_factor(values[ParameterName.CONTACT_ANGLE])))
        )

    def sphere_center(self, geometry: SquareGeometry, values: ParameterValues) -> float:
        floor = -0.5 * accessible_side(values[ParameterName.WALL_LAYER_THICKNESS])
        return floor - self.sphere_radius(geometry, values) * math.cos(values[ParameterName.CONTACT_ANGLE])

    def feasible(self, geometry: SquareGeometry, values: ParameterValues) -> bool:
        """cond_droplet_w1c0_square: θ ≥ 90°, 2R < l, 2R ≤ a′."""
        side = accessible_side(values[ParameterName.WALL_LAYER_THICKNESS])
        if side <= 0.0 or math.degrees(values[ParameterName.CONTACT_ANGLE]) < REGIME_BOUNDARY_DEG:
            return False
        if values[ParameterName.VOLUME_FRACTION] <= 0.0:
            return False
        diameter = 2.0 * self.sphere_radius(geometry, values)
        return diameter < geometry.pore_length_ratio and diameter <= side

    def center_of_mass(self, geometry: SquareGeometry, values: ParameterValues) -> float:
        radius = self.sphere_radius(geometry, values)
        return self.sphere_center(geometry, values) + spherical_cap_centroid_offset(
            radius, values[ParameterName.CONTACT_ANGLE]
        )

    def surface_terms(self, geometry: SquareGeometry, values: ParameterValues) -> SurfaceTerms:
        """S_LV = 2π R² (1 − cos θ), S_SL = π R² sin²θ (area_droplet_w1c0_square)."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        radius = self.sphere_radius(geometry, values)
        return SurfaceTerms(
            liquid_liquid_area=2.0 * math.pi * radius**2 * (1.0 - math.cos(contact_angle)),
            liquid_solid_area=math.pi * radius**2 * math.sin(contact_angle) ** 2,
        )

    def profile(self, coordinate: FloatArray, geometry: SquareGeometry, values: ParameterValues) -> FloatArray:
        """ρ/ρ_bulk = π (R² − (z − z_c)²) / (l · ly) (rho_droplet_w1c0_square)."""
        radius = self.sphere_radius(geometry, values)
        center = self.sphere_center(geometry, values)
        floor = -0.5 * accessible_side(values[ParameterName.WALL_LAYER_THICKNESS])
        inside = (coordinate >= floor) & (coordinate <= center + radius)
        section_area = math.pi * np.maximum(radius**2 - (coordinate - center) ** 2, 0.0)
        layer_area = geometry.pore_length_ratio * geometry.layer_width_ratio
        return np.where(inside, section_area / layer_area, 0.0)


class SquareCornerDroplet(SquareShape):
    """Droplet in the dihedral corner between the lower and a side wall: profile shared by droplet_w2c0 and w2c1.

    The cross-section at height z is a disk of radius r_z with a segment cut off by the side wall while
    z lies between the contact points L1 and L2 (rho_droplet_w2c0_square, rho_droplet_w2c1_square).
    """

    parameters: ClassVar[tuple[Parameter, ...]] = (CONTACT_ANGLE, WALL_LAYER_THICKNESS, VOLUME_FRACTION)

    @staticmethod
    @abstractmethod
    def volume_factor(contact_angle: float) -> float:
        """F(θ): droplet volume = 2 R³ F(θ) / 3."""

    def sphere_radius(self, geometry: SquareGeometry, values: ParameterValues) -> float:
        """R = (3 φ l / (2 F(θ)))^(1/3)."""
        volume = values[ParameterName.VOLUME_FRACTION] * geometry.pore_length_ratio
        return float(np.cbrt(3.0 * volume / (2.0 * self.volume_factor(values[ParameterName.CONTACT_ANGLE]))))

    def fits(self, geometry: SquareGeometry, values: ParameterValues) -> bool:
        """φ > 0, 2R ≤ l and R (1 − cos θ) ≤ a′ — conditions shared by both corner droplets."""
        if values[ParameterName.VOLUME_FRACTION] <= 0.0:
            return False
        radius = self.sphere_radius(geometry, values)
        cap_height = radius * (1.0 - math.cos(values[ParameterName.CONTACT_ANGLE]))
        return 2.0 * radius <= geometry.pore_length_ratio and cap_height <= self.side(values)

    def profile(self, coordinate: FloatArray, geometry: SquareGeometry, values: ParameterValues) -> FloatArray:
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        radius = self.sphere_radius(geometry, values)
        floor = self.floor(values)
        wall_distance = -radius * math.cos(contact_angle)  # from the sphere centre to each of the two walls
        center = floor + wall_distance
        cut_start = floor - math.sqrt(2.0) * radius * math.sin(contact_angle + math.pi / 4.0)  # L1_square
        cut_end = floor + math.sqrt(2.0) * radius * math.sin(contact_angle - math.pi / 4.0)  # L2_square
        inside = (coordinate >= floor) & (coordinate <= center + radius)
        section_radius_squared = np.maximum(0.0, radius**2 - (coordinate - center) ** 2)
        section = np.pi * section_radius_squared
        cut = circular_segment_area(np.sqrt(section_radius_squared), wall_distance)
        section = np.where((coordinate > cut_start) & (coordinate < cut_end), section - cut, section)
        return np.where(inside, section / geometry.layer_area(), 0.0)


@SquarePore.register
class SquareDropletTwoWalls(SquareCornerDroplet):
    """droplet_w2c0: the droplet touches two adjacent walls, the corner between them is not filled (θ > 3π/4)."""

    name: ClassVar[str] = "droplet_w2c0"
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: NON_WETTING_ABOVE_THREE_QUARTERS_PI}

    @staticmethod
    def volume_factor(contact_angle: float) -> float:
        """F = π (−cos θ)(3 − cos² θ) (r_droplet_w2c0_square)."""
        cosine = math.cos(contact_angle)
        return math.pi * (-cosine) * (3.0 - cosine**2)

    def feasible(self, geometry: SquareGeometry, values: ParameterValues) -> bool:
        """cond_droplet_w2c0_square: θ > 3π/4, 2R ≤ l, R (1 − cos θ) ≤ a′."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        if self.side(values) <= 0.0 or not THREE_QUARTERS_PI < contact_angle <= math.pi:
            return False
        return self.volume_factor(contact_angle) > 0.0 and self.fits(geometry, values)

    def surface_terms(self, geometry: SquareGeometry, values: ParameterValues) -> SurfaceTerms:
        """S_LV = −4π R² cos θ, S_SL = 2π R² sin² θ."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        radius = self.sphere_radius(geometry, values)
        return SurfaceTerms(
            liquid_liquid_area=-4.0 * math.pi * radius**2 * math.cos(contact_angle),
            liquid_solid_area=2.0 * math.pi * radius**2 * math.sin(contact_angle) ** 2,
        )


@SquarePore.register
class SquareDropletTwoWallsOneCorner(SquareCornerDroplet):
    """droplet_w2c1: the droplet fills the corner between two adjacent walls (π/2 < θ < 3π/4).

    In the source the condition allows θ = 3π/4, but the area function no longer handles the shape at θ ≥ 3π/4;
    here the bound is strict.
    """

    name: ClassVar[str] = "droplet_w2c1"
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: NON_WETTING_BELOW_THREE_QUARTERS_PI}

    @staticmethod
    def volume_factor(contact_angle: float) -> float:
        """F = arccos(ctg² θ) − cos θ (2 + sin² θ) arccos(ctg θ) + cos² θ √(−cos 2θ)."""
        cosine = math.cos(contact_angle)
        cotangent = cosine / math.sin(contact_angle)
        return (
            math.acos(cotangent**2)
            - cosine * (2.0 + math.sin(contact_angle) ** 2) * math.acos(cotangent)
            + cosine**2 * math.sqrt(-math.cos(2.0 * contact_angle))
        )

    def feasible(self, geometry: SquareGeometry, values: ParameterValues) -> bool:
        """cond_droplet_w2c1_square: π/2 < θ < 3π/4, 2R ≤ l, R (1 − cos θ) ≤ a′."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        if self.side(values) <= 0.0 or not math.pi / 2.0 < contact_angle < THREE_QUARTERS_PI:
            return False
        return self.volume_factor(contact_angle) > 0.0 and self.fits(geometry, values)

    def surface_terms(self, geometry: SquareGeometry, values: ParameterValues) -> SurfaceTerms:
        """S_LV = 2R² (arccos ctg² θ − 2 cos θ arccos ctg θ), S_SL = 2R² (sin² θ arccos ctg θ − cos θ √(−cos 2θ))."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        cosine = math.cos(contact_angle)
        cotangent = cosine / math.sin(contact_angle)
        radius_squared = self.sphere_radius(geometry, values) ** 2
        liquid_liquid = 2.0 * radius_squared * (math.acos(cotangent**2) - 2.0 * cosine * math.acos(cotangent))
        liquid_solid = (
            2.0
            * radius_squared
            * (math.sin(contact_angle) ** 2 * math.acos(cotangent) - cosine * math.sqrt(-math.cos(2.0 * contact_angle)))
        )
        return SurfaceTerms(liquid_liquid_area=liquid_liquid, liquid_solid_area=liquid_solid)
