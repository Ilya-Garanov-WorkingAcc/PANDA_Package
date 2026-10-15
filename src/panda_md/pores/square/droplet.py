"""Капли (droplet) в квадратной поре. Формулы — PANDA_Polygon `square.py` (коммит 38ff491)."""

import math
from typing import ClassVar

import numpy as np

from panda_md.core.enums import Frame, ParameterName, Regime
from panda_md.core.parameters import CONTACT_ANGLE, VOLUME_FRACTION, WALL_LAYER_THICKNESS, Parameter, ParameterValues
from panda_md.core.shape import REGIME_BOUNDARY_DEG, InterfaceShape, SurfaceTerms
from panda_md.core.types import FloatArray
from panda_md.pores.common import spherical_cap_centroid_offset, spherical_cap_volume_factor
from panda_md.pores.square.pore import SquareGeometry, SquarePore, accessible_side


@SquarePore.register
class SquareDropletOneWall(InterfaceShape[SquareGeometry]):
    """droplet_w1c0: сферический сегмент на нижней стенке, не касается других стенок и углов."""

    name: ClassVar[str] = "droplet_w1c0"
    parameters: ClassVar[tuple[Parameter, ...]] = (CONTACT_ANGLE, WALL_LAYER_THICKNESS, VOLUME_FRACTION)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: (90.0, 180.0)}
    frame: ClassVar[Frame] = Frame.PORE

    def sphere_radius(self, geometry: SquareGeometry, values: ParameterValues) -> float:
        """R из π R³ (2 − 3 cos θ + cos³ θ) / 3 = φ l (r_droplet_w1c0_square)."""
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
