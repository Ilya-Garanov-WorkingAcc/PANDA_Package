"""Axisymmetric hole through two opposite walls of a square channel."""

import math
from functools import lru_cache
from typing import ClassVar

import numpy as np
from scipy.integrate import quad

from panda_md.core.enums import ParameterName, Regime
from panda_md.core.parameters import CONTACT_ANGLE, VOLUME_FRACTION, WALL_LAYER_THICKNESS, Parameter, ParameterValues
from panda_md.core.shape import SurfaceTerms
from panda_md.core.types import FloatArray
from panda_md.pores.square.pore import SquareGeometry, SquarePore
from panda_md.pores.square.shape import SquareShape


@lru_cache(maxsize=128)
def hole_geometry(angle: float) -> tuple[float, float, float, float]:
    """Unit-side radius, wall flare and first two moments of the radial flare."""
    radius = -0.5 / math.cos(angle)

    def flare(z: float) -> float:
        return z**2 / (radius + math.sqrt(max(0.0, radius**2 - z**2)))

    return radius, flare(0.5), 2.0 * quad(flare, 0.0, 0.5)[0], 2.0 * quad(lambda z: flare(z) ** 2, 0.0, 0.5)[0]


@SquarePore.register
class SquarePerforation(SquareShape):
    name: ClassVar[str] = "perforation"
    parameters: ClassVar[tuple[Parameter, ...]] = (CONTACT_ANGLE, WALL_LAYER_THICKNESS, VOLUME_FRACTION)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: (90.0, 180.0)}

    def dimensions(self, geometry: SquareGeometry, values: ParameterValues) -> tuple[float, float, float]:
        side = self.side(values)
        radius, flare, first, second = hole_geometry(values[ParameterName.CONTACT_ANGLE])
        gas_volume = (side**2 - values[ParameterName.VOLUME_FRACTION]) * geometry.pore_length_ratio
        target = gas_volume / (math.pi * side**3)
        if target <= second:
            return radius * side, 0.0, flare * side
        waist = (target - second) / (math.sqrt(first**2 + target - second) + first)
        return radius * side, waist * side, (waist + flare) * side

    def feasible(self, geometry: SquareGeometry, values: ParameterValues) -> bool:
        side = self.side(values)
        if (
            side <= 0.0
            or values[ParameterName.CONTACT_ANGLE] <= math.pi / 2.0
            or not 0.0 < values[ParameterName.VOLUME_FRACTION] < side**2
        ):
            return False
        _, waist, wall = self.dimensions(geometry, values)
        return waist > 0.0 and 2.0 * wall < min(side, geometry.pore_length_ratio)

    def profile(self, coordinate: FloatArray, geometry: SquareGeometry, values: ParameterValues) -> FloatArray:
        radius, waist, _ = self.dimensions(geometry, values)
        side = self.side(values)
        flare = coordinate**2 / (radius + np.sqrt(np.maximum(0.0, radius**2 - coordinate**2)))
        area = side * geometry.pore_length_ratio - math.pi * (waist + flare) ** 2
        return np.where(np.abs(coordinate) <= side / 2.0, area / geometry.layer_area(), 0.0)

    def surface_terms(self, geometry: SquareGeometry, values: ParameterValues) -> SurfaceTerms:
        radius, waist, wall = self.dimensions(geometry, values)
        half_angle = values[ParameterName.CONTACT_ANGLE] - math.pi / 2.0
        # Integrate 2π times the hole radius along the circular generating arc.
        arc_integral = waist * half_angle + radius * (half_angle - math.sin(half_angle))
        return SurfaceTerms(
            liquid_liquid_area=4.0 * math.pi * radius * arc_integral,
            liquid_solid_area=4.0 * self.side(values) * geometry.pore_length_ratio - 2.0 * math.pi * wall**2,
        )
