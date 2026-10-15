"""Капли (droplet) в треугольной поре. Формулы — PANDA_Polygon `docs/model/triangle.py` (коммит 38ff491).

Профиль плотности в источнике выведен только у droplet_w1c0; остальные капли — для диаграммы устойчивости.
"""

import math
from typing import ClassVar

import numpy as np

from panda_md.core.enums import ParameterName, Regime
from panda_md.core.parameters import CONTACT_ANGLE, VOLUME_FRACTION, WALL_LAYER_THICKNESS, Parameter, ParameterValues
from panda_md.core.shape import SurfaceTerms
from panda_md.core.types import FloatArray
from panda_md.pores.common import spherical_cap_centroid_offset
from panda_md.pores.triangle.pore import SQRT3, TRIANGLE_SECTION_AREA, TriangleGeometry, TrianglePore, accessible_bottom
from panda_md.pores.triangle.shape import (
    NON_WETTING_ABOVE_TWO_THIRDS_PI,
    NON_WETTING_BELOW_TWO_THIRDS_PI,
    NON_WETTING_FULL,
    THIRD_PI,
    TriangleShape,
)

DROPLET_PARAMETERS = (CONTACT_ANGLE, WALL_LAYER_THICKNESS, VOLUME_FRACTION)
FIXED_VOLUME_TOLERANCE = 0.015  # droplet_w3c0: объём задан геометрией, φ допускается в пределах 1.5 % (источник)


def liquid_volume(geometry: TriangleGeometry, values: ParameterValues) -> float:
    """V = φ · (√3/4) · l — объём жидкости в канале длины l."""
    return values[ParameterName.VOLUME_FRACTION] * TRIANGLE_SECTION_AREA * geometry.pore_length_ratio


def accessible_height_of(side: float) -> float:
    return SQRT3 * side / 2.0


@TrianglePore.register
class TriangleDropletOneWall(TriangleShape):
    """droplet_w1c0: сферический сегмент на основании, боковых стенок не касается."""

    name: ClassVar[str] = "droplet_w1c0"
    parameters: ClassVar[tuple[Parameter, ...]] = DROPLET_PARAMETERS
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: NON_WETTING_FULL}

    @staticmethod
    def volume_factor(contact_angle: float) -> float:
        """π/3 (2 − 3 cos θ + cos³ θ): объём сегмента = R³ · (этот множитель)."""
        cosine = math.cos(contact_angle)
        return math.pi / 3.0 * (2.0 - 3.0 * cosine + cosine**3)

    def sphere_radius(self, geometry: TriangleGeometry, values: ParameterValues) -> float:
        return float(np.cbrt(liquid_volume(geometry, values) / self.volume_factor(values[ParameterName.CONTACT_ANGLE])))

    def sphere_center(self, geometry: TriangleGeometry, values: ParameterValues) -> float:
        bottom = accessible_bottom(values[ParameterName.WALL_LAYER_THICKNESS])
        return bottom - self.sphere_radius(geometry, values) * math.cos(values[ParameterName.CONTACT_ANGLE])

    def feasible(self, geometry: TriangleGeometry, values: ParameterValues) -> bool:
        """Сфера короче канала, пятно на основании, сегмент не касается боковых стенок (cond_droplet_w1c0)."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        side = self.side(values)
        if side <= 0.0 or contact_angle < math.pi / 2.0 or self.volume_factor(contact_angle) <= 0.0:
            return False
        if not 0.0 < values[ParameterName.VOLUME_FRACTION] < 1.0:
            return False
        radius = self.sphere_radius(geometry, values)
        fits_length = 2.0 * radius <= geometry.pore_length_ratio
        fits_base = 2.0 * radius * math.sin(contact_angle) <= side
        # расстояние до боковой стенки как квадратичная функция высоты над основанием: max < 0 — касания нет
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
        """ρ/ρ_bulk = π (R² − (z − z_c)²) / (l · ly) между основанием и верхушкой капли."""
        radius = self.sphere_radius(geometry, values)
        center = self.sphere_center(geometry, values)
        bottom = accessible_bottom(values[ParameterName.WALL_LAYER_THICKNESS])
        inside = (coordinate >= bottom) & (coordinate <= center + radius)
        section_area = math.pi * np.maximum(0.0, radius**2 - (coordinate - center) ** 2)
        return np.where(inside, section_area / geometry.layer_area(), 0.0)


@TrianglePore.register
class TriangleDropletTwoWalls(TriangleShape):
    """droplet_w2c0: капля в угле между двумя стенками без заполнения угла (θ > 2π/3)."""

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


@TrianglePore.register
class TriangleDropletTwoWallsOneCorner(TriangleShape):
    """droplet_w2c1: капля заполняет угол между двумя стенками (π/2 < θ < 2π/3).

    Отступление от источника: при θ > 2 arctg(3/2) ≈ 112.6° формула S_LV = 2π R² (1 + cos θ − ⅔ sin θ)
    отрицательна, а cond_droplet_w2c1_triangle форму допускает; здесь форма там не существует
    (05_polygon_pores.md §4, вопрос автору).
    """

    name: ClassVar[str] = "droplet_w2c1"
    parameters: ClassVar[tuple[Parameter, ...]] = DROPLET_PARAMETERS
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: NON_WETTING_BELOW_TWO_THIRDS_PI}

    @staticmethod
    def area_factors(contact_angle: float) -> tuple[float, float]:
        """(S_LV, S_SL) / R²: 2π (1 + cos θ − 2 sin θ / 3) и 2π sin² θ."""
        liquid_liquid = 2.0 * math.pi * (1.0 + math.cos(contact_angle) - 2.0 * math.sin(contact_angle) / 3.0)
        return liquid_liquid, 2.0 * math.pi * math.sin(contact_angle) ** 2

    def volume_factor(self, contact_angle: float) -> float:
        """(S_LV − cos θ · S_SL) / (3 R²)."""
        liquid_liquid, liquid_solid = self.area_factors(contact_angle)
        return (liquid_liquid - liquid_solid * math.cos(contact_angle)) / 3.0

    def sphere_radius(self, geometry: TriangleGeometry, values: ParameterValues) -> float:
        return float(np.cbrt(liquid_volume(geometry, values) / self.volume_factor(values[ParameterName.CONTACT_ANGLE])))

    def feasible(self, geometry: TriangleGeometry, values: ParameterValues) -> bool:
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        side = self.side(values)
        if side <= 0.0 or not math.pi / 2.0 < contact_angle < 2.0 * THIRD_PI:
            return False
        if self.volume_factor(contact_angle) <= 0.0 or self.area_factors(contact_angle)[0] <= 0.0:
            return False
        if not 0.0 < values[ParameterName.VOLUME_FRACTION] < side**2:
            return False
        radius = self.sphere_radius(geometry, values)
        fits_length = 2.0 * radius <= geometry.pore_length_ratio
        misses_third_wall = radius * (1.0 - 2.0 * math.cos(contact_angle)) <= accessible_height_of(side)
        fits_width = radius * (math.sin(contact_angle) - SQRT3 * math.cos(contact_angle)) <= side
        return fits_length and misses_third_wall and fits_width

    def surface_terms(self, geometry: TriangleGeometry, values: ParameterValues) -> SurfaceTerms:
        liquid_liquid, liquid_solid = self.area_factors(values[ParameterName.CONTACT_ANGLE])
        radius_squared = self.sphere_radius(geometry, values) ** 2
        return SurfaceTerms(
            liquid_liquid_area=liquid_liquid * radius_squared, liquid_solid_area=liquid_solid * radius_squared
        )


@TrianglePore.register
class TriangleDropletThreeWalls(TriangleShape):
    """droplet_w3c0: капля касается трёх стенок; её размер задан сечением, φ — с допуском 1.5 % (источник)."""

    name: ClassVar[str] = "droplet_w3c0"
    parameters: ClassVar[tuple[Parameter, ...]] = DROPLET_PARAMETERS
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: NON_WETTING_ABOVE_TWO_THIRDS_PI}

    def sphere_radius(self, values: ParameterValues) -> float:
        """R = −a′ / (2√3 cos θ) — сфера, касающаяся трёх стенок под углом θ."""
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
