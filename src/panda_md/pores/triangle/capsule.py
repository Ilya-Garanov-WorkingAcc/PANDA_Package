"""Капсулы (capsule) в треугольной поре: перемычка через сечение с двумя менисками на торцах.

Формулы — PANDA_Polygon `docs/model/triangle.py` (коммит 38ff491). Профиль выведен только у capsule_w3c0.
"""

import math
from typing import ClassVar

import numpy as np

from panda_md.core.enums import ParameterName, Regime
from panda_md.core.parameters import CONTACT_ANGLE, VOLUME_FRACTION, WALL_LAYER_THICKNESS, Parameter, ParameterValues
from panda_md.core.shape import SurfaceTerms
from panda_md.core.types import FloatArray
from panda_md.pores.common import circular_segment_area
from panda_md.pores.triangle.pore import (
    SQRT3,
    TRIANGLE_SECTION_AREA,
    TriangleGeometry,
    TrianglePore,
    accessible_bottom,
    accessible_top,
)
from panda_md.pores.triangle.shape import (
    NON_WETTING_ABOVE_TWO_THIRDS_PI,
    NON_WETTING_BELOW_TWO_THIRDS_PI,
    THIRD_PI,
    TriangleShape,
)

CAPSULE_PARAMETERS = (CONTACT_ANGLE, WALL_LAYER_THICKNESS, VOLUME_FRACTION)


def meniscus_radius(side: float, contact_angle: float) -> float:
    """r = −a′ / (2√3 cos θ) — мениск, опирающийся на три стенки под углом θ."""
    return -side / (2.0 * SQRT3 * math.cos(contact_angle))


@TrianglePore.register
class TriangleCapsuleThreeWalls(TriangleShape):
    """capsule_w3c0: вставка касается трёх стенок, углы пусты (θ > 2π/3)."""

    name: ClassVar[str] = "capsule_w3c0"
    parameters: ClassVar[tuple[Parameter, ...]] = CAPSULE_PARAMETERS
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: NON_WETTING_ABOVE_TWO_THIRDS_PI}

    @staticmethod
    def section_factor(contact_angle: float) -> float:
        """A / a′² = (3θ − 2π − 3 sin θ cos θ) / (12 cos² θ) — сечение вставки (три сегмента срезаны стенками)."""
        cosine = math.cos(contact_angle)
        return (3.0 * contact_angle - 2.0 * math.pi - 3.0 * math.sin(contact_angle) * cosine) / (12.0 * cosine**2)

    def insertion_length(self, geometry: TriangleGeometry, values: ParameterValues) -> float:
        """l̃ = (√3/4 φ l − V_торцов) / A, V_торцов = π a′³ (2 + 9 cos θ − 3 cos³ θ) / (72√3 cos³ θ).

        В источнике l̃ записана трижды разными выражениями (условие, площади, профиль); они тождественны.
        """
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        side = self.side(values)
        cosine = math.cos(contact_angle)
        ends_volume = math.pi * (2.0 + 9.0 * cosine - 3.0 * cosine**3) / (72.0 * SQRT3 * cosine**3) * side**3
        liquid_volume = TRIANGLE_SECTION_AREA * values[ParameterName.VOLUME_FRACTION] * geometry.pore_length_ratio
        return (liquid_volume - ends_volume) / (self.section_factor(contact_angle) * side**2)

    def feasible(self, geometry: TriangleGeometry, values: ParameterValues) -> bool:
        """cond_capsule_w3c0_triangle: 2π/3 < θ ≤ π, l̃ ≥ 0, l̃ + 2r ≤ l."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        side = self.side(values)
        if side <= 0.0 or not 2.0 * THIRD_PI < contact_angle <= math.pi:
            return False
        if self.section_factor(contact_angle) <= 0.0 or values[ParameterName.VOLUME_FRACTION] <= 0.0:
            return False
        insertion = self.insertion_length(geometry, values)
        radius = meniscus_radius(side, contact_angle)
        return insertion >= 0.0 and insertion + 2.0 * radius <= geometry.pore_length_ratio

    def surface_terms(self, geometry: TriangleGeometry, values: ParameterValues) -> SurfaceTerms:
        """S_LV = 2π r (3h − r) + 2 r (π − 3 arccos(h/r)) l̃, S_SL = 6 l̃ √(r² − h²) + 3π (r² − h²).

        h = a′ / (2√3) — расстояние от центроида до доступной стенки.
        """
        side = self.side(values)
        wall_distance = side / (2.0 * SQRT3)
        radius = meniscus_radius(side, values[ParameterName.CONTACT_ANGLE])
        insertion = self.insertion_length(geometry, values)
        half_chord_squared = radius**2 - wall_distance**2
        liquid_liquid = 2.0 * math.pi * radius * (3.0 * wall_distance - radius)
        liquid_liquid += 2.0 * radius * (math.pi - 3.0 * math.acos(wall_distance / radius)) * insertion
        liquid_solid = 6.0 * insertion * math.sqrt(half_chord_squared) + 3.0 * math.pi * half_chord_squared
        return SurfaceTerms(liquid_liquid_area=liquid_liquid, liquid_solid_area=liquid_solid)

    def profile(self, coordinate: FloatArray, geometry: TriangleGeometry, values: ParameterValues) -> FloatArray:
        """Сечение на высоте z — «стадион» 2 l̃ r_z + π r_z², у которого наклонные стенки срезают сегменты."""
        wall_layer = values[ParameterName.WALL_LAYER_THICKNESS]
        side = self.side(values)
        radius = meniscus_radius(side, values[ParameterName.CONTACT_ANGLE])
        insertion = self.insertion_length(geometry, values)
        top = accessible_top(wall_layer)
        inside = (coordinate >= accessible_bottom(wall_layer)) & (coordinate <= radius)
        section_radius = np.sqrt(np.maximum(0.0, radius**2 - coordinate**2))
        half_width = (top - coordinate) / SQRT3
        unclipped = 2.0 * insertion * section_radius + np.pi * section_radius**2
        segment = circular_segment_area(section_radius, half_width)
        clipped = 2.0 * insertion * half_width + np.pi * section_radius**2 - 2.0 * segment
        section = np.where(section_radius <= half_width, unclipped, clipped)
        return np.where(inside, section / geometry.layer_area(), 0.0)


@TrianglePore.register
class TriangleCapsuleThreeWallsThreeCorners(TriangleShape):
    """capsule_w3c3: вставка заполняет всё сечение, мениски на торцах (π/2 < θ < 2π/3).

    Перенесено как в источнике. При δ > 0 источник неточен: длина вставки и площадь контакта 3 L_ins записаны
    для a′ = 1 (сечение вставки √3/4 вместо √3 a′²/4) — вопрос автору (05_polygon_pores.md §4).
    """

    name: ClassVar[str] = "capsule_w3c3"
    parameters: ClassVar[tuple[Parameter, ...]] = CAPSULE_PARAMETERS
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: NON_WETTING_BELOW_TWO_THIRDS_PI}

    @staticmethod
    def _ends(contact_angle: float) -> tuple[float, float]:
        """(1 − 4 cos² θ, аргумент arccos) — общие для объёма и площадей торцов."""
        cosine = math.cos(contact_angle)
        argument = (1.0 - 6.0 * cosine**2) / (2.0 - 6.0 * cosine**2)
        return 1.0 - 4.0 * cosine**2, min(1.0, max(-1.0, argument))

    def insertion_length(self, geometry: TriangleGeometry, values: ParameterValues) -> float:
        """L_ins = φ l − 4 V_торцов / √3 (как в источнике)."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        cosine = math.cos(contact_angle)
        root_argument, arccos_argument = self._ends(contact_angle)
        ends_volume = -(3.0 * math.acos(arccos_argument) - math.pi) / (36.0 * SQRT3 * cosine**3)
        ends_volume += cosine / (3.0 * math.sqrt(root_argument))
        ends_volume *= self.side(values) ** 3
        return values[ParameterName.VOLUME_FRACTION] * geometry.pore_length_ratio - 4.0 * ends_volume / SQRT3

    def feasible(self, geometry: TriangleGeometry, values: ParameterValues) -> bool:
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        side = self.side(values)
        if side <= 0.0 or not math.pi / 2.0 < contact_angle < 2.0 * THIRD_PI:
            return False
        if self._ends(contact_angle)[0] <= 0.0 or values[ParameterName.VOLUME_FRACTION] <= 0.0:
            return False
        insertion = self.insertion_length(geometry, values)
        radius = meniscus_radius(side, contact_angle)
        return radius > 0.0 and insertion >= 0.0 and insertion + 2.0 * radius <= geometry.pore_length_ratio

    def surface_terms(self, geometry: TriangleGeometry, values: ParameterValues) -> SurfaceTerms:
        """S_LV — свободные поверхности торцов; S_SL — контакт на торцах + 3 L_ins по стенкам вставки."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        cosine = math.cos(contact_angle)
        tangent = math.tan(contact_angle)
        root_argument, arccos_argument = self._ends(contact_angle)
        side_squared = self.side(values) ** 2
        arcsin_argument = min(1.0, max(-1.0, SQRT3 / tangent))
        contact_ends = (
            -math.sqrt(3.0 * root_argument) / cosine - tangent**2 * math.asin(arcsin_argument)
        ) * side_squared
        free_ends = (3.0 * math.acos(arccos_argument) - math.pi) / (6.0 * cosine**2) * side_squared
        return SurfaceTerms(
            liquid_liquid_area=free_ends,
            liquid_solid_area=contact_ends + 3.0 * self.insertion_length(geometry, values),
        )
