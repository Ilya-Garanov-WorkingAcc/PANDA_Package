"""Червь (worm) в щели: цилиндрический сегмент на нижней стенке, однородный вдоль короткой стороны ячейки.

Формулы: Semenchuk et al., Colloids Surf. A 708 (2025) 135994, уравнения (10)–(14); границы существования —
по старой PANDA (`interface/alpha.py`, ветка θ > 90°). Профиль записан в системе поры: червь лежит на слое δ над
нижней стенкой. В ячейке L × kL ось червя идёт вдоль короткой стороны (так площадь меньше — Shistko et al., 2025),
сечение лежит в плоскости длинной стороны, поэтому от k профиль и границы не зависят.
"""

import math
from typing import ClassVar

import numpy as np

from panda_md.core.centering import SymmetryAxis
from panda_md.core.enums import AxisLocator, AxisOrientation, ParameterName, Regime
from panda_md.core.parameters import CONTACT_ANGLE, VOLUME_FRACTION, WALL_LAYER_THICKNESS, Parameter, ParameterValues
from panda_md.core.shape import SurfaceTerms
from panda_md.core.types import FloatArray
from panda_md.pores.common import circular_segment_centroid_offset, circular_segment_factor
from panda_md.pores.slit.pore import SLIT_HALF_WIDTH, SlitGeometry, SlitPore
from panda_md.pores.slit.shape import CrossSection, SlitShape


@SlitPore.register
class SlitWorm(SlitShape):
    name: ClassVar[str] = "worm"
    cross_section: ClassVar[CrossSection] = CrossSection.BAND
    symmetry_axis: ClassVar[SymmetryAxis | None] = SymmetryAxis(
        orientation=AxisOrientation.HORIZONTAL, locate=AxisLocator.LIQUID_CENTER
    )
    parameters: ClassVar[tuple[Parameter, ...]] = (CONTACT_ANGLE, WALL_LAYER_THICKNESS, VOLUME_FRACTION)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: (90.0, 180.0)}

    def cylinder_radius(self, geometry: SlitGeometry, values: ParameterValues) -> float:
        """R = √(φ l / (θ − sin θ cos θ)): площадь сечения сегмента равна φ l, уравнение (10)."""
        segment = circular_segment_factor(values[ParameterName.CONTACT_ANGLE])
        return math.sqrt(values[ParameterName.VOLUME_FRACTION] * geometry.pore_length_ratio / segment)

    def cylinder_center(self, geometry: SlitGeometry, values: ParameterValues) -> float:
        floor = -SLIT_HALF_WIDTH + values[ParameterName.WALL_LAYER_THICKNESS]
        return floor - self.cylinder_radius(geometry, values) * math.cos(values[ParameterName.CONTACT_ANGLE])

    def phi_limits(self, geometry: SlitGeometry, values: ParameterValues) -> tuple[float, float]:
        """φ ≤ min(червь не выше доступной щели, цилиндр не шире ячейки) — уравнение (12) и alpha.py."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        accessible_width = self.accessible_width(values)
        length_ratio = geometry.pore_length_ratio
        double_segment = 2.0 * contact_angle - math.sin(2.0 * contact_angle)
        by_height = accessible_width**2 * double_segment / (2.0 * length_ratio * (1.0 - math.cos(contact_angle)) ** 2)
        by_length = length_ratio * double_segment / 8.0
        return 0.0, min(by_height, by_length)

    def center_of_mass(self, geometry: SlitGeometry, values: ParameterValues) -> float:
        radius = self.cylinder_radius(geometry, values)
        offset = circular_segment_centroid_offset(radius, values[ParameterName.CONTACT_ANGLE])
        return self.cylinder_center(geometry, values) + offset

    def surface_terms(self, geometry: SlitGeometry, values: ParameterValues) -> SurfaceTerms:
        """Дуга 2Rθ и хорда 2R sin θ, умноженные на длину червя k l (короткая сторона)."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        radius = self.cylinder_radius(geometry, values)
        length = geometry.short_side_ratio()
        return SurfaceTerms(
            liquid_liquid_area=2.0 * radius * contact_angle * length,
            liquid_solid_area=2.0 * radius * math.sin(contact_angle) * length,
        )

    def profile(self, coordinate: FloatArray, geometry: SlitGeometry, values: ParameterValues) -> FloatArray:
        """ρ/ρ_bulk = 2 √(R² − (z − z_c)²) / l между опорой и верхушкой, уравнение (14)."""
        radius = self.cylinder_radius(geometry, values)
        center = self.cylinder_center(geometry, values)
        floor = -SLIT_HALF_WIDTH + values[ParameterName.WALL_LAYER_THICKNESS]
        inside = (coordinate > floor) & (coordinate < center + radius)  # строгие границы, как в alpha.py
        chord = 2.0 * np.sqrt(np.maximum(radius**2 - (coordinate - center) ** 2, 0.0))
        return np.where(inside, chord / geometry.pore_length_ratio, 0.0)
