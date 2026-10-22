"""Worm in a slit: a cylindrical segment on the lower wall, uniform along the short side of the cell.

Formulas: Semenchuk et al., Colloids Surf. A 708 (2025) 135994, equations (10)–(14); existence bounds follow
the old PANDA (`interface/alpha.py`, branch θ > 90°). The profile is written in the pore frame: the worm lies on
the δ layer above the lower wall. In an L × kL cell the worm axis runs along the short side (the area is smaller
that way — Shistko et al., 2025), the cross-section lies in the plane of the long side, so the profile and the
bounds do not depend on k.
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
        """R = √(φ l / (θ − sin θ cos θ)): the segment cross-section area equals φ l, equation (10)."""
        segment = circular_segment_factor(values[ParameterName.CONTACT_ANGLE])
        return math.sqrt(values[ParameterName.VOLUME_FRACTION] * geometry.pore_length_ratio / segment)

    def cylinder_center(self, geometry: SlitGeometry, values: ParameterValues) -> float:
        floor = -SLIT_HALF_WIDTH + values[ParameterName.WALL_LAYER_THICKNESS]
        return floor - self.cylinder_radius(geometry, values) * math.cos(values[ParameterName.CONTACT_ANGLE])

    def phi_limits(self, geometry: SlitGeometry, values: ParameterValues) -> tuple[float, float]:
        """φ ≤ min(worm within the accessible slit, cylinder not wider than the cell), equation (12), alpha.py."""
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
        """Arc 2Rθ and chord 2R sin θ, multiplied by the worm length k l (the short side)."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        radius = self.cylinder_radius(geometry, values)
        length = geometry.short_side_ratio()
        return SurfaceTerms(
            liquid_liquid_area=2.0 * radius * contact_angle * length,
            liquid_solid_area=2.0 * radius * math.sin(contact_angle) * length,
        )

    def profile(self, coordinate: FloatArray, geometry: SlitGeometry, values: ParameterValues) -> FloatArray:
        """ρ/ρ_bulk = 2 √(R² − (z − z_c)²) / l between the base and the top, equation (14)."""
        radius = self.cylinder_radius(geometry, values)
        center = self.cylinder_center(geometry, values)
        floor = -SLIT_HALF_WIDTH + values[ParameterName.WALL_LAYER_THICKNESS]
        inside = (coordinate > floor) & (coordinate < center + radius)  # strict bounds, as in alpha.py
        chord = 2.0 * np.sqrt(np.maximum(radius**2 - (coordinate - center) ** 2, 0.0))
        return np.where(inside, chord / geometry.pore_length_ratio, 0.0)
