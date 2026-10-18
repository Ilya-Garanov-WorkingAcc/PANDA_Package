"""Roll in a slit: a bridge between the walls, uniform along one axis.

Formulas: Semenchuk et al., Colloids Surf. A 725 (2025) 137586, equations (1)–(4);
at δ = 0 they coincide with Semenchuk et al., Colloids Surf. A 708 (2025) 135994, equations (15)–(18).
"""

import math
from typing import ClassVar

import numpy as np

from panda_md.core.centering import SymmetryAxis
from panda_md.core.enums import AxisLocator, AxisOrientation, ParameterName, Regime
from panda_md.core.parameters import CONTACT_ANGLE, VOLUME_FRACTION, WALL_LAYER_THICKNESS, Parameter, ParameterValues
from panda_md.core.shape import SurfaceTerms
from panda_md.core.types import FloatArray
from panda_md.pores.slit.pore import SLIT_HALF_WIDTH, SlitGeometry, SlitPore
from panda_md.pores.slit.shape import CrossSection, SlitShape


@SlitPore.register
class SlitRoll(SlitShape):
    name: ClassVar[str] = "roll"
    cross_section: ClassVar[CrossSection] = CrossSection.BAND
    symmetry_axis: ClassVar[SymmetryAxis | None] = SymmetryAxis(
        orientation=AxisOrientation.HORIZONTAL, locate=AxisLocator.LIQUID_CENTER
    )
    parameters: ClassVar[tuple[Parameter, ...]] = (CONTACT_ANGLE, WALL_LAYER_THICKNESS, VOLUME_FRACTION)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: (90.0, 180.0)}

    def contact_spot_width(self, geometry: SlitGeometry, values: ParameterValues) -> float:
        """d_roll — contact patch width, equation (1)."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        accessible_width = self.accessible_width(values)
        length_ratio = geometry.pore_length_ratio
        return (
            values[ParameterName.VOLUME_FRACTION] * length_ratio / accessible_width
            + accessible_width * (math.pi - 2.0 * contact_angle) / (4.0 * math.cos(contact_angle) ** 2)
            - 0.5 * accessible_width * math.tan(contact_angle)
        )

    def phi_limits(self, geometry: SlitGeometry, values: ParameterValues) -> tuple[float, float]:
        """φ_min and φ_max, equations (3) and (4)."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        accessible_width = self.accessible_width(values)
        length_ratio = geometry.pore_length_ratio
        cos_sq = math.cos(contact_angle) ** 2
        lower = (
            accessible_width**2
            / (2.0 * length_ratio)
            * (math.tan(contact_angle) - (math.pi - 2.0 * contact_angle) / (2.0 * cos_sq))
        )
        upper = accessible_width + accessible_width**2 / length_ratio * (
            1.0 / math.cos(contact_angle)
            - (math.pi - 2.0 * contact_angle) / (4.0 * cos_sq)
            - 0.5 * math.tan(contact_angle)
        )
        return lower, upper

    def center_of_mass(self, geometry: SlitGeometry, values: ParameterValues) -> float:
        return 0.0  # the roll is symmetric about the middle of the slit

    def surface_terms(self, geometry: SlitGeometry, values: ParameterValues) -> SurfaceTerms:
        """Two meniscus arcs R(2θ − π) · 2 = (1−2δ)(π − 2θ)/cos θ and two contact patches 2d, times the roll length k l.
        At θ = 180°, δ = 0 the sum is kl (2d + π), equation (21) of Shistko et al. (2025).
        """
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        accessible_width = self.accessible_width(values)
        length = geometry.short_side_ratio()
        return SurfaceTerms(
            liquid_liquid_area=length * accessible_width * (math.pi - 2.0 * contact_angle) / math.cos(contact_angle),
            liquid_solid_area=2.0 * self.contact_spot_width(geometry, values) * length,
        )

    def profile(self, coordinate: FloatArray, geometry: SlitGeometry, values: ParameterValues) -> FloatArray:
        """ρ/ρ_bulk, equation (2): (d + √((1−2δ)²/cos²θ − 4z²) + (1−2δ) tan θ) / l for |z| < 1/2 − δ."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        wall_layer = values[ParameterName.WALL_LAYER_THICKNESS]
        accessible_width = self.accessible_width(values)
        inside = np.abs(coordinate) < SLIT_HALF_WIDTH - wall_layer
        meniscus = np.sqrt(np.maximum(accessible_width**2 / math.cos(contact_angle) ** 2 - 4.0 * coordinate**2, 0.0))
        density = (
            self.contact_spot_width(geometry, values) + meniscus + accessible_width * math.tan(contact_angle)
        ) / geometry.pore_length_ratio
        return np.where(inside, density, 0.0)
