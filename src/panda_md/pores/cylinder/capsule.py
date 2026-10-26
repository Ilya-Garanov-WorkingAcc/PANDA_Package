"""Capsule in a cylinder: a liquid plug filling the tube cross-section, closed by two spherical menisci.

Formulas: alpha.py (`rollx`). Notation: w = 1 − 2δ (accessible diameter), R′ = w / 2, c = cos θ, s = sin θ;
the meniscus radius is r = R′ / |c|, the cap height h = r (1 − s). Derived for θ > 90° only: with θ < 90° the menisci
are concave and the alpha.py profile does not conserve the liquid volume.
"""

import math
from typing import ClassVar

import numpy as np

from panda_md.core.centering import SymmetryAxis
from panda_md.core.enums import Frame, ParameterName, Regime
from panda_md.core.parameters import CONTACT_ANGLE, VOLUME_FRACTION, WALL_LAYER_THICKNESS, Parameter, ParameterValues
from panda_md.core.shape import InterfaceShape, SurfaceTerms
from panda_md.core.types import BoolArray, FloatArray
from panda_md.pores.common import circular_segment_area
from panda_md.pores.cylinder.pore import CylinderGeometry, CylinderPore, accessible_radius


@CylinderPore.register
class CylinderCapsule(InterfaceShape[CylinderGeometry]):
    name: ClassVar[str] = "capsule"
    frame: ClassVar[Frame] = Frame.PORE
    symmetry_axis: ClassVar[SymmetryAxis | None] = None
    parameters: ClassVar[tuple[Parameter, ...]] = (CONTACT_ANGLE, WALL_LAYER_THICKNESS, VOLUME_FRACTION)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: (90.0, 180.0)}

    @staticmethod
    def accessible_diameter(values: ParameterValues) -> float:
        """w = 1 − 2δ."""
        return 2.0 * accessible_radius(values[ParameterName.WALL_LAYER_THICKNESS])

    def meniscus_radius(self, values: ParameterValues) -> float:
        """r = (1 − 2δ) / (2 |cos θ|) — a spherical meniscus meeting the wall at angle θ."""
        return self.accessible_diameter(values) / (2.0 * abs(math.cos(values[ParameterName.CONTACT_ANGLE])))

    def _caps_term(self, values: ParameterValues) -> float:
        """w (1 − s)² (2 + s) / (3 c³): volume of the two caps divided by π w² / 4 (negative for θ > 90°)."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        sine, cosine = math.sin(contact_angle), math.cos(contact_angle)
        return self.accessible_diameter(values) * (1.0 - sine) ** 2 * (2.0 + sine) / (3.0 * cosine**3)

    def plug_length(self, geometry: CylinderGeometry, values: ParameterValues) -> float:
        """d = l φ / w² + w (1 − s)² (2 + s) / (3 c³) — length of the cylindrical part (d_rollx_alpha)."""
        width = self.accessible_diameter(values)
        return values[ParameterName.VOLUME_FRACTION] * geometry.pore_length_ratio / width**2 + self._caps_term(values)

    def phi_limits(self, geometry: CylinderGeometry, values: ParameterValues) -> tuple[float, float]:
        """φ_min: d = 0 (the caps meet); φ_max: d + 2h = l (the menisci touch the periodic image of the plug)."""
        width = self.accessible_diameter(values)
        length = geometry.pore_length_ratio
        cap_height = self.meniscus_radius(values) * (1.0 - math.sin(values[ParameterName.CONTACT_ANGLE]))
        lower = -(width**2) * self._caps_term(values) / length
        upper = width**2 * (length - 2.0 * cap_height - self._caps_term(values)) / length
        return lower, upper

    def feasible(self, geometry: CylinderGeometry, values: ParameterValues) -> bool:
        if self.accessible_diameter(values) <= 0.0 or math.cos(values[ParameterName.CONTACT_ANGLE]) >= 0.0:
            return False
        lower, upper = self.phi_limits(geometry, values)
        return bool(max(lower, 0.0) < values[ParameterName.VOLUME_FRACTION] < upper)

    def center_of_mass(self, geometry: CylinderGeometry, values: ParameterValues) -> float:
        return 0.0  # the capsule is symmetric about the tube axis

    def surface_terms(self, geometry: CylinderGeometry, values: ParameterValues) -> SurfaceTerms:
        """S_LV = π w² (1 − s) / c² (two spherical caps 2π r h), S_SL = π w d (wall under the plug)."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        width = self.accessible_diameter(values)
        return SurfaceTerms(
            liquid_liquid_area=math.pi * width**2 * (1.0 - math.sin(contact_angle)) / math.cos(contact_angle) ** 2,
            liquid_solid_area=math.pi * width * self.plug_length(geometry, values),
        )

    def profile(self, coordinate: FloatArray, geometry: CylinderGeometry, values: ParameterValues) -> FloatArray:
        """ρ/ρ_bulk = (d · H₁ + 2 · segment) / (l · ly): the section of the plug by the plane z = const.

        H₁ = 2√(R′² − z²) — chord of the tube; segment — of a cap circle of radius r₁ = √(r² − z²) cut by the chord
        H₁, which lies at r · sin θ from the circle centre for every z (S_layer_rollx_alpha writes the same segment as
        r₁² (α − sin α) / 2 with α = 2 arcsin(H₁ / 2r₁)).
        """
        radius = accessible_radius(values[ParameterName.WALL_LAYER_THICKNESS])
        inside = np.abs(coordinate) < radius
        height = np.where(inside, coordinate, 0.0)  # outside the formulas are undefined (r may be < 1/2)
        half_chord = np.sqrt(radius**2 - height**2)
        meniscus_radius = self.meniscus_radius(values)
        cap_radius = np.sqrt(meniscus_radius**2 - height**2)  # r ≥ R′ > |z|
        chord_distance = meniscus_radius * math.sin(values[ParameterName.CONTACT_ANGLE])
        segment = circular_segment_area(cap_radius, chord_distance)
        section = self.plug_length(geometry, values) * 2.0 * half_chord + 2.0 * segment
        return np.where(inside, section / geometry.layer_area(), 0.0)

    def contains(self, points: FloatArray, geometry: CylinderGeometry, values: ParameterValues) -> BoolArray:
        """Points: along the tube axis (periodic with l), along the layer, along the profile; axis at y = z = 0."""
        length = geometry.pore_length_ratio
        along_axis = np.abs(points[:, 0] - length * np.round(points[:, 0] / length))
        radial_squared = points[:, 1] ** 2 + points[:, 2] ** 2
        in_tube = radial_squared < accessible_radius(values[ParameterName.WALL_LAYER_THICKNESS]) ** 2
        half_plug = 0.5 * self.plug_length(geometry, values)
        cap_radius = self.meniscus_radius(values)
        cap_center = half_plug - cap_radius * math.sin(values[ParameterName.CONTACT_ANGLE])
        in_cap = (along_axis - cap_center) ** 2 + radial_squared < cap_radius**2
        inside: BoolArray = in_tube & ((along_axis <= half_plug) | in_cap)
        return inside
