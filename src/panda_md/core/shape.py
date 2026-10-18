"""Interface shape contract: formulas only, without state, data or optimizer."""

import math
from abc import ABC, abstractmethod
from typing import ClassVar, Generic, TypeVar

import numpy as np
from pydantic import Field
from scipy.integrate import trapezoid

from panda_md.core.centering import SymmetryAxis
from panda_md.core.enums import Frame, ParameterName, Regime
from panda_md.core.errors import NotDerivedError, UnsupportedRegimeError
from panda_md.core.parameters import Parameter, ParameterValues
from panda_md.core.types import BoolArray, FloatArray, PandaModel

REGIME_BOUNDARY_DEG = 90.0  # boundary between wetting and non-wetting
REGIME_GAP_DEG = 0.5  # θ ∈ (90° − gap, 90° + gap) is not evaluated: the formulas contain tan θ and 1/cos θ
MAX_CONTACT_ANGLE_DEG = 180.0
ANGLE_TOLERANCE_DEG = 1e-9  # degrees → radians → degrees: degrees(radians(120)) = 119.99999999999999
CENTER_OF_MASS_GRID_POINTS = 20001  # grid for the numerical centre of mass if the shape gives no formula


class PoreGeometry(PandaModel, ABC):
    """Dimensionless pore geometry passed to the shape formulas."""

    @abstractmethod
    def coordinate_bounds(self) -> tuple[float, float]:
        """Dimensionless wall coordinates along the profile axis in the pore frame (at δ = 0)."""

    @abstractmethod
    def profile_integral(self) -> float:
        """∫ρ dz / φ for any shape of this pore: original cross-section area / width of the normalization layer."""


class SurfaceTerms(PandaModel):
    """Interfacial areas of the shape (dimensionless)."""

    liquid_liquid_area: float = Field(ge=0.0)  # S_LV
    liquid_solid_area: float = Field(ge=0.0)  # S_SL

    def capillary_energy(self, contact_angle_rad: float) -> float:
        """Capillary functional S_LV − cos θ · S_SL: the stable shape is chosen by its minimum."""
        return self.liquid_liquid_area - math.cos(contact_angle_rad) * self.liquid_solid_area


def regime_of(contact_angle_deg: float) -> Regime:
    return Regime.WETTING if contact_angle_deg < REGIME_BOUNDARY_DEG else Regime.NON_WETTING


GeometryT = TypeVar("GeometryT", bound=PoreGeometry)


class InterfaceShape(ABC, Generic[GeometryT]):
    """One interface shape in one pore type. Parameter values are in internal units."""

    name: ClassVar[str]
    parameters: ClassVar[tuple[Parameter, ...]]
    regimes: ClassVar[dict[Regime, tuple[float, float]]]  # θ range in degrees for each derived regime
    frame: ClassVar[Frame]  # frame of reference in which the formulas are written
    symmetry_axis: ClassVar[SymmetryAxis | None] = None  # axis for axial centering; None — none
    default_optimize: ClassVar[tuple[str, ...]] = (
        ParameterName.CONTACT_ANGLE.value,
        ParameterName.WALL_LAYER_THICKNESS.value,
    )

    @abstractmethod
    def feasible(self, geometry: GeometryT, values: ParameterValues) -> bool:
        """Whether the shape exists at these parameters."""

    def profile(self, coordinate: FloatArray, geometry: GeometryT, values: ParameterValues) -> FloatArray:
        """ρ/ρ_bulk on the dimensionless coordinate. Called only for feasible parameters."""
        raise NotDerivedError(f"density profile is not derived for shape '{self.name}'")

    def contains(self, points: FloatArray, geometry: GeometryT, values: ParameterValues) -> BoolArray:
        """Which points (n × 3, dimensionless: along the long side, the short side, the profile axis) are in the liquid.

        The symmetry axis of the shape (if any) is at the origin of the wall plane; coordinates in the wall plane are
        periodic with the cell period. Called only for feasible parameters.
        """
        raise NotDerivedError(f"three-dimensional shape (contains) is not derived for shape '{self.name}'")

    def phi_limits(self, geometry: GeometryT, values: ParameterValues) -> tuple[float, float]:
        """Feasible interval of φ at the other parameters."""
        raise NotDerivedError(f"φ interval is not derived for shape '{self.name}'")

    def surface_terms(self, geometry: GeometryT, values: ParameterValues) -> SurfaceTerms:
        raise NotDerivedError(f"surface areas are not derived for shape '{self.name}'")

    def center_of_mass(self, geometry: GeometryT, values: ParameterValues) -> float:
        """Liquid centre of mass along the profile coordinate, in the shape formula frame; numerical by default."""
        lower, upper = geometry.coordinate_bounds()
        grid = np.linspace(lower, upper, CENTER_OF_MASS_GRID_POINTS)
        density = self.profile(grid, geometry, values)
        return float(trapezoid(grid * density, grid) / trapezoid(density, grid))

    def has_phi_limits(self) -> bool:
        return type(self).phi_limits is not InterfaceShape.phi_limits

    def has_contains(self) -> bool:
        return type(self).contains is not InterfaceShape.contains

    def has_profile(self) -> bool:
        return type(self).profile is not InterfaceShape.profile

    def has_surface_terms(self) -> bool:
        return type(self).surface_terms is not InterfaceShape.surface_terms

    def require_supported_contact_angle(self, contact_angle_deg: float) -> Regime:
        """Regime of the angle if the shape has formulas derived for it; otherwise UnsupportedRegimeError."""
        if (
            not 0.0 <= contact_angle_deg <= MAX_CONTACT_ANGLE_DEG
        ):  # the tolerance below applies only to the inner bounds
            raise UnsupportedRegimeError(f"θ = {contact_angle_deg!r}° is outside [0°, {MAX_CONTACT_ANGLE_DEG:g}°]")
        if abs(contact_angle_deg - REGIME_BOUNDARY_DEG) < REGIME_GAP_DEG - ANGLE_TOLERANCE_DEG:
            raise UnsupportedRegimeError(
                f"θ = {contact_angle_deg:.2f}° is inside the gap ±{REGIME_GAP_DEG}° around 90°: "
                "the formulas are undefined there"
            )
        regime = regime_of(contact_angle_deg)
        if regime not in self.regimes:
            raise UnsupportedRegimeError(
                f"regime {regime.value} (θ = {contact_angle_deg:.4g}°) is not declared by shape '{self.name}'; "
                f"its formulas are not derived"
            )
        lower, upper = self.regimes[regime]
        if not lower - ANGLE_TOLERANCE_DEG <= contact_angle_deg <= upper + ANGLE_TOLERANCE_DEG:
            raise UnsupportedRegimeError(
                f"θ = {contact_angle_deg:.4g}° is outside the range of regime {regime.value} of shape '{self.name}' "
                f"[{lower}, {upper}]"
            )
        return regime
