"""Axisymmetric slit shapes (droplet, doughnut, perforation) in a cell with base L × kL.

The formulas of these shapes are written for a square cell (Semenchuk et al., 2025). If the shape does not touch the
lateral cell boundaries, it "sees" only the short side kL: an L × kL cell is equivalent to a square kL × kL cell in
which the same liquid occupies a different volume fraction. A subclass provides the square-cell formulas and the rule
for rescaling φ; the profile, φ bounds and centre of mass in the L × kL cell are derived here. At k = 1 the
rescaling is the identity. Checked against the formulas of Shistko et al. (2025) at θ = 180°, δ = 0
(tests/unit/test_slit_rectangular_cell.py).
"""

from abc import abstractmethod

from panda_md.core.enums import ParameterName
from panda_md.core.parameters import ParameterValues
from panda_md.core.shape import InterfaceShape
from panda_md.core.types import FloatArray
from panda_md.pores.slit.pore import SlitGeometry
from panda_md.pores.slit.shape import SlitShape


class AxisymmetricSlitShape(SlitShape):
    """Base of an axisymmetric slit shape: square-cell formulas + rescaling to the L × kL cell."""

    @abstractmethod
    def square_profile(self, coordinate: FloatArray, square: SlitGeometry, values: ParameterValues) -> FloatArray:
        """ρ/ρ_bulk in a square cell with side l (k = 1)."""

    @abstractmethod
    def square_phi_limits(self, square: SlitGeometry, values: ParameterValues) -> tuple[float, float]:
        """φ interval in the square cell."""

    def square_center_of_mass(self, square: SlitGeometry, values: ParameterValues) -> float:
        return InterfaceShape.center_of_mass(self, square, values)

    def equivalent_volume_fraction(self, geometry: SlitGeometry, values: ParameterValues) -> float:
        """φ in the equivalent square cell: the liquid volume is preserved, the base area k l² → (k l)²."""
        return values[ParameterName.VOLUME_FRACTION] / geometry.base_aspect_ratio

    def volume_fraction_from_equivalent(
        self, geometry: SlitGeometry, values: ParameterValues, equivalent: float
    ) -> float:
        """Inverse rescaling: φ in the L × kL cell from φ of the equivalent square cell."""
        return equivalent * geometry.base_aspect_ratio

    def profile_from_equivalent(
        self, geometry: SlitGeometry, values: ParameterValues, coordinate: FloatArray, square_density: FloatArray
    ) -> FloatArray:
        """ρ in the L × kL cell: the same liquid cross-section, base area ratio square / cell (kl)²/(k l²) = k."""
        return geometry.base_aspect_ratio * square_density

    def square_values(self, geometry: SlitGeometry, values: ParameterValues) -> dict[str, float]:
        converted = dict(values)
        converted[ParameterName.VOLUME_FRACTION] = self.equivalent_volume_fraction(geometry, values)
        return converted

    def profile(self, coordinate: FloatArray, geometry: SlitGeometry, values: ParameterValues) -> FloatArray:
        square_density = self.square_profile(
            coordinate, geometry.equivalent_square(), self.square_values(geometry, values)
        )
        return self.profile_from_equivalent(geometry, values, coordinate, square_density)

    def phi_limits(self, geometry: SlitGeometry, values: ParameterValues) -> tuple[float, float]:
        lower, upper = self.square_phi_limits(geometry.equivalent_square(), self.square_values(geometry, values))
        return (
            self.volume_fraction_from_equivalent(geometry, values, lower),
            self.volume_fraction_from_equivalent(geometry, values, upper),
        )

    def center_of_mass(self, geometry: SlitGeometry, values: ParameterValues) -> float:
        return self.square_center_of_mass(geometry.equivalent_square(), self.square_values(geometry, values))
