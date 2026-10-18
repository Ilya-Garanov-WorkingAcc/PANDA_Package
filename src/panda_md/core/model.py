"""A shape placed into a concrete pore — the object that is fitted."""

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from panda_md.core.enums import LossFunction, Normalization, OptimizerMethod, ParameterName, Regime
from panda_md.core.errors import (
    InfeasibleParametersError,
    MissingParameterValueError,
    PandaError,
    UnsupportedRegimeError,
)
from panda_md.core.parameters import PROFILE_SHIFT, ParameterSet
from panda_md.core.profile import DensityProfile
from panda_md.core.shape import InterfaceShape, SurfaceTerms
from panda_md.core.types import FloatArray

if TYPE_CHECKING:
    from panda_md.core.pore import Pore
    from panda_md.core.result import FitResult


class InterfaceModel:
    """A 'pore + shape' pair. Parameter values passed to the methods are in user units."""

    pore: "Pore"
    # Any: the shape is parameterized by the geometry of its pore; consistency is guaranteed by the pore registry.
    shape: InterfaceShape[Any]
    parameters: ParameterSet

    def __init__(self, pore: "Pore", shape: InterfaceShape[Any]) -> None:
        self.pore = pore
        self.shape = shape
        self.parameters = ParameterSet([*shape.parameters, PROFILE_SHIFT])

    def __repr__(self) -> str:
        return f"InterfaceModel(pore={self.pore.name}, shape={self.shape.name})"

    @property
    def default_optimize(self) -> tuple[str, ...]:
        return tuple(name for name in self.shape.default_optimize if name in self.parameters.names)

    def internal_values(self, user_values: Mapping[str, float]) -> dict[str, float]:
        """Values of all parameters in internal units; missing ones take their defaults."""
        self.parameters.require_known(user_values)
        resolved: dict[str, float] = {}
        for name in self.parameters.names:
            parameter = self.parameters[name]
            if name in user_values:
                resolved[name] = parameter.to_internal(user_values[name])
            elif parameter.default is not None:
                resolved[name] = parameter.to_internal(parameter.default)
            else:
                raise MissingParameterValueError(f"parameter '{name}' has no value; set its value")
        return resolved

    def supported_values(self, user_values: Mapping[str, float]) -> dict[str, float]:
        """Internal values if the shape has formulas derived for the angle θ; otherwise UnsupportedRegimeError.

        Formulas outside the declared regime are not called: they are either wrong (θ < 90° for slit shapes) or
        divide by 0.
        """
        values = self.internal_values(user_values)
        if ParameterName.CONTACT_ANGLE in values:
            self.shape.require_supported_contact_angle(math.degrees(values[ParameterName.CONTACT_ANGLE]))
        return values

    def _feasible_values(self, user_values: Mapping[str, float]) -> dict[str, float]:
        """Values for the area formulas: outside the existence domain the formulas give unphysical numbers."""
        values = self.supported_values(user_values)
        if not self.shape.feasible(self.pore.geometry(), values):
            raise InfeasibleParametersError(f"shape '{self.shape.name}' does not exist at {dict(user_values)}")
        return values

    def profile(self, coordinate: ArrayLike, **user_values: float) -> FloatArray:
        """Model ρ/ρ_bulk in the frame of reference of the shape formula."""
        values = self.supported_values(user_values)
        return self.shape.profile(np.asarray(coordinate, dtype=np.float64), self.pore.geometry(), values)

    def is_feasible(self, **user_values: float) -> bool:
        """Whether the shape exists; outside the declared wetting regimes it does not."""
        try:
            values = self.supported_values(user_values)
        except UnsupportedRegimeError:
            return False
        return bool(self.shape.feasible(self.pore.geometry(), values))

    def phi_limits(self, **user_values: float) -> tuple[float, float]:
        return self.shape.phi_limits(self.pore.geometry(), self.supported_values(user_values))

    def surface_terms(self, **user_values: float) -> SurfaceTerms:
        return self.shape.surface_terms(self.pore.geometry(), self._feasible_values(user_values))

    def capillary_energy(self, **user_values: float) -> float:
        values = self._feasible_values(user_values)
        terms = self.shape.surface_terms(self.pore.geometry(), values)
        if ParameterName.CONTACT_ANGLE in values:
            return terms.capillary_energy(values[ParameterName.CONTACT_ANGLE])
        if terms.liquid_solid_area > 0.0:
            raise PandaError(
                f"shape '{self.shape.name}' has no θ parameter but touches the walls (S_SL > 0): "
                "the energy cannot be computed without the angle"
            )
        return terms.liquid_liquid_area

    def fit(
        self,
        profile: DensityProfile,
        *,
        optimize: Sequence[str] | None = None,
        fixed: Mapping[str, float] | None = None,
        bounds: Mapping[str, tuple[float, float]] | None = None,
        initial: Mapping[str, float] | None = None,
        normalization: Normalization = Normalization.AUTO,
        density_bulk_per_nm3: float | None = None,
        loss: LossFunction = LossFunction.L1,
        window: tuple[float, float] | None = None,
        starts_per_parameter: int = 5,
        optimizer: OptimizerMethod = OptimizerMethod.NELDER_MEAD,
        regime: Regime | None = None,
    ) -> "FitResult":
        from panda_md.core.fitting import fit
        from panda_md.core.spec import FitSpec

        spec = FitSpec(
            optimize=None if optimize is None else tuple(optimize),
            fixed=dict(fixed or {}),
            bounds=dict(bounds or {}),
            initial=dict(initial or {}),
            normalization=normalization,
            density_bulk_per_nm3=density_bulk_per_nm3,
            loss=loss,
            window=window,
            starts_per_parameter=starts_per_parameter,
            optimizer=optimizer,
            regime=regime,
        )
        return fit(self, profile, spec)
