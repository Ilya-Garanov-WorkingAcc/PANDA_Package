"""Форма, подставленная в конкретную пору, — объект, который фитируют."""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from panda_md.core.enums import LossFunction, Normalization, OptimizerMethod, ParameterName, Regime
from panda_md.core.errors import MissingParameterValueError
from panda_md.core.parameters import PROFILE_SHIFT, ParameterSet
from panda_md.core.profile import DensityProfile
from panda_md.core.shape import InterfaceShape, SurfaceTerms
from panda_md.core.types import FloatArray

if TYPE_CHECKING:
    from panda_md.core.pore import Pore
    from panda_md.core.result import FitResult


class InterfaceModel:
    """Связка «пора + форма». Значения параметров на входе методов — в единицах пользователя."""

    pore: "Pore"
    # Any: форма параметризована геометрией своей поры; согласованность гарантирует реестр поры.
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
        """Значения всех параметров во внутренних единицах; пропущенные — по умолчанию."""
        self.parameters.require_known(user_values)
        resolved: dict[str, float] = {}
        for name in self.parameters.names:
            parameter = self.parameters[name]
            if name in user_values:
                resolved[name] = parameter.to_internal(user_values[name])
            elif parameter.default is not None:
                resolved[name] = parameter.to_internal(parameter.default)
            else:
                raise MissingParameterValueError(f"задай значение параметра {name}")
        return resolved

    def profile(self, coordinate: ArrayLike, **user_values: float) -> FloatArray:
        """Модельный ρ/ρ_bulk в системе отсчёта формулы формы."""
        values = self.internal_values(user_values)
        return self.shape.profile(np.asarray(coordinate, dtype=np.float64), self.pore.geometry(), values)

    def is_feasible(self, **user_values: float) -> bool:
        return bool(self.shape.feasible(self.pore.geometry(), self.internal_values(user_values)))

    def phi_limits(self, **user_values: float) -> tuple[float, float]:
        return self.shape.phi_limits(self.pore.geometry(), self.internal_values(user_values))

    def surface_terms(self, **user_values: float) -> SurfaceTerms:
        return self.shape.surface_terms(self.pore.geometry(), self.internal_values(user_values))

    def capillary_energy(self, **user_values: float) -> float:
        values = self.internal_values(user_values)
        terms = self.shape.surface_terms(self.pore.geometry(), values)
        if ParameterName.CONTACT_ANGLE not in values:
            return terms.liquid_liquid_area + terms.liquid_solid_area  # формы без θ не касаются стенок: S_SL = 0
        return terms.capillary_energy(values[ParameterName.CONTACT_ANGLE])

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
