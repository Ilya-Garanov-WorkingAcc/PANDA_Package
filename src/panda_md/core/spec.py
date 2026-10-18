"""All settings of one fit. Stored in the result and read from the CLI config."""

from pydantic import Field

from panda_md.core.enums import LossFunction, Normalization, OptimizerMethod, Regime
from panda_md.core.types import FrozenBounds, FrozenFloats, PandaModel


class FitSpec(PandaModel):
    optimize: tuple[str, ...] | None = None  # None — the model's default parameters (θ, δ)
    fixed: FrozenFloats = Field(default_factory=dict)  # user units
    bounds: FrozenBounds = Field(default_factory=dict)  # narrow the bounds for this fit
    initial: FrozenFloats = Field(default_factory=dict)  # additional starting point
    normalization: Normalization = Normalization.AUTO
    density_bulk_per_nm3: float | None = Field(default=None, gt=0.0)
    loss: LossFunction = LossFunction.L1
    window: tuple[float, float] | None = None  # window in the dimensionless coordinate; None — the whole pore
    starts_per_parameter: int = Field(default=5, ge=1)
    optimizer: OptimizerMethod = OptimizerMethod.NELDER_MEAD
    regime: Regime | None = None  # None — all regimes declared by the shape and allowed by the θ bounds
