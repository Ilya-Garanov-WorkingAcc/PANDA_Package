"""Все настройки одного фита. Сохраняются в результате и читаются из конфига CLI."""

from pydantic import Field

from panda_md.core.enums import LossFunction, Normalization, OptimizerMethod, Regime
from panda_md.core.types import FrozenBounds, FrozenFloats, PandaModel


class FitSpec(PandaModel):
    optimize: tuple[str, ...] | None = None  # None — параметры модели по умолчанию (θ, δ)
    fixed: FrozenFloats = Field(default_factory=dict)  # единицы пользователя
    bounds: FrozenBounds = Field(default_factory=dict)  # сузить границы на этот фит
    initial: FrozenFloats = Field(default_factory=dict)  # дополнительная стартовая точка
    normalization: Normalization = Normalization.AUTO
    density_bulk_per_nm3: float | None = Field(default=None, gt=0.0)
    loss: LossFunction = LossFunction.L1
    window: tuple[float, float] | None = None  # окно по безразмерной координате; None — вся пора
    starts_per_parameter: int = Field(default=5, ge=1)
    optimizer: OptimizerMethod = OptimizerMethod.NELDER_MEAD
    regime: Regime | None = None  # None — все режимы, объявленные формой и разрешённые границами θ
