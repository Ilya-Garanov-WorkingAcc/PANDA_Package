"""Перебор форм: фит одного профиля всеми формами поры и таблица по качеству."""

import warnings
from collections.abc import Sequence

from pydantic import model_validator

from panda_md.core.errors import (
    InfeasibleParametersError,
    InvalidBoundsError,
    MissingParameterValueError,
    NotDerivedError,
    PandaError,
    PandaWarning,
    ParameterConflictError,
    UnidentifiableParameterError,
    UnknownParameterError,
    UnsupportedFrameError,
    UnsupportedRegimeError,
)
from panda_md.core.fitting import fit
from panda_md.core.pore import Pore
from panda_md.core.profile import DensityProfile
from panda_md.core.result import FitResult
from panda_md.core.spec import FitSpec
from panda_md.core.types import PandaModel

LOSS_TIE_FRACTION = 0.02  # ошибки форм ближе 2 % — по одному профилю формы не различить

# Ошибки, которые зависят от формы: форма пропускается с причиной. Ошибки профиля и настроек (профиль не
# совпадает с порой, пустое окно, нет ρ_bulk) одинаковы для всех форм — их перебор не глушит.
SHAPE_LEVEL_ERRORS = (
    NotDerivedError,
    InfeasibleParametersError,
    UnsupportedRegimeError,
    UnsupportedFrameError,
    UnidentifiableParameterError,
    UnknownParameterError,
    ParameterConflictError,
    MissingParameterValueError,
    InvalidBoundsError,
)


class ShapeFit(PandaModel):
    """Одна строка сравнения: результат фита или причина, по которой форма не фитировалась."""

    shape_name: str
    result: FitResult | None = None
    skipped_reason: str | None = None

    @model_validator(mode="after")
    def _one_of(self) -> "ShapeFit":
        if (self.result is None) == (self.skipped_reason is None):
            raise ValueError("нужен либо результат, либо причина пропуска")
        return self


class ShapeComparison(PandaModel):
    """Формы по возрастанию ошибки фита; пропущенные — в конце с причиной.

    Ошибки сравнимы между формами только при одинаковых данных и настройках; меньшая ошибка — довод, а не
    доказательство: у разных форм разное число свободных параметров.
    """

    pore_name: str
    rows: tuple[ShapeFit, ...]

    @property
    def fitted(self) -> tuple[FitResult, ...]:
        return tuple(row.result for row in self.rows if row.result is not None)

    @property
    def best(self) -> FitResult:
        """Допустимый результат с наименьшей ошибкой."""
        feasible = [result for result in self.fitted if result.feasible]
        if not feasible:
            reasons = "; ".join(f"{row.shape_name}: {row.skipped_reason}" for row in self.rows if row.skipped_reason)
            raise PandaError(
                f"ни одна форма не дала допустимого результата ({reasons or 'все вне области существования'})"
            )
        return feasible[0]

    @property
    def indistinguishable(self) -> tuple[str, ...]:
        """Допустимые формы с ошибкой не больше лучшей на LOSS_TIE_FRACTION: профиль их не различает."""
        feasible = [result for result in self.fitted if result.feasible]
        if not feasible:
            return ()
        threshold = feasible[0].loss * (1.0 + LOSS_TIE_FRACTION)
        tied = tuple(result.shape_name for result in feasible if result.loss <= threshold)
        return tied if len(tied) > 1 else ()

    def summary(self) -> str:
        lines = [f"{self.pore_name}: сравнение форм"]
        tied = self.indistinguishable
        for row in self.rows:
            if row.result is None:
                lines.append(f"  {row.shape_name:<24} пропущена: {row.skipped_reason}")
                continue
            result = row.result
            values = ", ".join(f"{name} = {value:.4g}" for name, value in result.values.items())
            mark = "" if result.feasible else "  (вне области существования)"
            tie = "≈ " if row.shape_name in tied else "  "
            lines.append(f"{tie}{row.shape_name:<24} loss = {result.loss:.4g}  {values}{mark}")
        if tied:
            lines.append(f"≈ ошибки в пределах {LOSS_TIE_FRACTION:.0%}: профиль не различает {', '.join(tied)}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def _ranking(row: ShapeFit) -> tuple[bool, float]:
    """Сначала допустимые результаты, внутри — по возрастанию ошибки."""
    assert row.result is not None  # сортируются только фитированные формы
    return not row.result.feasible, row.result.loss


def compare_shapes(
    pore: Pore, profile: DensityProfile, spec: FitSpec | None = None, shape_names: Sequence[str] | None = None
) -> ShapeComparison:
    """Фит профиля каждой формой поры (по умолчанию — всеми) с одними настройками."""
    settings = spec if spec is not None else FitSpec()
    names = list(pore.own_registry()) if shape_names is None else list(shape_names)
    if len(set(names)) != len(names):
        raise PandaError(f"повтор в списке форм: {names}")
    fitted: list[ShapeFit] = []
    skipped: list[ShapeFit] = []
    for name in names:
        model = pore.interface(name)
        try:
            fitted.append(ShapeFit(shape_name=name, result=fit(model, profile, settings)))
        except SHAPE_LEVEL_ERRORS as error:  # нет профиля, нет допустимого старта, режим не объявлен, …
            skipped.append(ShapeFit(shape_name=name, skipped_reason=str(error)))
    fitted.sort(key=_ranking)
    comparison = ShapeComparison(pore_name=pore.name, rows=(*fitted, *skipped))
    if comparison.indistinguishable:
        warnings.warn(
            f"формы {', '.join(comparison.indistinguishable)} неразличимы по профилю (ошибки в пределах "
            f"{LOSS_TIE_FRACTION:.0%}): выбор по loss ненадёжен — сверь с 2D-картой плотности и "
            "диаграммой устойчивости",
            PandaWarning,
            stacklevel=2,
        )
    return comparison
