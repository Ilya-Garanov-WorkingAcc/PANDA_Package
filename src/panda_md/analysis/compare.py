"""Перебор форм: фит одного профиля всеми формами поры и таблица по качеству."""

from collections.abc import Sequence

from pydantic import model_validator

from panda_md.core.errors import PandaError
from panda_md.core.fitting import fit
from panda_md.core.pore import Pore
from panda_md.core.profile import DensityProfile
from panda_md.core.result import FitResult
from panda_md.core.spec import FitSpec
from panda_md.core.types import PandaModel


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
            raise PandaError("ни одна форма не дала допустимого результата")
        return feasible[0]

    def summary(self) -> str:
        lines = [f"{self.pore_name}: сравнение форм"]
        for row in self.rows:
            if row.result is None:
                lines.append(f"  {row.shape_name:<24} пропущена: {row.skipped_reason}")
                continue
            result = row.result
            values = ", ".join(f"{name} = {value:.4g}" for name, value in result.values.items())
            mark = "" if result.feasible else "  (вне области существования)"
            lines.append(f"  {row.shape_name:<24} loss = {result.loss:.4g}  {values}{mark}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def compare_shapes(
    pore: Pore, profile: DensityProfile, spec: FitSpec | None = None, shape_names: Sequence[str] | None = None
) -> ShapeComparison:
    """Фит профиля каждой формой поры (по умолчанию — всеми) с одними настройками."""
    settings = spec if spec is not None else FitSpec()
    names = list(pore.own_registry()) if shape_names is None else list(shape_names)
    fitted: list[ShapeFit] = []
    skipped: list[ShapeFit] = []
    for name in names:
        model = pore.interface(name)
        try:
            fitted.append(ShapeFit(shape_name=name, result=fit(model, profile, settings)))
        except PandaError as error:  # нет профиля, нет допустимого старта, режим не объявлен, …
            skipped.append(ShapeFit(shape_name=name, skipped_reason=str(error)))
    fitted.sort(key=lambda row: (row.result is None or not row.result.feasible, row.result.loss if row.result else 0.0))
    return ShapeComparison(pore_name=pore.name, rows=(*fitted, *skipped))
