"""Точность метода на синтетике: ошибка восстановленного θ при заданных шуме, числе бинов, угле и центрировании.

Профиль строится по модели (`synthetic_profile`) с гауссовым шумом в долях ρ_bulk на бин, затем фитируется той же
формой. Ошибка зависит только от метода (фит, дискретизация, шум), а не от адекватности формы — это нижняя граница
ошибки на реальных данных. Шум задаётся для 200 бинов; при другом числе бинов он пересчитывается как √(бинов / 200):
та же траектория, разбитая мельче, даёт меньше частиц на бин.
"""

import math
from collections.abc import Mapping, Sequence

import numpy as np
from pydantic import Field, model_validator

from panda_md.core.enums import Frame, ParameterName
from panda_md.core.model import InterfaceModel
from panda_md.core.types import Array1D, Array2D, FloatArray, PandaModel
from panda_md.synthetic.profiles import synthetic_profile

REFERENCE_BIN_COUNT = 200  # шум задаётся для этого числа бинов


class AccuracyCase(PandaModel):
    """Одна точка исследования: истинные параметры и условия синтетики."""

    truth: dict[str, float]  # единицы пользователя
    noise_std: float = Field(ge=0.0)  # доля ρ_bulk на бин при REFERENCE_BIN_COUNT бинах
    bin_count: int = Field(default=REFERENCE_BIN_COUNT, ge=10)
    offset: float = 0.0  # сдвиг интерфейса вдоль оси профиля в долях размера поры (ошибка положения стенок)
    frame: Frame = Frame.PORE
    optimize: tuple[str, ...] | None = None
    fixed: dict[str, float] = Field(default_factory=dict)


class AccuracyStudy(PandaModel):
    """Ошибки θ (градусы, восстановленное − истинное) по зёрнам шума для ряда значений одного условия."""

    shape_name: str
    varied: str  # что меняется: «шум, % ρ_bulk», «бинов», «θ, °», «центрирование»
    labels: tuple[str, ...]  # подпись каждого значения
    values: Array1D  # численное значение условия (для оси графика)
    errors_deg: Array2D  # значения × зёрна

    @model_validator(mode="after")
    def _check_shapes(self) -> "AccuracyStudy":
        if self.errors_deg.shape[0] != self.values.size or len(self.labels) != self.values.size:
            raise ValueError("по строке ошибок и подписи на каждое значение условия")
        return self

    def mean_absolute(self) -> FloatArray:
        mean: FloatArray = np.mean(np.abs(self.errors_deg), axis=1)
        return mean

    def maximum_absolute(self) -> FloatArray:
        maximum: FloatArray = np.max(np.abs(self.errors_deg), axis=1)
        return maximum

    def bias(self) -> FloatArray:
        """Систематический сдвиг: среднее (восстановленное − истинное)."""
        bias: FloatArray = np.mean(self.errors_deg, axis=1)
        return bias

    def table(self) -> str:
        """Таблица markdown: условие, средняя и максимальная |Δθ|, смещение."""
        lines = [f"| {self.varied} | среднее \\|Δθ\\|, ° | максимум \\|Δθ\\|, ° | смещение, ° |", "|---|---|---|---|"]
        for label, mean, maximum, bias in zip(
            self.labels, self.mean_absolute(), self.maximum_absolute(), self.bias(), strict=True
        ):
            lines.append(f"| {label} | {mean:.2f} | {maximum:.2f} | {bias:+.2f} |")
        return "\n".join(lines)


def effective_noise_std(case: AccuracyCase) -> float:
    """Шум на бин при case.bin_count бинах: та же статистика, разбитая мельче, — σ · √(бинов / 200)."""
    return case.noise_std * math.sqrt(case.bin_count / REFERENCE_BIN_COUNT)


def angle_errors(
    model: InterfaceModel, case: AccuracyCase, *, density_bulk_per_nm3: float, seeds: Sequence[int]
) -> FloatArray:
    """θ_фит − θ_истинный (градусы) для каждого зерна шума."""
    noise = effective_noise_std(case)
    errors = []
    for seed in seeds:
        profile = synthetic_profile(
            model,
            case.truth,
            density_bulk_per_nm3=density_bulk_per_nm3,
            bin_count=case.bin_count,
            noise_std=noise,
            frame=case.frame,
            offset=case.offset,
            seed=seed,
        )
        result = model.fit(profile, density_bulk_per_nm3=density_bulk_per_nm3, optimize=case.optimize, fixed=case.fixed)
        errors.append(result.values[ParameterName.CONTACT_ANGLE.value] - case.truth[ParameterName.CONTACT_ANGLE.value])
    angles: FloatArray = np.array(errors)
    return angles


def accuracy_study(
    model: InterfaceModel,
    varied: str,
    cases: Mapping[str, tuple[float, AccuracyCase]],
    *,
    density_bulk_per_nm3: float,
    seeds: Sequence[int],
) -> AccuracyStudy:
    """Исследование: для каждой подписи — (значение условия, условия синтетики)."""
    errors = [
        angle_errors(model, case, density_bulk_per_nm3=density_bulk_per_nm3, seeds=seeds) for _, case in cases.values()
    ]
    return AccuracyStudy(
        shape_name=model.shape.name,
        varied=varied,
        labels=tuple(cases),
        values=np.array([value for value, _ in cases.values()]),
        errors_deg=np.array(errors),
    )
