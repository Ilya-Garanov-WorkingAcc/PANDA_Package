"""Фит по блокам времени и статистическая неэффективность — оценка ошибки угла и выбор длины блока."""

import math
import warnings
from collections.abc import Sequence

import numpy as np
from pydantic import Field, model_validator

from panda_md.core.errors import PandaError, PandaWarning
from panda_md.core.fitting import fit
from panda_md.core.model import InterfaceModel
from panda_md.core.profile import ProfileSet
from panda_md.core.result import FitResult
from panda_md.core.spec import FitSpec
from panda_md.core.types import Array1D, FloatArray, PandaModel

SAME_VALUE_TOLERANCE = 1e-12  # относительная разница значений, которую считаем шумом float
MIN_BLOCKS_FOR_SPREAD = 2  # разброс по одному блоку не определён
MIN_SERIES_LENGTH = 4  # статистическую неэффективность по меньшему ряду не оценить
MIN_BLOCKS = 20  # меньше блоков — дисперсия средних по блокам слишком шумная для плато
MIN_PLATEAU_LENGTHS = 3  # плато — среднее по последней трети длин: меньше трёх надёжных длин — не плато


class FitResultSet(PandaModel):
    """Результаты фита каждого блока: среднее, разброс и ошибка среднего по блокам для любого параметра."""

    results: tuple[FitResult, ...] = Field(min_length=1)
    block_start_ps: Array1D
    block_stop_ps: Array1D

    @model_validator(mode="after")
    def _check_lengths(self) -> "FitResultSet":
        count = len(self.results)
        if self.block_start_ps.shape != (count,) or self.block_stop_ps.shape != (count,):
            raise ValueError("времена блоков: по одному на результат")
        return self

    def values(self, name: str) -> FloatArray:
        """Значение параметра (единицы пользователя) в каждом блоке."""
        if name not in self.results[0].values:
            raise PandaError(f"параметра {name} нет; есть: {list(self.results[0].values)}")
        return np.array([result.values[name] for result in self.results])

    def mean(self, name: str) -> float:
        return float(np.mean(self.values(name)))

    def std(self, name: str) -> float:
        """Разброс по блокам; при одном блоке не определён (nan)."""
        values = self.values(name)
        if len(values) < MIN_BLOCKS_FOR_SPREAD:
            return math.nan
        if np.ptp(values) <= SAME_VALUE_TOLERANCE * max(1.0, float(np.max(np.abs(values)))):
            return 0.0  # одинаковое во всех блоках значение: без шума float вида 2e-17
        return float(np.std(values, ddof=1))

    def sem(self, name: str) -> float:
        """Ошибка среднего: std / √(число блоков) — верна, если блоки длиннее времени корреляции."""
        return self.std(name) / math.sqrt(len(self.results))

    def summary(self) -> str:
        names = list(self.results[0].values)
        header = "блок  t, пс           " + "".join(f"{name:>11}" for name in names) + "       loss  флаги"
        lines = [f"{self.results[0].pore_name}/{self.results[0].shape_name}: {len(self.results)} блоков", header]
        for index, result in enumerate(self.results):
            flags = ", ".join(_flags(result))
            span = f"{self.block_start_ps[index]:g}–{self.block_stop_ps[index]:g}"
            row = "".join(f"{result.values[name]:>11.5g}" for name in names)
            lines.append(f"{index:>4}  {span:<16}{row}  {result.loss:>9.4g}  {flags}")
        if len(self.results) == 1:
            lines.append("один блок — разброс и ошибку среднего не оценить")
        else:
            lines.append(
                "среднее ± ошибка среднего: "
                + ", ".join(f"{name} = {self.mean(name):.5g} ± {self.sem(name):.2g}" for name in names)
            )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def _flags(result: FitResult) -> list[str]:
    flags = [f"у границы {name}" for name in result.at_bounds]
    if result.at_regime_boundary:
        flags.append("θ у 90°")
    if not result.feasible:
        flags.append("вне области существования")
    return flags


def fit_blocks(model: InterfaceModel, profiles: ProfileSet, spec: FitSpec | None = None) -> FitResultSet:
    """Фит профиля каждого блока с одними настройками; разброс по блокам — оценка ошибки параметров."""
    settings = spec if spec is not None else FitSpec()
    results: list[FitResult] = []
    for index in range(profiles.block_count):
        try:
            results.append(fit(model, profiles.block(index), settings))
        except PandaError as error:
            span = f"{profiles.block_start_ps[index]:g}–{profiles.block_stop_ps[index]:g} пс"
            raise type(error)(f"блок {index} ({span}): {error}") from error
    return FitResultSet(
        results=tuple(results), block_start_ps=profiles.block_start_ps, block_stop_ps=profiles.block_stop_ps
    )


class StatisticalInefficiency(PandaModel):
    """s(t) = t · var(средних по блокам из t значений) / var(ряда). Выходит на плато s ≈ 2τ + 1 (τ — время
    корреляции в шагах ряда), когда блоки длиннее τ; блоки длиннее плато дают независимые оценки."""

    block_lengths: tuple[int, ...]
    inefficiency: Array1D
    series_length: int = Field(ge=MIN_SERIES_LENGTH)

    def plateau(self) -> float:
        """Среднее s по последней трети длин, при которых блоков не меньше MIN_BLOCKS (иначе — по последней трети
        всех длин) — оценка числа шагов ряда на одно независимое значение."""
        lengths = np.array(self.block_lengths)
        reliable = self.inefficiency[self.series_length // lengths >= MIN_BLOCKS]
        if reliable.size < MIN_PLATEAU_LENGTHS:
            warnings.warn(
                f"плато не оценить: длин блоков, при которых блоков не меньше {MIN_BLOCKS}, всего {reliable.size} — "
                f"нужен ряд длиннее {MIN_PLATEAU_LENGTHS * MIN_BLOCKS} значений; s, скорее всего, занижена",
                PandaWarning,
                stacklevel=2,
            )
        values = reliable if reliable.size else self.inefficiency
        return float(np.mean(values[2 * len(values) // 3 :]))


def statistical_inefficiency(
    values: Sequence[float] | FloatArray, max_block_length: int | None = None
) -> StatisticalInefficiency:
    """Статистическая неэффективность ряда (θ по коротким блокам, φ по кадрам, …) — как `stat_ineff.py` старой PANDA."""
    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 1 or series.size < MIN_SERIES_LENGTH:
        raise PandaError(f"нужен одномерный ряд хотя бы из {MIN_SERIES_LENGTH} значений")
    variance = float(np.var(series, ddof=1))
    if variance == 0.0:
        raise PandaError("ряд постоянен: неэффективность не определена")
    if max_block_length is None and series.size // MIN_BLOCKS < MIN_PLATEAU_LENGTHS:
        raise PandaError(
            f"ряд из {series.size} значений слишком короток: для плато нужно не меньше "
            f"{MIN_PLATEAU_LENGTHS * MIN_BLOCKS} (хотя бы {MIN_PLATEAU_LENGTHS} длины блока по {MIN_BLOCKS} блоков); "
            f"возьми блоки короче или задай max_block_length"
        )
    longest = series.size // MIN_BLOCKS if max_block_length is None else max_block_length
    longest = min(longest, series.size // 2)
    lengths: list[int] = []
    inefficiency: list[float] = []
    for length in range(1, longest + 1):
        block_count = series.size // length
        block_means = series[: block_count * length].reshape(block_count, length).mean(axis=1)
        block_variance = float(np.var(block_means, ddof=1))
        lengths.append(length)
        inefficiency.append(length * block_variance / variance)
    return StatisticalInefficiency(
        block_lengths=tuple(lengths), inefficiency=np.array(inefficiency), series_length=series.size
    )
