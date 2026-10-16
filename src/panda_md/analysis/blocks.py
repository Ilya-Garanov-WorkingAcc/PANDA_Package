"""Фит по блокам времени и статистическая неэффективность — оценка ошибки угла и выбор длины блока."""

import math
from collections.abc import Sequence

import numpy as np
from pydantic import Field, model_validator

from panda_md.core.errors import PandaError
from panda_md.core.fitting import fit
from panda_md.core.model import InterfaceModel
from panda_md.core.profile import ProfileSet
from panda_md.core.result import FitResult
from panda_md.core.spec import FitSpec
from panda_md.core.types import Array1D, FloatArray, PandaModel

MIN_SERIES_LENGTH = 4  # статистическую неэффективность по меньшему ряду не оценить


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
        """Разброс по блокам (0 при одном блоке)."""
        return float(np.std(self.values(name), ddof=1)) if len(self.results) > 1 else 0.0

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
    results = tuple(fit(model, profiles.block(index), settings) for index in range(profiles.block_count))
    return FitResultSet(results=results, block_start_ps=profiles.block_start_ps, block_stop_ps=profiles.block_stop_ps)


class StatisticalInefficiency(PandaModel):
    """s(t) = t · var(средних по блокам из t значений) / var(ряда). Выходит на плато s ≈ 2τ + 1 (τ — время
    корреляции в шагах ряда), когда блоки длиннее τ; блоки длиннее плато дают независимые оценки."""

    block_lengths: tuple[int, ...]
    inefficiency: Array1D

    def plateau(self) -> float:
        """Среднее s по последней трети длин блоков — оценка числа шагов на одно независимое значение."""
        tail = self.inefficiency[2 * len(self.inefficiency) // 3 :]
        return float(np.mean(tail))


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
    longest = series.size // 2 if max_block_length is None else min(max_block_length, series.size // 2)
    mean = float(np.mean(series))
    lengths: list[int] = []
    inefficiency: list[float] = []
    for length in range(1, longest + 1):
        block_count = series.size // length
        block_means = series[: block_count * length].reshape(block_count, length).mean(axis=1)
        block_variance = float(np.sum((block_means - mean) ** 2)) / (block_count - 1)
        lengths.append(length)
        inefficiency.append(length * block_variance / variance)
    return StatisticalInefficiency(block_lengths=tuple(lengths), inefficiency=np.array(inefficiency))
