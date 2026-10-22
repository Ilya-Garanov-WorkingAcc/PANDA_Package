"""Fit over time blocks and statistical inefficiency — angle error estimate and choice of the block length."""

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

SAME_VALUE_TOLERANCE = 1e-12  # relative difference of values treated as float noise
MIN_BLOCKS_FOR_SPREAD = 2  # the spread over one block is not defined
MIN_SERIES_LENGTH = 4  # the statistical inefficiency of a shorter series cannot be estimated
MIN_BLOCKS = 20  # with fewer blocks the variance of the block means is too noisy for a plateau
MIN_PLATEAU_LENGTHS = 3  # plateau — mean over the last third of the lengths: fewer than three reliable lengths is none


class FitResultSet(PandaModel):
    """Fit results of each block: mean, spread and standard error of the mean over blocks for any parameter."""

    results: tuple[FitResult, ...] = Field(min_length=1)
    block_start_ps: Array1D
    block_stop_ps: Array1D

    @model_validator(mode="after")
    def _check_lengths(self) -> "FitResultSet":
        count = len(self.results)
        if self.block_start_ps.shape != (count,) or self.block_stop_ps.shape != (count,):
            raise ValueError("invalid block times; expected one per result")
        return self

    def values(self, name: str) -> FloatArray:
        """Value of the parameter (user units) in each block."""
        if name not in self.results[0].values:
            raise PandaError(f"unknown parameter '{name}'; available: {', '.join(self.results[0].values)}")
        return np.array([result.values[name] for result in self.results])

    def mean(self, name: str) -> float:
        return float(np.mean(self.values(name)))

    def std(self, name: str) -> float:
        """Spread over blocks; not defined (nan) for a single block."""
        values = self.values(name)
        if len(values) < MIN_BLOCKS_FOR_SPREAD:
            return math.nan
        if np.ptp(values) <= SAME_VALUE_TOLERANCE * max(1.0, float(np.max(np.abs(values)))):
            return 0.0  # the same value in all blocks: without float noise like 2e-17
        return float(np.std(values, ddof=1))

    def sem(self, name: str) -> float:
        """Standard error of the mean: std / √(number of blocks) — valid if the blocks exceed the correlation time."""
        return self.std(name) / math.sqrt(len(self.results))

    def summary(self) -> str:
        names = list(self.results[0].values)
        header = "block t, ps           " + "".join(f"{name:>11}" for name in names) + "       loss  flags"
        lines = [f"{self.results[0].pore_name}/{self.results[0].shape_name}: {len(self.results)} block(s)", header]
        for index, result in enumerate(self.results):
            flags = ", ".join(_flags(result))
            span = f"{self.block_start_ps[index]:g}–{self.block_stop_ps[index]:g}"
            row = "".join(f"{result.values[name]:>11.5g}" for name in names)
            lines.append(f"{index:>4}  {span:<16}{row}  {result.loss:>9.4g}  {flags}")
        if len(self.results) == 1:
            lines.append("a single block: the spread and the standard error cannot be estimated")
        else:
            lines.append(
                "mean ± standard error: "
                + ", ".join(f"{name} = {self.mean(name):.5g} ± {self.sem(name):.2g}" for name in names)
            )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def _flags(result: FitResult) -> list[str]:
    flags = [f"at bound: {name}" for name in result.at_bounds]
    if result.at_regime_boundary:
        flags.append("θ at 90°")
    if not result.feasible:
        flags.append("outside the existence domain")
    return flags


def fit_blocks(model: InterfaceModel, profiles: ProfileSet, spec: FitSpec | None = None) -> FitResultSet:
    """Fit the profile of each block with the same settings; the spread over blocks estimates the parameter error."""
    settings = spec if spec is not None else FitSpec()
    results: list[FitResult] = []
    for index in range(profiles.block_count):
        try:
            results.append(fit(model, profiles.block(index), settings))
        except PandaError as error:
            span = f"{profiles.block_start_ps[index]:g}–{profiles.block_stop_ps[index]:g} ps"
            raise type(error)(f"block {index} ({span}): {error}") from error
    return FitResultSet(
        results=tuple(results), block_start_ps=profiles.block_start_ps, block_stop_ps=profiles.block_stop_ps
    )


class StatisticalInefficiency(PandaModel):
    """s(t) = t · var(means of blocks of t values) / var(series). Reaches a plateau s ≈ 2τ + 1 (τ is the correlation
    time in series steps) when the blocks are longer than τ; blocks longer than the plateau give independent estimates.
    """

    block_lengths: tuple[int, ...]
    inefficiency: Array1D
    series_length: int = Field(ge=MIN_SERIES_LENGTH)

    def plateau(self) -> float:
        """Mean s over the last third of the lengths with at least MIN_BLOCKS blocks (otherwise over the last third
        of all lengths) — an estimate of the number of series steps per independent value."""
        lengths = np.array(self.block_lengths)
        reliable = self.inefficiency[self.series_length // lengths >= MIN_BLOCKS]
        if reliable.size < MIN_PLATEAU_LENGTHS:
            warnings.warn(
                f"the plateau cannot be estimated: only {reliable.size} block length(s) give at least {MIN_BLOCKS} "
                f"blocks; a series longer than {MIN_PLATEAU_LENGTHS * MIN_BLOCKS} values is needed; s is likely "
                f"underestimated",
                PandaWarning,
                stacklevel=2,
            )
        values = reliable if reliable.size else self.inefficiency
        return float(np.mean(values[2 * len(values) // 3 :]))


def statistical_inefficiency(
    values: Sequence[float] | FloatArray, max_block_length: int | None = None
) -> StatisticalInefficiency:
    """Statistical inefficiency of a series (θ over short blocks, φ over frames, …), as `stat_ineff.py` of old PANDA."""
    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 1 or series.size < MIN_SERIES_LENGTH:
        raise PandaError(f"invalid series; expected a one-dimensional series of at least {MIN_SERIES_LENGTH} values")
    variance = float(np.var(series, ddof=1))
    if variance == 0.0:
        raise PandaError("the series is constant: the inefficiency is not defined")
    if max_block_length is None and series.size // MIN_BLOCKS < MIN_PLATEAU_LENGTHS:
        raise PandaError(
            f"series of {series.size} values is too short: a plateau needs at least "
            f"{MIN_PLATEAU_LENGTHS * MIN_BLOCKS} (at least {MIN_PLATEAU_LENGTHS} block lengths of {MIN_BLOCKS} "
            f"blocks); use shorter blocks or set max_block_length"
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
