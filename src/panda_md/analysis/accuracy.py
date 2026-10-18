"""Accuracy of the method on synthetic data: error of the recovered θ for given noise, bin count, angle and centering.

The profile is built from the model (`synthetic_profile`) like an MD histogram — the density is averaged over a bin,
the noise is Poisson (σ ∝ √ρ, in fractions of ρ_bulk at density ρ_bulk) — and fitted with the same shape, whose model
is taken at the bin centres. The error depends only on the method (fit, discretization, noise), not on whether the
shape is adequate — it is a lower bound of the error on real data. The noise is given for 200 bins; for another bin
count it is rescaled as √(bins / 200): the same trajectory split more finely gives fewer particles per bin.
"""

import math
from collections.abc import Mapping, Sequence

import numpy as np
from pydantic import Field, model_validator

from panda_md.core.enums import Frame, NoiseModel, ParameterName, ProfileSampling
from panda_md.core.model import InterfaceModel
from panda_md.core.types import Array1D, Array2D, FloatArray, PandaModel
from panda_md.synthetic.profiles import synthetic_profile

REFERENCE_BIN_COUNT = 200  # the noise is given for this bin count


class AccuracyCase(PandaModel):
    """One point of the study: the true parameters and the synthetic data conditions."""

    truth: dict[str, float]  # user units
    noise_std: float = Field(ge=0.0)  # fraction of ρ_bulk per bin at REFERENCE_BIN_COUNT bins
    bin_count: int = Field(default=REFERENCE_BIN_COUNT, ge=10)
    offset: float = 0.0  # interface shift along the profile axis in fractions of the pore size (wall position error)
    frame: Frame = Frame.PORE
    optimize: tuple[str, ...] | None = None
    fixed: dict[str, float] = Field(default_factory=dict)
    noise_model: NoiseModel = NoiseModel.POISSON
    sampling: ProfileSampling = ProfileSampling.BIN_AVERAGE


class AngleErrors(PandaModel):
    """θ_fit − θ_true (degrees) per seed and the “θ at the fit bound” flag (then the error is limited by the bound)."""

    errors_deg: Array1D
    at_bound: tuple[bool, ...]


class AccuracyStudy(PandaModel):
    """θ errors (degrees, recovered − true) per noise seed for a series of values of one condition."""

    shape_name: str
    varied: str  # what varies: “noise, % ρ_bulk”, “bins”, “θ, °”, “centering”
    labels: tuple[str, ...]  # label of each value
    values: Array1D  # numeric value of the condition (ordinal number for categorical conditions)
    errors_deg: Array2D  # values × seeds
    at_bound_counts: tuple[int, ...]  # how many fits in the row hit the θ bound

    @model_validator(mode="after")
    def _check_shapes(self) -> "AccuracyStudy":
        rows = self.values.size
        if self.errors_deg.shape[0] != rows or len(self.labels) != rows or len(self.at_bound_counts) != rows:
            raise ValueError("expected one row of errors and one label per value of the condition")
        return self

    def mean_absolute(self) -> FloatArray:
        mean: FloatArray = np.mean(np.abs(self.errors_deg), axis=1)
        return mean

    def maximum_absolute(self) -> FloatArray:
        maximum: FloatArray = np.max(np.abs(self.errors_deg), axis=1)
        return maximum

    def bias(self) -> FloatArray:
        """Systematic shift: the mean of (recovered − true)."""
        bias: FloatArray = np.mean(self.errors_deg, axis=1)
        return bias

    def table(self) -> str:
        """Markdown table: condition, mean and maximum |Δθ|, bias."""
        seeds = self.errors_deg.shape[1]
        lines = [
            f"| {self.varied} | mean \\|Δθ\\|, ° | max \\|Δθ\\|, ° | bias, ° | at θ bound, of {seeds} |",
            "|---|---|---|---|---|",
        ]
        for label, mean, maximum, bias, at_bound in zip(
            self.labels, self.mean_absolute(), self.maximum_absolute(), self.bias(), self.at_bound_counts, strict=True
        ):
            lines.append(f"| {label} | {mean:.2f} | {maximum:.2f} | {bias:+.2f} | {at_bound} |")
        return "\n".join(lines)


def effective_noise_std(case: AccuracyCase) -> float:
    """Noise per bin at case.bin_count bins: the same statistics split more finely — σ · √(bins / 200)."""
    return case.noise_std * math.sqrt(case.bin_count / REFERENCE_BIN_COUNT)


def angle_errors(
    model: InterfaceModel, case: AccuracyCase, *, density_bulk_per_nm3: float, seeds: Sequence[int]
) -> AngleErrors:
    """θ_fit − θ_true (degrees) and the “at the θ bound” flag for each noise seed."""
    noise = effective_noise_std(case)
    errors: list[float] = []
    at_bound: list[bool] = []
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
            noise_model=case.noise_model,
            sampling=case.sampling,
        )
        result = model.fit(profile, density_bulk_per_nm3=density_bulk_per_nm3, optimize=case.optimize, fixed=case.fixed)
        angle = ParameterName.CONTACT_ANGLE.value
        errors.append(result.values[angle] - case.truth[angle])
        at_bound.append(angle in result.at_bounds)
    return AngleErrors(errors_deg=np.array(errors), at_bound=tuple(at_bound))


def accuracy_study(
    model: InterfaceModel,
    varied: str,
    cases: Mapping[str, tuple[float, AccuracyCase]],
    *,
    density_bulk_per_nm3: float,
    seeds: Sequence[int],
) -> AccuracyStudy:
    """Run the study: for each label — (value of the condition, synthetic data conditions)."""
    if not list(seeds):
        raise ValueError("no noise seeds; at least one is needed")
    errors = [
        angle_errors(model, case, density_bulk_per_nm3=density_bulk_per_nm3, seeds=seeds) for _, case in cases.values()
    ]
    return AccuracyStudy(
        shape_name=model.shape.name,
        varied=varied,
        labels=tuple(cases),
        values=np.array([value for value, _ in cases.values()]),
        errors_deg=np.array([row.errors_deg for row in errors]),
        at_bound_counts=tuple(sum(row.at_bound) for row in errors),
    )
