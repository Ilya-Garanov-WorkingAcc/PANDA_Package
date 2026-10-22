"""Shape search: fit one profile with every shape of the pore and rank them by quality."""

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

LOSS_TIE_FRACTION = 0.02  # shape losses closer than 2 % — one profile cannot tell the shapes apart
NO_PROFILE_REASON = "no density profile (stability diagram only)"  # skip reason of shapes without a profile

# Errors that depend on the shape: the shape is skipped with a reason. Errors of the profile and the settings (the
# profile does not match the pore, an empty window, no ρ_bulk) are the same for all shapes — the search keeps them.
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
    """One comparison row: the fit result or the reason why the shape was not fitted."""

    shape_name: str
    result: FitResult | None = None
    skipped_reason: str | None = None

    @model_validator(mode="after")
    def _one_of(self) -> "ShapeFit":
        if (self.result is None) == (self.skipped_reason is None):
            raise ValueError("expected either a result or a skip reason")
        return self


class ShapeComparison(PandaModel):
    """Shapes in ascending order of the fit loss; skipped ones at the end with a reason.

    Losses are comparable between shapes only for the same data and settings; a lower loss is an argument, not a
    proof: different shapes have different numbers of free parameters.
    """

    pore_name: str
    rows: tuple[ShapeFit, ...]

    @property
    def fitted(self) -> tuple[FitResult, ...]:
        return tuple(row.result for row in self.rows if row.result is not None)

    @property
    def best(self) -> FitResult:
        """The feasible result with the lowest loss."""
        feasible = [result for result in self.fitted if result.feasible]
        if not feasible:
            reasons = "; ".join(f"{row.shape_name}: {row.skipped_reason}" for row in self.rows if row.skipped_reason)
            raise PandaError(f"no shape gave a feasible result ({reasons or 'all are outside the existence domain'})")
        return feasible[0]

    @property
    def indistinguishable(self) -> tuple[str, ...]:
        """Feasible shapes with a loss at most LOSS_TIE_FRACTION above the best: the profile cannot tell them apart."""
        feasible = [result for result in self.fitted if result.feasible]
        if not feasible:
            return ()
        threshold = feasible[0].loss * (1.0 + LOSS_TIE_FRACTION)
        tied = tuple(result.shape_name for result in feasible if result.loss <= threshold)
        return tied if len(tied) > 1 else ()

    def summary(self) -> str:
        lines = [f"{self.pore_name}: shape comparison"]
        tied = self.indistinguishable
        without_profile = [row.shape_name for row in self.rows if row.skipped_reason == NO_PROFILE_REASON]
        for row in self.rows:
            if row.skipped_reason == NO_PROFILE_REASON:
                continue
            if row.result is None:
                lines.append(f"  {row.shape_name:<24} skipped: {row.skipped_reason}")
                continue
            result = row.result
            values = ", ".join(f"{name} = {value:.4g}" for name, value in result.values.items())
            mark = "" if result.feasible else "  (outside the existence domain)"
            tie = "≈ " if row.shape_name in tied else "  "
            lines.append(f"{tie}{row.shape_name:<24} loss = {result.loss:.4g}  {values}{mark}")
        if without_profile:
            lines.append(f"  {NO_PROFILE_REASON}: {', '.join(without_profile)}")
        if tied:
            lines.append(f"≈ losses within {LOSS_TIE_FRACTION:.0%}: the profile cannot tell apart {', '.join(tied)}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def _ranking(row: ShapeFit) -> tuple[bool, float]:
    """Feasible results first, within them in ascending order of the loss."""
    assert row.result is not None  # only fitted shapes are sorted
    return not row.result.feasible, row.result.loss


def compare_shapes(
    pore: Pore, profile: DensityProfile, spec: FitSpec | None = None, shape_names: Sequence[str] | None = None
) -> ShapeComparison:
    """Fit the profile with each shape of the pore (all by default) with the same settings."""
    settings = spec if spec is not None else FitSpec()
    names = list(pore.own_registry()) if shape_names is None else list(shape_names)
    if len(set(names)) != len(names):
        raise PandaError(f"duplicate in the shape list: {', '.join(names)}")
    fitted: list[ShapeFit] = []
    skipped: list[ShapeFit] = []
    for name in names:
        model = pore.interface(name)
        if not model.shape.has_profile():
            skipped.append(ShapeFit(shape_name=name, skipped_reason=NO_PROFILE_REASON))
            continue
        try:
            fitted.append(ShapeFit(shape_name=name, result=fit(model, profile, settings)))
        except SHAPE_LEVEL_ERRORS as error:  # no profile, no feasible start, regime not declared, …
            skipped.append(ShapeFit(shape_name=name, skipped_reason=str(error)))
    fitted.sort(key=_ranking)
    comparison = ShapeComparison(pore_name=pore.name, rows=(*fitted, *skipped))
    if comparison.indistinguishable:
        warnings.warn(
            f"shapes {', '.join(comparison.indistinguishable)} are indistinguishable by the profile (losses within "
            f"{LOSS_TIE_FRACTION:.0%}): the choice by loss is unreliable; check against the 2D density map and "
            "the stability diagram",
            PandaWarning,
            stacklevel=2,
        )
    return comparison
