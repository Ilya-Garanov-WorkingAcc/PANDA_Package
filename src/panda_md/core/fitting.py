"""Generic fit: one algorithm for any shape, any pore and any set of free parameters."""

import itertools
import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import Bounds, minimize

from panda_md.core.enums import Frame, LossFunction, Normalization, OptimizerMethod, ParameterName, Regime
from panda_md.core.errors import (
    EmptyWindowError,
    InfeasibleParametersError,
    InvalidProfileError,
    NotDerivedError,
    ProfileMismatchError,
    UnidentifiableParameterError,
    UnsupportedFrameError,
    UnsupportedRegimeError,
)
from panda_md.core.parameters import FreeSpec, ParameterValues
from panda_md.core.profile import DensityProfile, ReducedProfile
from panda_md.core.result import FitResult
from panda_md.core.shape import REGIME_BOUNDARY_DEG, REGIME_GAP_DEG, PoreGeometry
from panda_md.core.spec import FitSpec
from panda_md.core.types import BoolArray, FloatArray
from panda_md.core.version import library_version

if TYPE_CHECKING:
    from panda_md.core.model import InterfaceModel


MIN_WINDOW_POINTS = 10  # fewer points in the fit window — EmptyWindowError
# steps of the 'parameter has an effect' check, as fractions of its bounds width: large ones are needed for
# parameters that move only the edge of the profile (φ of the layer) — a small edge shift stays within a bin
IDENTIFIABILITY_STEP_FRACTIONS = (1e-3, 1e-2, 5e-2)
IDENTIFIABILITY_TOLERANCE = 1e-9  # relative profile change below which the parameter 'has no effect'
BOUND_PROXIMITY_FRACTION = 1e-3  # closer than this fraction of the bounds width — the parameter is 'at bound'
MEAN_PHI_FLOOR = 1e-3  # mean normalization: the barrier estimate uses φ not below this (it degenerates as φ → 0)
BARRIER_LEVEL_FACTOR = 10.0  # barrier level: this many times the upper estimate of the feasible loss (+1)
BARRIER_SLOPE = 100.0  # barrier slope in the amount by which φ leaves its existence limits
COVERAGE_TOLERANCE = 0.05  # the data must reach the walls to within 5 % of the pore size
VOLUME_TOLERANCE = 0.01  # pore volume of the profile vs. the fitted pore: a difference above 1 % means a different pore
OUTSIDE_MASS_TOLERANCE = 0.05  # allowed fraction of ∫ρ outside the pore (pore frame)

CONTACT_ANGLE_NAME = ParameterName.CONTACT_ANGLE.value
WALL_LAYER_NAME = ParameterName.WALL_LAYER_THICKNESS.value
VOLUME_FRACTION_NAME = ParameterName.VOLUME_FRACTION.value
PROFILE_SHIFT_NAME = ParameterName.PROFILE_SHIFT.value


class _Objective:
    """Fit objective: the model in the frame of reference of the data, comparison with the data, barrier."""

    model: "InterfaceModel"
    geometry: PoreGeometry
    free_spec: FreeSpec
    coordinate: FloatArray
    data: FloatArray
    frame: Frame
    normalization: Normalization
    loss_function: LossFunction
    barrier_level: float
    evaluation_count: int

    def __init__(
        self,
        model: "InterfaceModel",
        free_spec: FreeSpec,
        reduced: ReducedProfile,
        window_mask: BoolArray,
        loss_function: LossFunction,
    ) -> None:
        self.model = model
        self.geometry = model.pore.geometry()
        self.free_spec = free_spec
        self.coordinate = reduced.coordinate[window_mask]
        self.data = reduced.density[window_mask]
        self.frame = reduced.frame
        self.normalization = reduced.normalization
        self.loss_function = loss_function
        self.barrier_level = BARRIER_LEVEL_FACTOR * (self._feasible_loss_bound() + 1.0)
        self.evaluation_count = 0

    def _feasible_loss_bound(self) -> float:
        """Upper estimate of the loss at any feasible point: |model − data| ≤ |model| + |data|.
        The model ρ/ρ_bulk ≤ 1; with mean normalization it is divided by φ — the estimate uses the smallest
        reachable φ (but not below MEAN_PHI_FLOOR: mean normalization degenerates as φ → 0).
        """
        largest_model = 1.0
        if self.normalization is Normalization.MEAN:
            largest_model = 1.0 / max(self._smallest_volume_fraction(), MEAN_PHI_FLOOR)
        worst = np.abs(self.data) + largest_model
        if self.loss_function is LossFunction.L1:
            return float(np.mean(worst))
        return float(np.mean(worst**2))

    def _smallest_volume_fraction(self) -> float:
        spec = self.free_spec
        if VOLUME_FRACTION_NAME in spec.fixed_values:
            return spec.fixed_values[VOLUME_FRACTION_NAME]
        if VOLUME_FRACTION_NAME in spec.free_names:
            return float(spec.lower_bounds[spec.free_names.index(VOLUME_FRACTION_NAME)])
        return MEAN_PHI_FLOOR

    def loss(self, curve: FloatArray) -> float:
        residual = curve - self.data
        if self.loss_function is LossFunction.L1:
            return float(np.mean(np.abs(residual)))
        return float(np.mean(residual**2))

    def prediction(self, values: ParameterValues) -> FloatArray:
        """Model dimensionless density at the data coordinates."""
        shape = self.model.shape
        position = self.coordinate - values.get(PROFILE_SHIFT_NAME, 0.0)
        if self.frame is not shape.frame:
            if self.frame is Frame.LIQUID and shape.frame is Frame.PORE:
                position = position + shape.center_of_mass(self.geometry, values)
            else:
                raise UnsupportedFrameError(
                    f"data are in the {self.frame.value} frame, the formulas of shape '{shape.name}' are in the "
                    f"{shape.frame.value} frame: this conversion is not supported"
                )
        curve = shape.profile(position, self.geometry, values)
        if self.normalization is Normalization.MEAN:
            curve = curve / values[VOLUME_FRACTION_NAME]
        return curve

    def phi_violation(self, values: ParameterValues) -> float:
        shape = self.model.shape
        if not shape.has_phi_limits() or VOLUME_FRACTION_NAME not in values:
            return 0.0
        with np.errstate(all="ignore"):
            lower, upper = shape.phi_limits(self.geometry, values)
        volume_fraction = values[VOLUME_FRACTION_NAME]
        violation = max(lower - volume_fraction, volume_fraction - upper, 0.0)
        return violation if math.isfinite(violation) else 0.0

    def is_feasible(self, values: ParameterValues) -> bool:
        with np.errstate(all="ignore"):
            return bool(self.model.shape.feasible(self.geometry, values))

    def __call__(self, vector: FloatArray) -> float:
        self.evaluation_count += 1
        values = self.free_spec.unpack(vector)
        if not self.is_feasible(values):
            return self.barrier_level + BARRIER_SLOPE * (1.0 + self.phi_violation(values))
        return self.loss(self.prediction(values))


class _LocalOptimum:
    """Best point of a local minimization (a common form for any scipy method)."""

    position: FloatArray
    loss: float
    success: bool
    message: str

    def __init__(self, position: FloatArray, loss: float, success: bool, message: str) -> None:
        self.position = position
        self.loss = loss
        self.success = success
        self.message = message


class _RegimeOutcome:
    regime: Regime | None
    free_spec: FreeSpec
    optimum: _LocalOptimum
    start_count: int
    evaluation_count: int  # objective evaluations in this regime

    def __init__(
        self,
        regime: Regime | None,
        free_spec: FreeSpec,
        optimum: _LocalOptimum,
        start_count: int,
        evaluation_count: int,
    ) -> None:
        self.regime = regime
        self.free_spec = free_spec
        self.optimum = optimum
        self.start_count = start_count
        self.evaluation_count = evaluation_count


def _resolve_normalization(spec: FitSpec) -> Normalization:
    if spec.normalization is not Normalization.AUTO:
        return spec.normalization
    return Normalization.BULK if spec.density_bulk_per_nm3 is not None else Normalization.MEAN


def histogram_integral(centers: FloatArray, density: FloatArray) -> float:
    """∫ρ of a histogram: Σ ρ_i · width of bin i (edges are midpoints between centres, outer ones symmetric).
    The trapezoid rule over bin centres loses half a bin at each edge: if the liquid touches a wall, φ is
    underestimated.
    """
    midpoints = 0.5 * (centers[:-1] + centers[1:])
    first = centers[0] - (midpoints[0] - centers[0])
    last = centers[-1] + (centers[-1] - midpoints[-1])
    edges = np.concatenate(([first], midpoints, [last]))
    return float(np.sum(density * np.diff(edges)))


def _window_mask(reduced: ReducedProfile, geometry: PoreGeometry, window: tuple[float, float] | None) -> BoolArray:
    if window is not None:
        lower, upper = window
    elif reduced.frame is Frame.PORE:
        lower, upper = geometry.coordinate_bounds()
    else:
        lower, upper = -math.inf, math.inf
    mask = (reduced.coordinate >= lower) & (reduced.coordinate <= upper)
    point_count = int(mask.sum())
    if point_count < MIN_WINDOW_POINTS:
        raise EmptyWindowError(
            f"fit window [{lower:.3g}, {upper:.3g}] has {point_count} point(s), at least {MIN_WINDOW_POINTS} needed; "
            f"data coordinate as a fraction of the pore size: [{float(np.min(reduced.coordinate)):.3g}, "
            f"{float(np.max(reduced.coordinate)):.3g}]; check the pore size and the frame of reference of the profile"
        )
    return mask


def _check_pore_volume(profile: DensityProfile, pore_volume_nm3: float) -> None:
    """The pore volume to which N and φ of the profile refer must match the volume of the fitted pore."""
    if profile.pore_volume_nm3 is None:
        return
    mismatch = abs(profile.pore_volume_nm3 / pore_volume_nm3 - 1.0)
    if mismatch > VOLUME_TOLERANCE:
        raise ProfileMismatchError(
            f"profile was computed for a pore volume of {profile.pore_volume_nm3:.4g} nm³, the fitted pore has "
            f"{pore_volume_nm3:.4g} nm³ ({mismatch:.1%}): φ and the normalization would be wrong; "
            "check the pore dimensions"
        )


def _check_profile_matches_pore(reduced: ReducedProfile, geometry: PoreGeometry) -> None:
    """The profile must match the pore in size: otherwise the fit silently returns a wrong angle."""
    coordinate, density = reduced.coordinate, reduced.density
    pore_lower, pore_upper = geometry.coordinate_bounds()
    data_range = f"[{float(np.min(coordinate)):.3g}, {float(np.max(coordinate)):.3g}]"
    if reduced.frame is Frame.PORE:
        if np.min(coordinate) > pore_lower + COVERAGE_TOLERANCE or np.max(coordinate) < pore_upper - COVERAGE_TOLERANCE:
            raise ProfileMismatchError(
                f"data do not cover the pore: coordinate as a fraction of the pore size {data_range}, walls "
                f"[{pore_lower:.3g}, {pore_upper:.3g}]; check the pore size and the profile units"
            )
        outside = (coordinate < pore_lower) | (coordinate > pore_upper)
        outside_fraction = float(np.sum(density[outside]) / np.sum(density))
        if outside_fraction > OUTSIDE_MASS_TOLERANCE:
            raise ProfileMismatchError(
                f"{outside_fraction:.0%} of the density is outside the pore (coordinate {data_range}); "
                "check the pore size, the profile units and the centering"
            )
        return
    occupied = coordinate[density > 0.0]
    occupied_width = float(np.max(occupied) - np.min(occupied))
    if occupied_width > (pore_upper - pore_lower) * (1.0 + COVERAGE_TOLERANCE):
        raise ProfileMismatchError(
            f"profile is wider than the pore: the liquid spans {occupied_width:.3g} of the pore size, "
            f"the pore width is {pore_upper - pore_lower:.3g}; check the pore size and the profile units"
        )


def _regime_intervals(
    model: "InterfaceModel", free_spec: FreeSpec, requested: Regime | None
) -> list[tuple[Regime | None, FreeSpec]]:
    shape = model.shape
    if CONTACT_ANGLE_NAME not in model.parameters.names:
        if requested is not None:
            raise UnsupportedRegimeError(
                f"shape '{shape.name}' has no contact angle; the wetting regime {requested.value} cannot be set"
            )
        return [(None, free_spec)]
    if CONTACT_ANGLE_NAME not in free_spec.free_names:
        regime = shape.require_supported_contact_angle(math.degrees(free_spec.fixed_values[CONTACT_ANGLE_NAME]))
        if requested is not None and requested is not regime:
            raise UnsupportedRegimeError(
                f"fixed θ of shape '{shape.name}' belongs to regime {regime.value}, but {requested.value} was requested"
            )
        return [(regime, free_spec)]

    index = free_spec.free_names.index(CONTACT_ANGLE_NAME)
    lower_deg = math.degrees(free_spec.lower_bounds[index])
    upper_deg = math.degrees(free_spec.upper_bounds[index])
    intervals: list[tuple[Regime | None, FreeSpec]] = []
    for regime, (regime_lower, regime_upper) in shape.regimes.items():
        if requested is not None and regime is not requested:
            continue
        if regime_lower == REGIME_BOUNDARY_DEG:
            regime_lower += REGIME_GAP_DEG
        if regime_upper == REGIME_BOUNDARY_DEG:
            regime_upper -= REGIME_GAP_DEG
        lower = max(lower_deg, regime_lower)
        upper = min(upper_deg, regime_upper)
        if lower < upper:
            intervals.append(
                (regime, free_spec.with_free_bounds(CONTACT_ANGLE_NAME, math.radians(lower), math.radians(upper)))
            )
    if not intervals:
        declared = [regime.value for regime in shape.regimes]
        raise UnsupportedRegimeError(
            f"θ bounds [{lower_deg:.1f}°, {upper_deg:.1f}°] of shape '{shape.name}' do not intersect the declared "
            f"regimes {declared}"
        )
    return intervals


def _start_points(free_spec: FreeSpec, starts_per_parameter: int) -> list[FloatArray]:
    """Cell centres of a uniform grid over the free parameters + the user's starting point."""
    axes = [
        lower + (np.arange(starts_per_parameter) + 0.5) / starts_per_parameter * (upper - lower)
        for lower, upper in zip(free_spec.lower_bounds, free_spec.upper_bounds, strict=True)
    ]
    points = [np.array(point, dtype=np.float64) for point in itertools.product(*axes)]
    if free_spec.initial_values:
        midpoint = 0.5 * (free_spec.lower_bounds + free_spec.upper_bounds)
        user_start = np.array(
            [
                free_spec.initial_values.get(name, float(midpoint[index]))
                for index, name in enumerate(free_spec.free_names)
            ],
            dtype=np.float64,
        )
        points.insert(0, np.clip(user_start, free_spec.lower_bounds, free_spec.upper_bounds))
    return points


def _check_identifiable(objective: _Objective, free_spec: FreeSpec, start: FloatArray) -> None:
    reference = objective.prediction(free_spec.unpack(start))
    scale = max(1.0, float(np.max(np.abs(reference))))
    for index, name in enumerate(free_spec.free_names):
        width = free_spec.upper_bounds[index] - free_spec.lower_bounds[index]
        changes_profile = False
        for step, direction in itertools.product(
            (fraction * width for fraction in IDENTIFIABILITY_STEP_FRACTIONS), (1.0, -1.0)
        ):
            trial = start.copy()
            trial[index] = np.clip(
                start[index] + direction * step, free_spec.lower_bounds[index], free_spec.upper_bounds[index]
            )
            values = free_spec.unpack(trial)
            if not objective.is_feasible(values):
                continue
            change = float(np.max(np.abs(objective.prediction(values) - reference)))
            if change > IDENTIFIABILITY_TOLERANCE * scale:
                changes_profile = True
                break
        if not changes_profile:
            raise UnidentifiableParameterError(
                f"parameter '{name}' does not affect the profile of shape '{objective.model.shape.name}' "
                f"in the {objective.frame.value} frame; fix it"
            )


def _fit_in_regime(
    objective: _Objective, regime: Regime | None, free_spec: FreeSpec, spec: FitSpec, method: OptimizerMethod
) -> _RegimeOutcome:
    objective.free_spec = free_spec
    evaluations_before = objective.evaluation_count
    if not free_spec.free_names:
        empty = np.zeros(0, dtype=np.float64)
        optimum = _LocalOptimum(empty, objective(empty), True, "no free parameters")
        return _RegimeOutcome(regime, free_spec, optimum, 1, objective.evaluation_count - evaluations_before)

    starts = [
        point
        for point in _start_points(free_spec, spec.starts_per_parameter)
        if objective.is_feasible(free_spec.unpack(point))
    ]
    if not starts:
        raise InfeasibleParametersError(
            f"shape '{objective.model.shape.name}', regime {regime.value if regime else '—'}: no starting point "
            f"lies in the existence domain at {_describe_ranges(free_spec)}; "
            f"check the fixed values and the bounds"
        )
    _check_identifiable(objective, free_spec, starts[0])
    box: Bounds = Bounds(free_spec.lower_bounds, free_spec.upper_bounds)
    best: _LocalOptimum | None = None
    for start in starts:
        local = minimize(objective, start, method=method.value, bounds=box)
        if best is None or float(local.fun) < best.loss:
            best = _LocalOptimum(
                np.asarray(local.x, dtype=np.float64), float(local.fun), bool(local.success), str(local.message)
            )
    assert best is not None  # there is at least one start
    return _RegimeOutcome(regime, free_spec, best, len(starts), objective.evaluation_count - evaluations_before)


def _describe_ranges(free_spec: FreeSpec) -> str:
    """Bounds of the free and values of the fixed parameters in user units — for error messages."""
    parts = []
    for index, name in enumerate(free_spec.free_names):
        parameter = free_spec.parameters[name]
        lower = parameter.to_user(float(free_spec.lower_bounds[index]))
        upper = parameter.to_user(float(free_spec.upper_bounds[index]))
        parts.append(f"{name} ∈ [{lower:.4g}, {upper:.4g}]")
    for name, value in free_spec.fixed_values.items():
        parts.append(f"{name} = {free_spec.parameters[name].to_user(value):.4g}")
    return ", ".join(parts)


def _boundary_flags(outcome: _RegimeOutcome) -> tuple[tuple[str, ...], bool]:
    free_spec = outcome.free_spec
    at_bounds: list[str] = []
    at_regime_boundary = False
    regime_edges = (
        math.radians(REGIME_BOUNDARY_DEG - REGIME_GAP_DEG),
        math.radians(REGIME_BOUNDARY_DEG + REGIME_GAP_DEG),
    )
    for index, name in enumerate(free_spec.free_names):
        lower, upper = free_spec.lower_bounds[index], free_spec.upper_bounds[index]
        tolerance = BOUND_PROXIMITY_FRACTION * (upper - lower)
        value = outcome.optimum.position[index]
        parameter = free_spec.parameters[name]
        natural_lower = parameter.to_internal(parameter.bounds[0]) if parameter.natural_lower_bound else None
        for edge in (lower, upper):
            if edge == lower and natural_lower is not None and math.isclose(edge, natural_lower):
                continue  # physical bound (δ = 0): a value on it is the answer, not the optimizer hitting a bound
            if abs(value - edge) <= tolerance:
                if name == CONTACT_ANGLE_NAME and any(math.isclose(edge, regime_edge) for regime_edge in regime_edges):
                    at_regime_boundary = True
                elif name not in at_bounds:
                    at_bounds.append(name)
    return tuple(at_bounds), at_regime_boundary


def _build_result(
    objective: _Objective,
    outcome: _RegimeOutcome,
    reduced: ReducedProfile,
    spec: FitSpec,
    volume_fraction_from_data: float | None,
    alternatives: tuple[FitResult, ...],
) -> FitResult:
    model = objective.model
    free_spec = outcome.free_spec
    values = free_spec.unpack(outcome.optimum.position)
    user_values = free_spec.to_user(values)
    user_values.update(
        {name: value for name, value in spec.fixed.items() if name in user_values}
    )  # as the user gave them
    at_bounds, at_regime_boundary = _boundary_flags(outcome)
    wall_layer_nm = (
        user_values[WALL_LAYER_NAME] * model.pore.characteristic_size_nm() if WALL_LAYER_NAME in user_values else None
    )
    density_bulk = (
        reduced.density_scale_per_nm3 / values[VOLUME_FRACTION_NAME]
        if reduced.normalization is Normalization.MEAN and VOLUME_FRACTION_NAME in values
        else None
    )
    feasible = objective.is_feasible(values)
    curve = objective.prediction(values) if feasible else np.zeros_like(objective.data)
    return FitResult(
        library_version=library_version(),
        pore_name=model.pore.name,
        shape_name=model.shape.name,
        regime=outcome.regime,
        frame=reduced.frame,
        normalization=reduced.normalization,
        values=user_values,
        free_parameters=free_spec.free_names,
        fixed_parameters=tuple(free_spec.fixed_values),
        loss=outcome.optimum.loss,
        success=outcome.optimum.success,
        message=outcome.optimum.message,
        start_count=outcome.start_count,
        evaluation_count=outcome.evaluation_count,
        feasible=feasible,
        at_bounds=at_bounds,
        at_regime_boundary=at_regime_boundary,
        wall_layer_thickness_nm=wall_layer_nm,
        density_bulk_derived_per_nm3=density_bulk,
        volume_fraction_from_data=volume_fraction_from_data,
        coordinate=objective.coordinate,
        data=objective.data,
        model_curve=curve,
        alternatives=alternatives,
        spec=spec,
    )


def _within_fit_bounds(model: "InterfaceModel", spec: FitSpec, name: str, user_value: float) -> bool:
    lower, upper = model.parameters[name].bounds
    if name in spec.bounds:
        lower, upper = max(lower, spec.bounds[name][0]), min(upper, spec.bounds[name][1])
    return lower <= user_value <= upper


def fit(model: "InterfaceModel", profile: DensityProfile, spec: FitSpec) -> FitResult:
    """Fit the free parameters of the model to the profile. Algorithm — specification §4."""
    if not model.shape.has_profile():
        fittable = [other.shape.name for other in model.pore.available_interfaces() if other.shape.has_profile()]
        raise NotDerivedError(
            f"shape '{model.shape.name}' cannot be fitted: its density profile is not derived (the shape takes part "
            f"only in the stability diagram); shapes of pore '{model.pore.name}' with a profile: {', '.join(fittable)}"
        )
    geometry = model.pore.geometry()
    normalization = _resolve_normalization(spec)
    reduced = model.pore.reduce(profile, normalization, spec.density_bulk_per_nm3)
    window_mask = _window_mask(reduced, geometry, spec.window)
    _check_profile_matches_pore(reduced, geometry)
    _check_pore_volume(profile, model.pore.volume_nm3())
    optimize = spec.optimize if spec.optimize is not None else model.default_optimize
    has_volume_fraction = VOLUME_FRACTION_NAME in model.parameters.names
    if normalization is Normalization.MEAN and not has_volume_fraction:
        raise InvalidProfileError(
            f"mean normalization divides the model by φ, but shape '{model.shape.name}' has no φ parameter; "
            f"set density_bulk_per_nm3"
        )

    phi_unknown = VOLUME_FRACTION_NAME not in spec.fixed and VOLUME_FRACTION_NAME not in optimize
    if normalization is Normalization.MEAN and has_volume_fraction and phi_unknown:
        raise InvalidProfileError(
            f"mean normalization cannot take φ from the data (φ = ∫ρ/ρ_bulk needs ρ_bulk); set "
            f"density_bulk_per_nm3 (--rho-bulk), fix phi or free it in optimize (shape '{model.shape.name}')"
        )
    values_from_data: dict[str, float] = {}
    initial = dict(spec.initial)
    volume_fraction_from_data: float | None = None
    if normalization is Normalization.BULK and has_volume_fraction and VOLUME_FRACTION_NAME not in spec.fixed:
        volume_fraction_from_data = (
            histogram_integral(reduced.coordinate, reduced.density) / geometry.profile_integral()
        )
        if VOLUME_FRACTION_NAME not in optimize:
            values_from_data[VOLUME_FRACTION_NAME] = volume_fraction_from_data
        elif VOLUME_FRACTION_NAME not in initial and _within_fit_bounds(
            model, spec, VOLUME_FRACTION_NAME, volume_fraction_from_data
        ):
            # ∫ρ/ρ_bulk is the exact φ for any shape; starting from it prevents sliding to the edge of a plateau (the
            # step profile of the layer makes the loss piecewise constant in φ)
            initial[VOLUME_FRACTION_NAME] = volume_fraction_from_data

    free_spec = model.parameters.resolve(optimize, spec.fixed, spec.bounds, initial, values_from_data)
    objective = _Objective(model, free_spec, reduced, window_mask, spec.loss)
    method = spec.optimizer

    outcomes: list[_RegimeOutcome] = []
    failures: list[InfeasibleParametersError] = []
    for regime, regime_spec in _regime_intervals(model, free_spec, spec.regime):
        try:
            outcomes.append(_fit_in_regime(objective, regime, regime_spec, spec, method))
        except InfeasibleParametersError as error:
            failures.append(error)
    if not outcomes:
        raise failures[0]
    outcomes.sort(key=lambda outcome: outcome.optimum.loss)

    alternatives = tuple(
        _build_result(objective, outcome, reduced, spec, volume_fraction_from_data, ()) for outcome in outcomes[1:]
    )
    return _build_result(objective, outcomes[0], reduced, spec, volume_fraction_from_data, alternatives)
