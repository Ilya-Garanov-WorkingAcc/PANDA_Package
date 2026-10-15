"""Общий фит: один алгоритм для любой формы, любой поры и любого набора свободных параметров."""

import itertools
import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import Bounds, minimize

from panda_md.core.enums import Frame, LossFunction, Normalization, OptimizerMethod, ParameterName, Regime
from panda_md.core.errors import (
    EmptyWindowError,
    InfeasibleParametersError,
    InvalidProfileError,
    NotDerivedError,
    PandaError,
    ProfileMismatchError,
    UnidentifiableParameterError,
    UnsupportedRegimeError,
)
from panda_md.core.parameters import FreeSpec, ParameterValues
from panda_md.core.profile import DensityProfile, ReducedProfile
from panda_md.core.result import FitResult
from panda_md.core.shape import REGIME_BOUNDARY_DEG, REGIME_GAP_DEG, PoreGeometry
from panda_md.core.spec import FitSpec
from panda_md.core.types import BoolArray, FloatArray

if TYPE_CHECKING:
    from panda_md.core.model import InterfaceModel


MIN_WINDOW_POINTS = 10  # меньше точек в окне фита — ошибка EmptyWindowError
# шаги проверки «параметр влияет», доли ширины его границ: крупные нужны параметрам, двигающим лишь край
# профиля (φ у слоя), — малый сдвиг края не выходит за бин
IDENTIFIABILITY_STEP_FRACTIONS = (1e-3, 1e-2, 5e-2)
IDENTIFIABILITY_TOLERANCE = 1e-9  # относительное изменение профиля, ниже которого параметр «не влияет»
BOUND_PROXIMITY_FRACTION = 1e-3  # ближе этой доли ширины границ — параметр «у границы»
BARRIER_LEVEL_FACTOR = 10.0  # уровень барьера: во столько раз выше ошибки нулевой модели (+1)
BARRIER_SLOPE = 100.0  # наклон барьера по величине выхода φ за границы существования
COVERAGE_TOLERANCE = 0.05  # данные должны доходить до стенок с точностью 5 % размера поры
OUTSIDE_MASS_TOLERANCE = 0.05  # допустимая доля ∫ρ за пределами поры (система поры)

CONTACT_ANGLE_NAME = ParameterName.CONTACT_ANGLE.value
WALL_LAYER_NAME = ParameterName.WALL_LAYER_THICKNESS.value
VOLUME_FRACTION_NAME = ParameterName.VOLUME_FRACTION.value
PROFILE_SHIFT_NAME = ParameterName.PROFILE_SHIFT.value


class _Objective:
    """Целевая функция фита: модель в системе отсчёта данных, сравнение с данными, барьер."""

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
        self.barrier_level = BARRIER_LEVEL_FACTOR * (self.loss(np.zeros_like(self.data)) + 1.0)
        self.evaluation_count = 0

    def loss(self, curve: FloatArray) -> float:
        residual = curve - self.data
        if self.loss_function is LossFunction.L1:
            return float(np.mean(np.abs(residual)))
        return float(np.mean(residual**2))

    def prediction(self, values: ParameterValues) -> FloatArray:
        """Модельная безразмерная плотность на координатах данных."""
        shape = self.model.shape
        position = self.coordinate - values.get(PROFILE_SHIFT_NAME, 0.0)
        if self.frame is not shape.frame:
            if self.frame is Frame.LIQUID and shape.frame is Frame.PORE:
                position = position + shape.center_of_mass(self.geometry, values)
            else:
                raise PandaError(
                    f"данные в системе {self.frame.value}, формулы формы {shape.name} — в системе "
                    f"{shape.frame.value}: такой пересчёт не поддержан"
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
    """Лучшая точка локальной минимизации (общий вид для любого метода scipy)."""

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
    evaluation_count: int  # вычислений целевой функции в этом режиме

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
            f"в окне фита [{lower:.3g}, {upper:.3g}] {point_count} точек (нужно не меньше {MIN_WINDOW_POINTS}); "
            f"координата данных в долях размера поры: [{float(np.min(reduced.coordinate)):.3g}, "
            f"{float(np.max(reduced.coordinate)):.3g}] — проверь размер поры и систему отсчёта профиля"
        )
    return mask


def _check_profile_matches_pore(reduced: ReducedProfile, geometry: PoreGeometry) -> None:
    """Профиль должен соответствовать поре по размеру: иначе фит молча вернёт неверный угол."""
    coordinate, density = reduced.coordinate, reduced.density
    pore_lower, pore_upper = geometry.coordinate_bounds()
    data_range = f"[{float(np.min(coordinate)):.3g}, {float(np.max(coordinate)):.3g}]"
    if reduced.frame is Frame.PORE:
        if np.min(coordinate) > pore_lower + COVERAGE_TOLERANCE or np.max(coordinate) < pore_upper - COVERAGE_TOLERANCE:
            raise ProfileMismatchError(
                f"данные не покрывают пору: координата в долях размера поры {data_range}, стенки — "
                f"[{pore_lower:.3g}, {pore_upper:.3g}]; проверь размер поры и единицы профиля"
            )
        outside = (coordinate < pore_lower) | (coordinate > pore_upper)
        outside_fraction = float(np.sum(density[outside]) / np.sum(density))
        if outside_fraction > OUTSIDE_MASS_TOLERANCE:
            raise ProfileMismatchError(
                f"{outside_fraction:.0%} плотности за пределами поры (координата {data_range}); проверь размер поры, "
                f"единицы профиля и центрирование"
            )
        return
    occupied = coordinate[density > 0.0]
    occupied_width = float(np.max(occupied) - np.min(occupied))
    if occupied_width > (pore_upper - pore_lower) * (1.0 + COVERAGE_TOLERANCE):
        raise ProfileMismatchError(
            f"профиль шире поры: жидкость занимает {occupied_width:.3g} в долях размера поры при ширине поры "
            f"{pore_upper - pore_lower:.3g}; проверь размер поры и единицы профиля"
        )


def _regime_intervals(
    model: "InterfaceModel", free_spec: FreeSpec, requested: Regime | None
) -> list[tuple[Regime | None, FreeSpec]]:
    shape = model.shape
    if CONTACT_ANGLE_NAME not in model.parameters.names:
        if requested is not None:
            raise UnsupportedRegimeError(
                f"у формы {shape.name} нет контактного угла — режим смачивания {requested.value} не задаётся"
            )
        return [(None, free_spec)]
    if CONTACT_ANGLE_NAME not in free_spec.free_names:
        regime = shape.require_supported_contact_angle(math.degrees(free_spec.fixed_values[CONTACT_ANGLE_NAME]))
        if requested is not None and requested is not regime:
            raise UnsupportedRegimeError(
                f"форма {shape.name}: зафиксированный θ относится к режиму {regime.value}, а запрошен {requested.value}"
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
            f"форма {shape.name}: границы θ [{lower_deg:.1f}°, {upper_deg:.1f}°] не пересекаются с объявленными "
            f"режимами {declared}"
        )
    return intervals


def _start_points(free_spec: FreeSpec, starts_per_parameter: int) -> list[FloatArray]:
    """Центры ячеек равномерной сетки по свободным параметрам + стартовая точка пользователя."""
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
                f"параметр {name} не влияет на профиль формы {objective.model.shape.name} "
                f"в системе отсчёта {objective.frame.value}: зафиксируй его"
            )


def _fit_in_regime(
    objective: _Objective, regime: Regime | None, free_spec: FreeSpec, spec: FitSpec, method: OptimizerMethod
) -> _RegimeOutcome:
    objective.free_spec = free_spec
    evaluations_before = objective.evaluation_count
    if not free_spec.free_names:
        empty = np.zeros(0, dtype=np.float64)
        optimum = _LocalOptimum(empty, objective(empty), True, "нет свободных параметров")
        return _RegimeOutcome(regime, free_spec, optimum, 1, objective.evaluation_count - evaluations_before)

    starts = [
        point
        for point in _start_points(free_spec, spec.starts_per_parameter)
        if objective.is_feasible(free_spec.unpack(point))
    ]
    if not starts:
        raise InfeasibleParametersError(
            f"форма {objective.model.shape.name}, режим {regime.value if regime else '—'}: ни одна стартовая точка "
            f"не попадает в область существования при {_describe_ranges(free_spec)}; "
            f"проверь фиксированные значения и границы"
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
    assert best is not None  # стартов не меньше одного
    return _RegimeOutcome(regime, free_spec, best, len(starts), objective.evaluation_count - evaluations_before)


def _describe_ranges(free_spec: FreeSpec) -> str:
    """Границы свободных и значения фиксированных параметров в единицах пользователя — для сообщений об ошибках."""
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
        for edge in (lower, upper):
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
    )  # как задал пользователь
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
    """Подобрать свободные параметры модели под профиль. Алгоритм — спецификация §4."""
    if not model.shape.has_profile():
        fittable = [other.shape.name for other in model.pore.available_interfaces() if other.shape.has_profile()]
        raise NotDerivedError(
            f"фит формы {model.shape.name} невозможен: профиль плотности не выведен (форма участвует только "
            f"в диаграмме устойчивости); формы поры {model.pore.name} с профилем: {', '.join(fittable)}"
        )
    geometry = model.pore.geometry()
    normalization = _resolve_normalization(spec)
    reduced = model.pore.reduce(profile, normalization, spec.density_bulk_per_nm3)
    window_mask = _window_mask(reduced, geometry, spec.window)
    _check_profile_matches_pore(reduced, geometry)
    optimize = spec.optimize if spec.optimize is not None else model.default_optimize
    has_volume_fraction = VOLUME_FRACTION_NAME in model.parameters.names
    if normalization is Normalization.MEAN and not has_volume_fraction:
        raise InvalidProfileError(
            f"нормировка mean делит модель на φ, а у формы {model.shape.name} параметра φ нет; "
            f"задай density_bulk_per_nm3"
        )

    values_from_data: dict[str, float] = {}
    initial = dict(spec.initial)
    volume_fraction_from_data: float | None = None
    if normalization is Normalization.BULK and has_volume_fraction and VOLUME_FRACTION_NAME not in spec.fixed:
        volume_fraction_from_data = float(trapezoid(reduced.density, reduced.coordinate) / geometry.profile_integral())
        if VOLUME_FRACTION_NAME not in optimize:
            values_from_data[VOLUME_FRACTION_NAME] = volume_fraction_from_data
        elif VOLUME_FRACTION_NAME not in initial and _within_fit_bounds(
            model, spec, VOLUME_FRACTION_NAME, volume_fraction_from_data
        ):
            # ∫ρ/ρ_bulk — точная φ для любой формы; старт из неё не даёт сползти на край плато (ступенчатый
            # профиль слоя делает ошибку кусочно-постоянной по φ)
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
