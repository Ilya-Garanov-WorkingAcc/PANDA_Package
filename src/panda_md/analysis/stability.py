"""Диаграмма устойчивости: какая форма интерфейса выгоднее при данных θ, δ на сетке (l, φ).

Устойчивой считается допустимая форма с наименьшим капиллярным функционалом S_LV − cos θ · S_SL
(у формы без θ — S_LV). Так же строит диаграмму PANDA_Polygon (`docs/model-runtime.js`, computePhase).
"""

import math
import warnings
from collections.abc import Sequence

from pydantic import ConfigDict

from panda_md.core.enums import ParameterName
from panda_md.core.errors import PandaError, PandaWarning, UnsupportedRegimeError
from panda_md.core.model import InterfaceModel
from panda_md.core.parameters import VOLUME_FRACTION, WALL_LAYER_THICKNESS
from panda_md.core.pore import Pore
from panda_md.core.shape import REGIME_BOUNDARY_DEG, REGIME_GAP_DEG, regime_of
from panda_md.core.types import PandaModel

LENGTH_FIELD = "length_nm"  # поле поры, которое меняется вдоль оси l диаграммы
MAX_CONTACT_ANGLE_DEG = 180.0
RADIAN_SUSPICION_DEG = 2.0 * math.pi  # θ ≤ 2π° — скорее всего, передан угол в радианах


class StabilityDiagram(PandaModel):
    """Результат: `stable_shapes[i][j]` — устойчивая форма при `length_ratios[i]`, `volume_fractions[j]`.

    None — ни одна форма не существует.
    """

    model_config = ConfigDict(frozen=True)

    pore_name: str
    contact_angle_deg: float
    wall_layer: float
    volume_fractions: tuple[float, ...]
    length_ratios: tuple[float, ...]  # l = L / (характерный размер поры)
    shape_names: tuple[str, ...]  # формы-кандидаты: у них выведены площади поверхностей
    stable_shapes: tuple[tuple[str | None, ...], ...]


def stability_diagram(
    pore: Pore,
    *,
    contact_angle_deg: float,
    wall_layer: float,
    volume_fractions: Sequence[float],
    length_ratios: Sequence[float],
    shape_names: Sequence[str] | None = None,
) -> StabilityDiagram:
    """Построить диаграмму для поры `pore`; длина поры берётся из `length_ratios`, остальные размеры — из `pore`."""
    if LENGTH_FIELD not in type(pore).model_fields:
        raise PandaError(f"у поры {pore.name} нет поля {LENGTH_FIELD}: диаграмму по длине не построить")
    if any(ratio <= 0.0 for ratio in length_ratios):
        raise PandaError(f"длина поры l должна быть положительной: {list(length_ratios)}")
    if not 0.0 <= contact_angle_deg <= MAX_CONTACT_ANGLE_DEG:
        raise PandaError(f"угол θ = {contact_angle_deg}° вне [0°, 180°]")
    if 0.0 < contact_angle_deg <= RADIAN_SUSPICION_DEG:
        warnings.warn(
            f"θ = {contact_angle_deg}° — проверь, что угол задан в градусах, а не в радианах",
            PandaWarning,
            stacklevel=2,
        )
    if shape_names is not None and len(set(shape_names)) != len(shape_names):
        raise PandaError(f"повтор в списке форм: {list(shape_names)}")
    _check_parameter_values(wall_layer, volume_fractions)
    candidates = _supported_candidates(pore, _candidate_names(pore, shape_names), contact_angle_deg)
    size_nm = pore.characteristic_size_nm()
    rows: list[tuple[str | None, ...]] = []
    for length_ratio in length_ratios:
        sized_pore = pore.model_copy(update={LENGTH_FIELD: length_ratio * size_nm})
        models = [sized_pore.interface(name) for name in candidates]
        rows.append(
            tuple(
                _stable_shape(models, contact_angle_deg, wall_layer, volume_fraction)
                for volume_fraction in volume_fractions
            )
        )
    return StabilityDiagram(
        pore_name=pore.name,
        contact_angle_deg=contact_angle_deg,
        wall_layer=wall_layer,
        volume_fractions=tuple(volume_fractions),
        length_ratios=tuple(length_ratios),
        shape_names=candidates,
        stable_shapes=tuple(rows),
    )


def _check_parameter_values(wall_layer: float, volume_fractions: Sequence[float]) -> None:
    """δ и φ — в границах параметров модели: за ними формулы молча экстраполируются."""
    lower, upper = WALL_LAYER_THICKNESS.bounds
    if not lower <= wall_layer <= upper:
        raise PandaError(f"δ = {wall_layer} вне границ параметра [{lower}, {upper}]")
    lower, upper = VOLUME_FRACTION.bounds
    outside = [value for value in volume_fractions if not lower <= value <= upper]
    if outside:
        raise PandaError(f"φ вне [{lower}, {upper}]: {outside}")


def _supported_candidates(pore: Pore, candidates: tuple[str, ...], contact_angle_deg: float) -> tuple[str, ...]:
    """Формы, которые при этом θ могут существовать и для которых выведены формулы.
    Форма с θ вне диапазона своего режима просто не существует (у многоугольных пор режимы узкие) — она выпадает
    молча. Форма, у которой режим этого θ не выведен вообще (θ < 90° у форм щели, O1), выпадает с предупреждением;
    если так у всех форм с углом, диаграмма не определена — ошибка, а не «слой во всей поре».
    """
    registry = pore.own_registry()
    supported: list[str] = []
    not_derived: list[str] = []
    with_angle = 0
    for name in candidates:
        shape = registry[name]()
        if ParameterName.CONTACT_ANGLE.value not in {parameter.name for parameter in shape.parameters}:
            supported.append(name)  # форма без θ (слой) от режима не зависит
            continue
        with_angle += 1
        try:
            shape.require_supported_contact_angle(contact_angle_deg)
        except UnsupportedRegimeError:
            if _in_regime_gap(contact_angle_deg) or regime_of(contact_angle_deg) not in shape.regimes:
                not_derived.append(name)
        else:
            supported.append(name)
    if with_angle and len(not_derived) == with_angle:
        raise UnsupportedRegimeError(
            f"при θ = {contact_angle_deg:g}° формулы не выведены ни у одной формы поры {pore.name} с углом "
            f"({', '.join(not_derived)}): диаграмма не определена"
        )
    if not_derived:
        warnings.warn(
            f"при θ = {contact_angle_deg:g}° формулы режима не выведены — в диаграмме не участвуют: "
            f"{', '.join(not_derived)}",
            PandaWarning,
            stacklevel=3,
        )
    return tuple(supported)


def _in_regime_gap(contact_angle_deg: float) -> bool:
    return abs(contact_angle_deg - REGIME_BOUNDARY_DEG) < REGIME_GAP_DEG


def _candidate_names(pore: Pore, shape_names: Sequence[str] | None) -> tuple[str, ...]:
    registered = pore.own_registry()
    with_energy = tuple(name for name, shape_class in registered.items() if shape_class().has_surface_terms())
    if shape_names is None:
        return with_energy
    unusable = [name for name in shape_names if name not in with_energy]
    if unusable:
        raise PandaError(
            f"в поре {pore.name} нет форм с выведенными площадями поверхностей: {unusable}; "
            f"доступны: {list(with_energy)}"
        )
    return tuple(shape_names)


def _stable_shape(
    models: Sequence[InterfaceModel], contact_angle_deg: float, wall_layer: float, volume_fraction: float
) -> str | None:
    """Допустимая форма с наименьшей энергией; при равенстве — первая в списке кандидатов."""
    point = {
        ParameterName.CONTACT_ANGLE.value: contact_angle_deg,
        ParameterName.WALL_LAYER_THICKNESS.value: wall_layer,
        ParameterName.VOLUME_FRACTION.value: volume_fraction,
    }
    best_name: str | None = None
    best_energy = float("inf")
    for model in models:
        values = {name: value for name, value in point.items() if name in model.parameters.names}
        if not model.is_feasible(**values):
            continue
        energy = model.capillary_energy(**values)
        if energy < best_energy:
            best_name, best_energy = model.shape.name, energy
    return best_name
