"""Диаграмма устойчивости: какая форма интерфейса выгоднее при данных θ, δ на сетке (l, φ).

Устойчивой считается допустимая форма с наименьшим капиллярным функционалом S_LV − cos θ · S_SL
(у формы без θ — S_LV). Так же строит диаграмму PANDA_Polygon (`docs/model-runtime.js`, computePhase).
"""

from collections.abc import Sequence

from pydantic import ConfigDict

from panda_md.core.enums import ParameterName
from panda_md.core.errors import PandaError
from panda_md.core.model import InterfaceModel
from panda_md.core.pore import Pore
from panda_md.core.types import PandaModel

LENGTH_FIELD = "length_nm"  # поле поры, которое меняется вдоль оси l диаграммы


class StabilityDiagram(PandaModel):
    """Результат: stable_shapes[i][j] — устойчивая форма при length_ratios[i], volume_fractions[j] (None — ни одной)."""

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
    candidates = _candidate_names(pore, shape_names)
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
