"""Stability diagram: which interface shape is favourable for given θ, δ on an (l, φ) grid.

The stable shape is the feasible shape with the lowest capillary functional S_LV − cos θ · S_SL
(S_LV for a shape without θ). PANDA_Polygon builds the diagram the same way (`docs/model-runtime.js`, computePhase).
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

LENGTH_FIELD = "length_nm"  # the pore field that varies along the l axis of the diagram
MAX_CONTACT_ANGLE_DEG = 180.0
RADIAN_SUSPICION_DEG = 2.0 * math.pi  # θ ≤ 2π° — most likely an angle in radians was passed


class StabilityDiagram(PandaModel):
    """Result: `stable_shapes[i][j]` is the stable shape at `length_ratios[i]`, `volume_fractions[j]`.

    None means that no shape exists.
    """

    model_config = ConfigDict(frozen=True)

    pore_name: str
    contact_angle_deg: float
    wall_layer: float
    volume_fractions: tuple[float, ...]
    length_ratios: tuple[float, ...]  # l = L / (characteristic pore size)
    shape_names: tuple[str, ...]  # candidate shapes: their surface areas are derived
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
    """Build the diagram for the pore `pore`; the pore length comes from `length_ratios`, other sizes from `pore`."""
    if LENGTH_FIELD not in type(pore).model_fields:
        raise PandaError(f"pore '{pore.name}' has no field {LENGTH_FIELD}; a diagram over the length cannot be built")
    if any(ratio <= 0.0 for ratio in length_ratios):
        raise PandaError(f"pore length l must be positive: {list(length_ratios)}")
    if not 0.0 <= contact_angle_deg <= MAX_CONTACT_ANGLE_DEG:
        raise PandaError(f"angle θ = {contact_angle_deg}° is outside [0°, 180°]")
    if 0.0 < contact_angle_deg <= RADIAN_SUSPICION_DEG:
        warnings.warn(
            f"θ = {contact_angle_deg}°; check that the angle is given in degrees, not in radians",
            PandaWarning,
            stacklevel=2,
        )
    if shape_names is not None and len(set(shape_names)) != len(shape_names):
        raise PandaError(f"duplicate in the shape list: {', '.join(shape_names)}")
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
    """δ and φ must be within the model parameter bounds: beyond them the formulas silently extrapolate."""
    lower, upper = WALL_LAYER_THICKNESS.bounds
    if not lower <= wall_layer <= upper:
        raise PandaError(f"δ = {wall_layer} is outside the parameter bounds [{lower}, {upper}]")
    lower, upper = VOLUME_FRACTION.bounds
    outside = [value for value in volume_fractions if not lower <= value <= upper]
    if outside:
        raise PandaError(f"φ outside [{lower}, {upper}]: {', '.join(f'{value:g}' for value in outside)}")


def _supported_candidates(pore: Pore, candidates: tuple[str, ...], contact_angle_deg: float) -> tuple[str, ...]:
    """Shapes that can exist at this θ and whose formulas are derived.
    A shape with θ outside the range of its regime simply does not exist (polygonal pores have narrow regimes) — it
    is dropped silently. A shape whose regime for this θ is not derived at all (θ < 90° for slit shapes) is
    dropped with a warning; if that holds for all shapes with an angle, the diagram is not defined — an error, not
    “a layer in the whole pore”.
    """
    registry = pore.own_registry()
    supported: list[str] = []
    not_derived: list[str] = []
    with_angle = 0
    for name in candidates:
        shape = registry[name]()
        if ParameterName.CONTACT_ANGLE.value not in {parameter.name for parameter in shape.parameters}:
            supported.append(name)  # a shape without θ (layer) does not depend on the regime
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
            f"at θ = {contact_angle_deg:g}° the formulas are not derived for any shape with an angle in pore "
            f"'{pore.name}' ({', '.join(not_derived)}): the diagram is not defined"
        )
    if not_derived:
        warnings.warn(
            f"at θ = {contact_angle_deg:g}° the regime formulas are not derived; excluded from the diagram: "
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
            f"pore '{pore.name}' has no shapes with derived surface areas: {unusable}; "
            f"available: {', '.join(with_energy)}"
        )
    return tuple(shape_names)


def _stable_shape(
    models: Sequence[InterfaceModel], contact_angle_deg: float, wall_layer: float, volume_fraction: float
) -> str | None:
    """The feasible shape with the lowest energy; on a tie, the first one in the candidate list."""
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
