"""Диаграмма устойчивости: в каждой точке (l, φ) — форма с наименьшим капиллярным функционалом."""

import math
from typing import Any, ClassVar

import pytest

from panda_md import SlitPore, SquarePore, TrianglePore
from panda_md.analysis.stability import stability_diagram
from panda_md.core.errors import PandaError, PandaWarning, UnsupportedRegimeError
from panda_md.pores.slit.roll import SlitRoll

SQUARE = SquarePore(side_nm=6.0, length_nm=12.0, layer_width_nm=6.0)


def test_each_point_holds_the_feasible_shape_with_lowest_energy() -> None:
    diagram = stability_diagram(
        SQUARE, contact_angle_deg=150.0, wall_layer=0.02, volume_fractions=(0.05, 0.3, 0.6), length_ratios=(1.0, 4.0)
    )
    assert diagram.volume_fractions == (0.05, 0.3, 0.6)
    assert diagram.length_ratios == (1.0, 4.0)
    for length_index, length_ratio in enumerate(diagram.length_ratios):
        pore = SquarePore(side_nm=6.0, length_nm=length_ratio * 6.0, layer_width_nm=6.0)
        for phi_index, phi in enumerate(diagram.volume_fractions):
            energies: dict[str, float] = {}
            for model in pore.available_interfaces():
                values = {"theta": 150.0, "delta": 0.02, "phi": phi}
                values = {name: value for name, value in values.items() if name in model.parameters.names}
                if model.is_feasible(**values):
                    energies[model.shape.name] = model.capillary_energy(**values)
            expected = min(energies, key=energies.__getitem__) if energies else None
            assert diagram.stable_shapes[length_index][phi_index] == expected


def test_shapes_can_be_restricted_and_unknown_names_fail() -> None:
    diagram = stability_diagram(
        SQUARE,
        contact_angle_deg=150.0,
        wall_layer=0.0,
        volume_fractions=(0.1, 0.5),
        length_ratios=(2.0,),
        shape_names=("filament_w1c0", "droplet_w1c0"),
    )
    assert diagram.shape_names == ("filament_w1c0", "droplet_w1c0")
    assert {name for row in diagram.stable_shapes for name in row} <= {"filament_w1c0", "droplet_w1c0", None}
    with pytest.raises(PandaError, match="perforation"):
        stability_diagram(
            SQUARE,
            contact_angle_deg=150.0,
            wall_layer=0.0,
            volume_fractions=(0.1,),
            length_ratios=(2.0,),
            shape_names=("perforation",),
        )


def test_shapes_without_surface_terms_are_left_out() -> None:
    slit = SlitPore(width_nm=8.0, length_nm=16.0)
    diagram = stability_diagram(
        slit, contact_angle_deg=120.0, wall_layer=0.0, volume_fractions=(0.2,), length_ratios=(2.0,)
    )
    assert "doughnut" not in diagram.shape_names
    assert "perforation" not in diagram.shape_names
    assert "roll" in diagram.shape_names


def test_points_where_no_shape_exists_are_empty() -> None:
    triangle = TrianglePore(side_nm=4.0, length_nm=8.0, layer_width_nm=4.0)
    diagram = stability_diagram(
        triangle, contact_angle_deg=150.0, wall_layer=0.1, volume_fractions=(0.9,), length_ratios=(2.0,)
    )
    assert diagram.stable_shapes == ((None,),)
    unreachable_layer = 1.0 / (2.0 * math.sqrt(3.0))  # a′ = 0: вне границ δ — ошибка (итоговый аудит)
    with pytest.raises(PandaError, match="вне"):
        stability_diagram(
            triangle,
            contact_angle_deg=150.0,
            wall_layer=unreachable_layer,
            volume_fractions=(0.1,),
            length_ratios=(2.0,),
        )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"length_ratios": (0.0, 2.0)}, "длина"),
        ({"contact_angle_deg": 2.6}, None),
        ({"contact_angle_deg": 190.0}, "угол"),
        ({"shape_names": ("roll", "roll")}, "повтор"),
    ],
)
def test_stability_diagram_checks_its_input(arguments: dict[str, object], message: str | None) -> None:
    slit = SlitPore(width_nm=8.0, length_nm=16.0)
    call: dict[str, Any] = {
        "contact_angle_deg": 120.0,
        "wall_layer": 0.0,
        "volume_fractions": (0.2,),
        "length_ratios": (2.0,),
        **arguments,
    }
    if message is None:  # угол в радианах: предупредить; формул при θ ≈ 2.6° у щели нет — ошибка (итоговый аудит)
        with pytest.warns(PandaWarning, match="градус"), pytest.raises(UnsupportedRegimeError):
            stability_diagram(slit, **call)
        return
    with pytest.raises(PandaError, match=message):
        stability_diagram(slit, **call)


class TiePore(SlitPore):
    name: ClassVar[str] = "tie"


@TiePore.register
class FirstRoll(SlitRoll):
    name: ClassVar[str] = "first_roll"


@TiePore.register
class SecondRoll(SlitRoll):
    name: ClassVar[str] = "second_roll"


def test_equal_energies_go_to_the_first_candidate() -> None:
    pore = TiePore(width_nm=8.0, length_nm=16.0)
    for order in (("first_roll", "second_roll"), ("second_roll", "first_roll")):
        diagram = stability_diagram(
            pore,
            contact_angle_deg=120.0,
            wall_layer=0.0,
            volume_fractions=(0.3,),
            length_ratios=(2.0,),
            shape_names=order,
        )
        assert diagram.stable_shapes == ((order[0],),)


def test_zero_angle_is_not_mistaken_for_radians() -> None:
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stability_diagram(
            SquarePore(side_nm=4.0, length_nm=8.0, layer_width_nm=4.0),  # у квадрата смачивание выведено
            contact_angle_deg=0.0,
            wall_layer=0.0,
            volume_fractions=(0.2,),
            length_ratios=(2.0,),
        )
    assert not [warning for warning in caught if "градус" in str(warning.message)]


def test_diagram_outside_derived_regime_is_not_silently_layer() -> None:
    """Итоговый аудит: при θ < 90° формы с θ молча выпадали, и вся щель становилась «layer»."""
    from panda_md import SlitPore, SquarePore
    from panda_md.core.errors import PandaWarning, UnsupportedRegimeError

    for angle in (60.0, 90.0):
        with pytest.raises(UnsupportedRegimeError, match="не выведены"):
            stability_diagram(SlitPore(width_nm=4.0, length_nm=20.0), contact_angle_deg=angle, wall_layer=0.0,
                              volume_fractions=[0.3], length_ratios=[2.0])  # fmt: skip
    square = SquarePore(side_nm=4.0, length_nm=8.0, layer_width_nm=4.0)
    with pytest.warns(PandaWarning, match="не участвуют"):
        diagram = stability_diagram(square, contact_angle_deg=60.0, wall_layer=0.0, volume_fractions=[0.3],
                                    length_ratios=[2.0])  # fmt: skip
    assert "filament_w1c0" in diagram.shape_names and "droplet_w1c0" not in diagram.shape_names


@pytest.mark.parametrize(("wall_layer", "volume_fraction"), [(-0.3, 0.3), (0.3, 0.3), (0.0, 1.3), (0.0, -0.1)])
def test_diagram_rejects_values_outside_parameter_bounds(wall_layer: float, volume_fraction: float) -> None:
    from panda_md import SlitPore
    from panda_md.core.errors import PandaError

    with pytest.raises(PandaError, match="вне"):
        stability_diagram(SlitPore(width_nm=4.0, length_nm=20.0), contact_angle_deg=130.0, wall_layer=wall_layer,
                          volume_fractions=[volume_fraction], length_ratios=[2.0])  # fmt: skip
