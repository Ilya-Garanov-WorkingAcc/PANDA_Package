"""Диаграмма устойчивости: в каждой точке (l, φ) — форма с наименьшим капиллярным функционалом."""

import math

import pytest

from panda_md import SlitPore, SquarePore, TrianglePore
from panda_md.analysis.stability import stability_diagram
from panda_md.core.errors import PandaError

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
    triangle = TrianglePore(side_nm=6.0, length_nm=12.0, layer_width_nm=6.0)
    unreachable_layer = 1.0 / (2.0 * math.sqrt(3.0))  # a′ = 0: доступного сечения нет
    diagram = stability_diagram(
        triangle, contact_angle_deg=150.0, wall_layer=unreachable_layer, volume_fractions=(0.1,), length_ratios=(2.0,)
    )
    assert diagram.stable_shapes == ((None,),)
