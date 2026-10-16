"""Трёхмерные формы щели (contains) и оси симметрии."""

import itertools
import math

import numpy as np
import pytest

from panda_md import SlitPore, SquarePore
from panda_md.core.centering import SymmetryAxis, axis_position
from panda_md.core.enums import AxisLocator, AxisOrientation
from panda_md.core.errors import NotDerivedError
from panda_md.core.types import FloatArray
from panda_md.pores.slit import SlitGeometry
from panda_md.pores.slit.droplet import SlitDroplet
from panda_md.pores.slit.worm import SlitWorm

SEED = 17
POINTS = 400_000


def cell_points(geometry: SlitGeometry, generator: np.random.Generator) -> FloatArray:
    """Точки, равномерно заполняющие ячейку щели (безразмерные, ось формы в начале координат)."""
    uniform = generator.uniform(-0.5, 0.5, size=(POINTS, 3))
    points: FloatArray = uniform * np.array([geometry.pore_length_ratio, geometry.short_side_ratio(), 1.0])
    return points


def test_droplet_is_a_spherical_cap_on_the_lower_wall() -> None:
    pore = SlitPore(width_nm=8.0, length_nm=32.0, base_aspect_ratio=0.5)
    values = pore.interface("droplet").internal_values({"theta": 130.0, "delta": 0.03, "phi": 0.02})
    shape, geometry = SlitDroplet(), pore.geometry()
    square = geometry.equivalent_square()
    square_values = shape.square_values(geometry, values)
    radius, center = shape.sphere_radius(square, square_values), shape.sphere_center(square, square_values)
    points = cell_points(geometry, np.random.default_rng(SEED))
    sphere = (points[:, 0] ** 2 + points[:, 1] ** 2 + (points[:, 2] - center) ** 2 < radius**2) & (
        points[:, 2] > -0.5 + 0.03
    )
    agreement = np.mean(shape.contains(points, geometry, values) == sphere)
    assert agreement > 0.9995  # расхождение — только точки на самой границе (дискретизация не участвует)


def test_worm_is_a_cylindrical_segment_along_the_short_side() -> None:
    pore = SlitPore(width_nm=8.0, length_nm=32.0, base_aspect_ratio=0.5)
    values = pore.interface("worm").internal_values({"theta": 130.0, "delta": 0.03, "phi": 0.1})
    shape, geometry = SlitWorm(), pore.geometry()
    radius, center = shape.cylinder_radius(geometry, values), shape.cylinder_center(geometry, values)
    points = cell_points(geometry, np.random.default_rng(SEED))
    cylinder = (points[:, 0] ** 2 + (points[:, 2] - center) ** 2 < radius**2) & (points[:, 2] > -0.5 + 0.03)
    assert np.mean(shape.contains(points, geometry, values) == cylinder) > 0.9995


CASES = [
    ("droplet", {"theta": 130.0, "delta": 0.03, "phi": 0.02}),
    ("doughnut", {"theta": 130.0, "delta": 0.03, "phi": 0.15}),
    ("worm", {"theta": 130.0, "delta": 0.03, "phi": 0.1}),
    ("roll", {"theta": 120.0, "delta": 0.03, "phi": 0.4}),
    ("perforation", {"theta": 130.0, "delta": 0.03, "phi": 0.8}),
    ("layer", {"delta": 0.03, "phi": 0.4}),
]


@pytest.mark.parametrize("aspect_ratio", [1.0, 0.4])
@pytest.mark.parametrize(("shape_name", "values"), CASES, ids=[name for name, _ in CASES])
def test_point_cloud_reproduces_the_profile(shape_name: str, values: dict[str, float], aspect_ratio: float) -> None:
    """Доля точек в жидкости по слоям = профиль (контракт §9: облако даёт тот же профиль; допуск — статистика)."""
    pore = SlitPore(width_nm=8.0, length_nm=16.0, base_aspect_ratio=aspect_ratio)
    model = pore.interface(shape_name)
    assert model.is_feasible(**values), values
    internal = model.internal_values(values)
    points = cell_points(pore.geometry(), np.random.default_rng(SEED))
    inside = model.shape.contains(points, pore.geometry(), internal)
    phi = internal["phi"]
    assert np.mean(inside) == pytest.approx(phi, abs=4.0 * math.sqrt(phi * (1.0 - phi) / POINTS))  # 4σ выборки
    edges = np.linspace(-0.5, 0.5, 21)
    counts, _ = np.histogram(points[inside, 2], bins=edges)
    totals, _ = np.histogram(points[:, 2], bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fine = np.linspace(-0.5, 0.5, 20001)
    expected = [
        np.mean(model.shape.profile(fine[(fine >= low) & (fine < high)], pore.geometry(), internal))
        for low, high in itertools.pairwise(edges)
    ]
    np.testing.assert_allclose(counts / totals, expected, atol=0.02, err_msg=str(centers))


def test_shapes_declare_their_symmetry_axes() -> None:
    pore = SlitPore(width_nm=8.0, length_nm=16.0)
    axes = {name: pore.interface(name).shape.symmetry_axis for name in pore.shape_names()}
    assert axes["roll"] == SymmetryAxis(orientation=AxisOrientation.HORIZONTAL, locate=AxisLocator.LIQUID_CENTER)
    assert axes["doughnut"] == SymmetryAxis(orientation=AxisOrientation.VERTICAL, locate=AxisLocator.LIQUID_CENTER)
    assert axes["perforation"] == SymmetryAxis(orientation=AxisOrientation.VERTICAL, locate=AxisLocator.VOID_CENTER)
    assert axes["droplet"] is None and axes["worm"] is None and axes["layer"] is None


def test_axis_position_through_box_boundary() -> None:
    periods = np.array([10.0, 4.0])
    around_edge = np.array([[9.5, 1.8], [0.5, 2.2], [9.8, 1.9], [0.2, 2.1]])
    vertical = SymmetryAxis(orientation=AxisOrientation.VERTICAL, locate=AxisLocator.LIQUID_CENTER)
    position = axis_position(around_edge, periods, vertical)
    assert min(position[0], 10.0 - position[0]) == pytest.approx(0.0, abs=0.2)
    assert position[1] == pytest.approx(2.0, abs=0.07)  # гистограмма: точность — доля бина
    horizontal = SymmetryAxis(orientation=AxisOrientation.HORIZONTAL, locate=AxisLocator.LIQUID_CENTER)
    assert axis_position(around_edge, periods, horizontal)[1] == 0.0
    void = SymmetryAxis(orientation=AxisOrientation.VERTICAL, locate=AxisLocator.VOID_CENTER)
    void_position = axis_position(around_edge, periods, void)
    assert void_position[0] == pytest.approx(5.0, abs=0.2)
    assert min(void_position[1], 4.0 - void_position[1]) == pytest.approx(0.0, abs=0.07)


def test_polygon_shapes_have_no_3d_form_yet() -> None:
    square = SquarePore(side_nm=6.0, length_nm=12.0, layer_width_nm=6.0)
    droplet = square.interface("droplet_w1c0")
    assert not droplet.shape.has_contains()
    with pytest.raises(NotDerivedError, match="contains"):
        droplet.shape.contains(
            np.zeros((1, 3)), square.geometry(), droplet.internal_values({"theta": 140.0, "delta": 0.0, "phi": 0.05})
        )
