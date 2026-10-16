"""Облака точек: форма в щели → траектория с блуждающей формой → стенки, центрирование, профиль, фит и 2D-карты.

Критерий спецификации (§9): «облако точек → полный путь данных → фит» — |Δθ| < 2°.
"""

from pathlib import Path

import numpy as np
import pytest

from panda_md import SlitPore, SquarePore, density_profile
from panda_md.core.errors import InvalidTrajectoryError, NotDerivedError
from panda_md.synthetic.points import synthetic_trajectory
from panda_md.trajectory.maps import DensityMap, density_map

DENSITY_BULK = 33.0
PORE = SlitPore(width_nm=4.0, length_nm=8.0, base_aspect_ratio=0.5)
MAX_ANGLE_ERROR_DEG = 2.0


@pytest.mark.parametrize(
    ("shape_name", "truth"),
    [
        ("roll", {"theta": 120.0, "delta": 0.03, "phi": 0.4}),
        ("droplet", {"theta": 130.0, "delta": 0.03, "phi": 0.05}),
        ("doughnut", {"theta": 130.0, "delta": 0.03, "phi": 0.2}),
    ],
)
def test_angle_recovered_from_point_clouds(shape_name: str, truth: dict[str, float]) -> None:
    model = PORE.interface(shape_name)
    trajectory = synthetic_trajectory(model, truth, density_bulk_per_nm3=DENSITY_BULK, frame_count=100, seed=1)
    found = SlitPore.from_trajectory(trajectory, walls="walls")
    assert found.width_nm == pytest.approx(PORE.width_nm, abs=1e-9)
    assert found.base_aspect_ratio == pytest.approx(PORE.base_aspect_ratio)
    profiles = density_profile(trajectory, found, "liquid", bins=100)
    result = found.interface(shape_name).fit(profiles.mean(), density_bulk_per_nm3=DENSITY_BULK)
    assert result.values["theta"] == pytest.approx(truth["theta"], abs=MAX_ANGLE_ERROR_DEG)
    assert result.values["phi"] == pytest.approx(truth["phi"], rel=0.03)


def mirror_agreement(density: np.ndarray) -> float:
    """Корреляция карты с её отражением x → −x: 1 — форма симметрична относительно оси в x = 0."""
    return float(np.corrcoef(density.ravel(), density[::-1].ravel())[0, 1])


def test_axis_centering_sharpens_the_map_of_a_wandering_roll() -> None:
    roll = PORE.interface("roll")
    trajectory = synthetic_trajectory(
        roll, {"theta": 120.0, "delta": 0.03, "phi": 0.3}, density_bulk_per_nm3=DENSITY_BULK, frame_count=20, seed=2
    )
    found = SlitPore.from_trajectory(trajectory, walls="walls")
    centered = density_map(trajectory, found, "liquid", shape="roll", bins=(40, 40))
    smeared = density_map(trajectory, found, "liquid", bins=(40, 40))
    assert centered.axis_centered and not smeared.axis_centered
    assert mirror_agreement(centered.density_per_nm3) > 0.95
    middle = centered.density_per_nm3[:, 20]
    assert middle[20] == pytest.approx(DENSITY_BULK, rel=0.15)  # в сердцевине валика — объёмная плотность
    assert middle[0] == pytest.approx(0.0, abs=1.0)  # на краю ячейки (за валиком) жидкости нет
    assert np.max(smeared.density_per_nm3) < 0.8 * np.max(centered.density_per_nm3)


def region_mean(density_map_: DensityMap, length_window: tuple[float, float], height_limit: float) -> float:
    """Средняя плотность карты в полосе |z| < height_limit и length_window[0] ≤ |x| < length_window[1] (нм)."""
    along = 0.5 * (density_map_.length_edges_nm[:-1] + density_map_.length_edges_nm[1:])
    height = 0.5 * (density_map_.height_edges_nm[:-1] + density_map_.height_edges_nm[1:])
    rows = (np.abs(along) >= length_window[0]) & (np.abs(along) < length_window[1])
    columns = np.abs(height) < height_limit
    return float(np.mean(density_map_.density_per_nm3[np.ix_(rows, columns)]))


def test_perforation_axis_is_the_center_of_the_hole() -> None:
    perforation = PORE.interface("perforation")
    trajectory = synthetic_trajectory(
        perforation,
        {"theta": 130.0, "delta": 0.03, "phi": 0.8},
        density_bulk_per_nm3=DENSITY_BULK,
        frame_count=50,
        seed=3,
    )
    found = SlitPore.from_trajectory(trajectory, walls="walls")
    centered = density_map(trajectory, found, "liquid", shape="perforation", bins=(40, 40))
    hole = region_mean(centered, (0.0, 0.5), 1.0)
    bulk = region_mean(centered, (3.0, 4.0), 1.0)
    assert bulk == pytest.approx(DENSITY_BULK, rel=0.1)
    assert hole < 0.6 * bulk


def test_map_needs_a_shape_with_an_axis_and_survives_saving(tmp_path: Path) -> None:
    roll = PORE.interface("roll")
    trajectory = synthetic_trajectory(
        roll, {"theta": 120.0, "delta": 0.03, "phi": 0.3}, density_bulk_per_nm3=DENSITY_BULK, frame_count=2, seed=4
    )
    found = SlitPore.from_trajectory(trajectory, walls="walls")
    with pytest.raises(InvalidTrajectoryError, match="оси симметрии"):
        density_map(trajectory, found, "liquid", shape="droplet")
    density = density_map(trajectory, found, "liquid", shape="roll", bins=(10, 12))
    density.save(tmp_path / "map.npz")
    assert DensityMap.load(tmp_path / "map.npz").model_dump_json() == density.model_dump_json()


def test_point_clouds_exist_only_for_slit_shapes() -> None:
    square = SquarePore(side_nm=6.0, length_nm=12.0, layer_width_nm=6.0)
    with pytest.raises(NotDerivedError, match="щели"):
        synthetic_trajectory(
            square.interface("droplet_w1c0"), {"theta": 140.0, "delta": 0.0, "phi": 0.05}, density_bulk_per_nm3=30.0
        )
