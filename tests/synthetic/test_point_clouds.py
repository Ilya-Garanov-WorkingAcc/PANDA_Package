"""Облака точек: форма в щели → траектория с блуждающей формой → стенки, центрирование, профиль, фит.

Критерий спецификации (§9): «облако точек → полный путь данных → фит» — |Δθ| < 2°.
"""

import pytest

from panda_md import SlitPore, SquarePore, density_profile
from panda_md.core.errors import NotDerivedError
from panda_md.synthetic.points import synthetic_trajectory

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


def test_point_clouds_exist_only_for_slit_shapes() -> None:
    square = SquarePore(side_nm=6.0, length_nm=12.0, layer_width_nm=6.0)
    with pytest.raises(NotDerivedError, match="щели"):
        synthetic_trajectory(
            square.interface("droplet_w1c0"), {"theta": 140.0, "delta": 0.0, "phi": 0.05}, density_bulk_per_nm3=30.0
        )
