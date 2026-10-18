"""Real MD: profiles computed once by the data path of panda_md from three trajectories (not in the repository).

The expectations are independent: a manual wall calculation and the previous PANDA on the same trajectories.
- graphite: water is a non-wetting bridge (roll), θ ≈ 172° ± 7° over blocks; the 2D map shows water separated
  from graphite by a decane film, hence θ is close to 180°;
- benzene: benzene wets the SOLD substrate (density near the walls is higher than in the middle) — θ < 90°, there
  are no wetting regime formulas, the fit must hit the regime boundary and raise the flag;
- calcite: the starting frame, the substrate is thick and made of different atoms — only the wall search and
  that all the liquid is in the pore are checked.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import trapezoid

from panda_md import ArrayTrajectory, Frame, ProfileSet, SlitPore, density_profile

DATA = Path(__file__).parent / "data"
WATER_DENSITY_BULK = 33.13  # ρ_bulk of water oxygens in the graphite system: 2D map median in the phase core (task 1)


def load_pore(name: str) -> SlitPore:
    return SlitPore.model_validate_json((DATA / f"{name}_pore.json").read_text())


HAND_WIDTHS_NM = {  # H from the outermost substrate atoms and the box length from .gro (task 1 and manual calculation)
    "graphite": 11.0 + 0.195 - 2.926,  # image of the graphite bottom − its top
    "benzene": 13.4607 - 3.147,  # task 1: H = Lz − top of SOLD = 10.31
    "calcite": 12.684 + 0.009 - 3.554,
}


def frame_trajectory(name: str) -> ArrayTrajectory:
    """One real frame: substrate and liquid with real heights (x, y are not needed for the z axis — zeros)."""
    with np.load(DATA / f"{name}_frame.npz") as frame:
        walls, liquid, box = (
            frame["wall_z_nm"].astype(np.float64),
            frame["liquid_z_nm"].astype(np.float64),
            frame["box_nm"],
        )
    heights = np.concatenate([walls, liquid])
    positions = np.column_stack([np.zeros((heights.size, 2)), heights])
    return ArrayTrajectory(
        positions_nm=positions[np.newaxis],
        boxes_nm=box[np.newaxis],
        times_ps=[0.0],
        groups={"walls": np.arange(walls.size), "liquid": walls.size + np.arange(liquid.size)},
    )


@pytest.mark.parametrize("name", list(HAND_WIDTHS_NM))
def test_walls_and_profile_recomputed_from_a_real_frame(name: str) -> None:
    """The wall search and profile code on real substrate geometry — not only the saved results."""
    trajectory = frame_trajectory(name)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # neither vacuum nor particles outside the pore
        pore = SlitPore.from_trajectory(trajectory, walls="walls")
        stored = load_pore(name)
        assert pore.width_nm == pytest.approx(HAND_WIDTHS_NM[name], abs=1e-3)
        assert pore.width_nm == pytest.approx(stored.width_nm, abs=1e-3)
        sized = stored.model_copy(update={"wall_selection": "walls"})
        profiles = density_profile(trajectory, sized, "liquid", bins=100)
    assert profiles.block_particle_counts[0] == trajectory.select("liquid").size


def test_walls_are_found_as_computed_by_hand() -> None:
    """H is the largest gap between substrate atoms across PBC (outermost atoms and box lengths from .gro)."""
    graphite = load_pore("graphite")
    assert graphite.width_nm == pytest.approx(11.0 + 0.195 - 2.926, abs=1e-3)  # image of the graphite bottom − its top
    assert graphite.base_aspect_ratio == pytest.approx(5.12817 / 17.27104, abs=1e-5)
    benzene = load_pore("benzene")
    assert benzene.width_nm == pytest.approx(13.4607 - 3.147, abs=1e-3)  # task 1: H = Lz − top of SOLD = 10.31
    calcite = load_pore("calcite")
    assert calcite.width_nm == pytest.approx(12.684 + 0.009 - 3.554, abs=1e-3)


@pytest.mark.parametrize(
    ("file_name", "particles"),
    [
        ("graphite_water_profiles.npz", 11807),
        ("benzene_profiles.npz", 16584),
        ("calcite_decane_profiles.npz", 13400),
        ("calcite_water_profiles.npz", 14458),
    ],
)
def test_whole_liquid_is_inside_the_pore(file_name: str, particles: int) -> None:
    profiles = ProfileSet.load(DATA / file_name)
    assert profiles.frame is Frame.PORE
    np.testing.assert_allclose(profiles.block_particle_counts, particles)


def test_graphite_water_is_a_non_wetting_roll_close_to_180_degrees() -> None:
    pore = load_pore("graphite")
    profiles = ProfileSet.load(DATA / "graphite_water_profiles.npz")
    assert profiles.block_frame_counts == (500,) * 8
    mean = profiles.mean()
    phi = trapezoid(mean.density_per_nm3, mean.coordinate_nm) / (WATER_DENSITY_BULK * pore.width_nm)
    assert phi == pytest.approx(0.487, abs=0.005)  # task 1: φ = 0.487
    roll = pore.interface("roll")
    result = roll.fit(mean, density_bulk_per_nm3=WATER_DENSITY_BULK)
    assert 165.0 < result.values["theta"] <= 180.0
    assert 0.02 < result.values["delta"] < 0.1
    assert result.loss < 0.03
    block_angles = [
        roll.fit(profiles.block(index), density_bulk_per_nm3=WATER_DENSITY_BULK).values["theta"] for index in range(8)
    ]
    assert 165.0 < float(np.mean(block_angles)) < 178.0
    assert float(np.std(block_angles)) < 10.0


@pytest.mark.parametrize("density_bulk", [10.0, 15.0, 20.0, 25.0])
def test_benzene_wets_its_substrate_and_fit_flags_the_regime_boundary(density_bulk: float) -> None:
    """ρ_bulk of benzene in this system is unknown, so the conclusion is checked over the whole reasonable range."""
    pore = load_pore("benzene")
    profiles = ProfileSet.load(DATA / "benzene_profiles.npz")
    mean = profiles.mean()
    near_walls = mean.density_per_nm3[np.abs(mean.coordinate_nm) > 0.4 * pore.width_nm]
    middle = mean.density_per_nm3[np.abs(mean.coordinate_nm) < 0.1 * pore.width_nm]
    assert near_walls[near_walls > 0].mean() > 1.3 * middle.mean()  # concave bridge: wider at the walls
    result = pore.interface("roll").fit(mean, density_bulk_per_nm3=density_bulk)
    assert result.at_regime_boundary
    assert result.values["theta"] == pytest.approx(90.5, abs=0.1)
