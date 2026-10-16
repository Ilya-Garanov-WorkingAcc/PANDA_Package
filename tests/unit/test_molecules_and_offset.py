"""Профиль по центрам масс молекул и поправка положения поверхности стенки (wall_offset)."""

import warnings
from typing import Any

import numpy as np
import pytest

from panda_md import ArrayTrajectory, SlitPore, density_profile
from panda_md.core.enums import ParticleGrouping
from panda_md.core.errors import InvalidTrajectoryError
from panda_md.core.pbc import molecule_centers
from panda_md.core.trajectory import renumber
from panda_md.trajectory.reader import MDATrajectory

BOX_NM = np.array([4.0, 4.0, 10.0])


def test_renumber_keeps_order_of_first_appearance() -> None:
    assert renumber(np.array([7, 7, 3, 3, 9, 7])).tolist() == [0, 0, 1, 1, 2, 0]


def test_molecule_cut_by_box_boundary_has_its_center_at_the_boundary() -> None:
    positions = np.array([[1.0, 1.0, 9.9], [1.0, 1.0, 0.1], [2.0, 2.0, 5.0], [2.0, 2.0, 5.4]])
    centers = molecule_centers(positions, BOX_NM, np.array([0, 0, 1, 1]), np.array([1.0, 1.0, 16.0, 1.0]))
    assert min(centers[0, 2], BOX_NM[2] - centers[0, 2]) == pytest.approx(0.0, abs=1e-12)  # а не 5.0
    assert centers[1, 2] == pytest.approx((16.0 * 5.0 + 5.4) / 17.0)
    with pytest.raises(InvalidTrajectoryError, match="масса"):
        molecule_centers(positions, BOX_NM, np.array([0, 0, 1, 1]), np.zeros(4))


def water_like_trajectory() -> ArrayTrajectory:
    """Молекулы из трёх атомов (16 + 1 + 1) в слое |z| < 1 нм вокруг центра поры z = 5; подложка 7.5–12.5 ≡ 2.5."""
    generator = np.random.default_rng(5)
    count = 300
    centers = np.column_stack([generator.uniform(0.0, 4.0, (count, 2)), generator.uniform(4.0, 6.0, count)])
    offsets = np.array([[0.0, 0.0, 0.0], [0.08, 0.06, 0.0], [-0.08, 0.06, 0.0]])
    liquid = (centers[:, np.newaxis, :] + offsets).reshape(-1, 3)
    grid = np.array([[x, y] for x in np.arange(0.0, 4.0, 0.5) for y in np.arange(0.0, 4.0, 0.5)])
    walls = np.vstack([np.column_stack([grid, np.full(len(grid), z)]) for z in np.mod(np.arange(7.5, 12.6, 0.5), 10.0)])
    positions = np.vstack([walls, liquid])
    wall_count = len(walls)
    molecule_ids = np.concatenate([np.arange(wall_count) + 10_000, np.repeat(np.arange(count), 3)])
    masses = np.concatenate([np.full(wall_count, 12.0), np.tile([16.0, 1.0, 1.0], count)])
    return ArrayTrajectory(
        positions_nm=positions[np.newaxis],
        boxes_nm=BOX_NM[np.newaxis],
        times_ps=[0.0],
        groups={"walls": np.arange(wall_count), "water": wall_count + np.arange(3 * count)},
        molecule_ids=molecule_ids,
        masses=masses,
    )


def test_profile_per_molecule_counts_molecules() -> None:
    trajectory = water_like_trajectory()
    pore = SlitPore.from_trajectory(trajectory, walls="walls")
    by_atom = density_profile(trajectory, pore, "water", bins=50)
    by_molecule = density_profile(trajectory, pore, "water", bins=50, per=ParticleGrouping.MOLECULE)
    assert by_atom.block_particle_counts[0] == 900
    assert by_molecule.block_particle_counts[0] == 300
    total = np.sum(by_molecule.block_densities_per_nm3[0] * np.diff(by_molecule.bin_edges_nm)) * 16.0
    assert total == pytest.approx(300.0)
    by_name = density_profile(trajectory, pore, "water", bins=50, per="molecule")
    np.testing.assert_allclose(by_name.block_densities_per_nm3, by_molecule.block_densities_per_nm3)


def test_molecules_need_ids_and_masses() -> None:
    plain = ArrayTrajectory(positions_nm=np.zeros((1, 2, 3)) + 1.0, boxes_nm=BOX_NM[np.newaxis], times_ps=[0.0])
    pore = SlitPore(width_nm=5.0, length_nm=4.0)
    with pytest.raises(InvalidTrajectoryError, match="molecule_ids"):
        density_profile(plain, pore, [0, 1], pore_center_nm=5.0, per="molecule")


def test_wall_offset_moves_surfaces_inward() -> None:
    trajectory = water_like_trajectory()
    bare = SlitPore.from_trajectory(trajectory, walls="walls")
    offset = SlitPore.from_trajectory(trajectory, walls="walls", wall_offset_nm=0.17)
    assert bare.width_nm == pytest.approx(5.0)
    assert offset.width_nm == pytest.approx(5.0 - 2 * 0.17)
    assert offset.wall_offset_nm == pytest.approx(0.17)
    assert offset.expected_wall_gap_nm() == pytest.approx(5.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # промежуток стенок = H + 2 · offset — без предупреждения
        profiles = density_profile(trajectory, offset, "water", bins=50)
    np.testing.assert_allclose(profiles.bin_edges_nm[[0, -1]], [-(5.0 - 0.34) / 2, (5.0 - 0.34) / 2])
    with pytest.raises(InvalidTrajectoryError, match="wall_offset_nm"):
        SlitPore.from_trajectory(trajectory, walls="walls", wall_offset_nm=3.0)


def test_mdanalysis_molecules_are_residues() -> None:
    mdanalysis = pytest.importorskip("MDAnalysis")
    universe: Any = mdanalysis.Universe.empty(4, n_residues=2, atom_resindex=[0, 0, 1, 1], trajectory=True)
    universe.add_TopologyAttr("masses", [16.0, 1.0, 12.0, 12.0])
    trajectory = MDATrajectory(universe)
    molecule_indices, masses = trajectory.molecules(np.array([1, 2, 3]))
    assert molecule_indices.tolist() == [0, 1, 1]
    np.testing.assert_allclose(masses, [1.0, 12.0, 12.0])
