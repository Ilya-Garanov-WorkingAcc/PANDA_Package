"""Профиль плотности по траектории: центрирование, бины, блоки, ориентация, проверки, сохранение."""

import warnings
from pathlib import Path

import numpy as np
import pytest

from panda_md import SlitPore, TrianglePore
from panda_md.core.enums import Frame, Orientation, ProfileCentering
from panda_md.core.errors import InvalidTrajectoryError, NotDerivedError, PandaWarning
from panda_md.core.profile import DensityProfile, ProfileSet
from panda_md.trajectory.arrays import ArrayTrajectory
from panda_md.trajectory.density import density_profile

WIDTH_NM = 4.0
LENGTH_NM = 6.0
ASPECT_RATIO = 0.5
BOX_NM = np.array([LENGTH_NM, ASPECT_RATIO * LENGTH_NM, 5.0])
LOWER_WALL_NM = 3.5  # стенки при z = 3.5 и z = 7.5 ≡ 2.5 (подложка — 1 нм): пора пересекает границу ящика
PORE_CENTER_NM = LOWER_WALL_NM + WIDTH_NM / 2.0
TIME_STEP_PS = 10.0
SEED = 7


def walls(frame_shift_nm: float = 0.0) -> np.ndarray:
    grid = np.array([[x, y] for x in np.arange(0.0, LENGTH_NM, 0.5) for y in np.arange(0.0, BOX_NM[1], 0.5)])
    lower = np.column_stack([grid, np.full(len(grid), LOWER_WALL_NM + frame_shift_nm)])
    upper = np.column_stack([grid, np.full(len(grid), np.mod(LOWER_WALL_NM + WIDTH_NM + frame_shift_nm, BOX_NM[2]))])
    return np.vstack([lower, upper])


def trajectory_from(
    liquid_z_per_frame: list[np.ndarray], frame_shifts_nm: list[float] | None = None
) -> ArrayTrajectory:
    """Жидкость — заданные z (в системе поры), x и y — равномерно; стенки — двойная решётка."""
    generator = np.random.default_rng(SEED)
    shifts = frame_shifts_nm or [0.0] * len(liquid_z_per_frame)
    frames = []
    for liquid_z, shift in zip(liquid_z_per_frame, shifts, strict=True):
        lateral = generator.uniform(0.0, 1.0, size=(liquid_z.size, 2)) * BOX_NM[:2]
        liquid = np.column_stack([lateral, np.mod(liquid_z + PORE_CENTER_NM + shift, BOX_NM[2])])
        frames.append(np.vstack([walls(shift), liquid]))
    wall_count = walls().shape[0]
    frame_count = len(frames)
    return ArrayTrajectory(
        positions_nm=np.array(frames),
        boxes_nm=np.tile(BOX_NM, (frame_count, 1)),
        times_ps=TIME_STEP_PS * np.arange(frame_count),
        groups={"walls": np.arange(wall_count), "liquid": wall_count + np.arange(liquid_z_per_frame[0].size)},
    )


def pore() -> SlitPore:
    return SlitPore(width_nm=WIDTH_NM, length_nm=LENGTH_NM, base_aspect_ratio=ASPECT_RATIO, wall_selection="walls")


def uniform_slab(lower_nm: float, upper_nm: float, count: int, frames: int) -> list[np.ndarray]:
    generator = np.random.default_rng(SEED)
    return [generator.uniform(lower_nm, upper_nm, count) for _ in range(frames)]


def test_uniform_slab_gives_flat_density_aligned_to_walls() -> None:
    count = 20000
    trajectory = trajectory_from(uniform_slab(-1.0, 1.0, count, 4))
    profiles = density_profile(trajectory, pore(), "liquid", bins=40)
    assert profiles.frame is Frame.PORE
    np.testing.assert_allclose(profiles.bin_edges_nm[[0, -1]], [-WIDTH_NM / 2, WIDTH_NM / 2])
    mean = profiles.mean()
    expected = count / (LENGTH_NM * ASPECT_RATIO * LENGTH_NM * 2.0)
    inside = np.abs(mean.coordinate_nm) < 1.0
    assert np.mean(mean.density_per_nm3[inside]) == pytest.approx(expected, rel=0.01)
    assert np.all(mean.density_per_nm3[np.abs(mean.coordinate_nm) > 1.05] == 0.0)
    assert mean.particle_count == count
    assert mean.pore_volume_nm3 == pytest.approx(pore().volume_nm3())


def test_walls_found_in_every_frame_follow_a_drifting_pore() -> None:
    slabs = uniform_slab(-1.9, -1.0, 5000, 3)  # слой у нижней стенки
    shifts = [0.0, 0.3, 0.6]
    by_walls = density_profile(trajectory_from(slabs, shifts), pore(), "liquid", bins=40, orientation=Orientation.KEEP)
    fixed = density_profile(
        trajectory_from(slabs), pore(), "liquid", bins=40, pore_center_nm=PORE_CENTER_NM, orientation=Orientation.KEEP
    )
    np.testing.assert_allclose(by_walls.mean().density_per_nm3, fixed.mean().density_per_nm3)


def test_liquid_centering_puts_center_of_mass_at_zero_across_boundary() -> None:
    trajectory = trajectory_from(uniform_slab(-0.5, 0.5, 4000, 2), [0.0, 0.7])
    profiles = density_profile(trajectory, pore(), "liquid", centering=ProfileCentering.LIQUID, bins=50)
    assert profiles.frame is Frame.LIQUID
    np.testing.assert_allclose(profiles.bin_edges_nm[[0, -1]], [-BOX_NM[2] / 2, BOX_NM[2] / 2])
    mean = profiles.mean()
    center_of_mass = np.sum(mean.coordinate_nm * mean.density_per_nm3) / np.sum(mean.density_per_nm3)
    assert center_of_mass == pytest.approx(0.0, abs=0.02)
    assert not profiles.flipped


def test_blocks_by_time_and_partial_block_is_dropped() -> None:
    trajectory = trajectory_from(uniform_slab(-1.0, 1.0, 500, 7))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # один кадр на границе последнего блока — не неполный блок
        profiles = density_profile(trajectory, pore(), "liquid", bins=20, block="20 ps")
    assert profiles.block_count == 3
    assert profiles.block_frame_counts == (2, 2, 2)
    np.testing.assert_allclose(profiles.block_start_ps, [0.0, 20.0, 40.0])
    np.testing.assert_allclose(profiles.block_stop_ps, [10.0, 30.0, 50.0])
    np.testing.assert_allclose(profiles.mean().density_per_nm3, profiles.block_densities_per_nm3.mean(axis=0))
    np.testing.assert_allclose(profiles.sem(), profiles.std() / np.sqrt(3))
    assert profiles.mean().density_sem_per_nm3 is not None
    window = density_profile(trajectory, pore(), "liquid", bins=20, start="0.02 ns", stop=40.0, stride=2)
    assert window.block_frame_counts == (2,)
    with pytest.raises(InvalidTrajectoryError, match="короче одного блока"):
        density_profile(trajectory, pore(), "liquid", bins=20, block="1 ns")


def test_liquid_at_upper_wall_is_flipped_in_pore_frame() -> None:
    trajectory = trajectory_from(uniform_slab(1.0, 1.9, 3000, 2))
    flipped = density_profile(trajectory, pore(), "liquid", bins=40)
    kept = density_profile(trajectory, pore(), "liquid", bins=40, orientation=Orientation.KEEP)
    assert flipped.flipped and not kept.flipped
    np.testing.assert_allclose(flipped.mean().density_per_nm3, kept.mean().density_per_nm3[::-1])
    assert flipped.mean().flipped


def test_bin_width_sets_bin_count() -> None:
    profiles = density_profile(trajectory_from(uniform_slab(-1.0, 1.0, 100, 1)), pore(), "liquid", bin_width_nm=0.1)
    assert profiles.block_densities_per_nm3.shape == (1, 40)


def test_problems_with_pore_and_selection_are_reported() -> None:
    trajectory = trajectory_from(uniform_slab(-1.0, 1.0, 100, 1))
    no_walls = SlitPore(width_nm=WIDTH_NM, length_nm=LENGTH_NM, base_aspect_ratio=ASPECT_RATIO)
    with pytest.raises(InvalidTrajectoryError, match="pore_center_nm"):
        density_profile(trajectory, no_walls, "liquid")
    too_wide = SlitPore(width_nm=12.0, length_nm=LENGTH_NM, base_aspect_ratio=ASPECT_RATIO)
    with pytest.raises(InvalidTrajectoryError, match="шире ящика"):
        density_profile(trajectory, too_wide, "liquid", pore_center_nm=PORE_CENTER_NM)
    triangle = TrianglePore(side_nm=4.0, length_nm=6.0, layer_width_nm=4.0, wall_selection="walls")
    with pytest.raises(NotDerivedError, match="pore_center_nm"):
        density_profile(trajectory, triangle, "liquid")
    with pytest.raises(InvalidTrajectoryError, match="пуста"):
        density_profile(trajectory, pore(), [])


def test_suspicious_geometry_gives_warnings() -> None:
    trajectory = trajectory_from(uniform_slab(-1.0, 3.0, 1000, 1))  # четверть частиц — за верхней стенкой
    with pytest.warns(PandaWarning, match="вне поры"):
        density_profile(trajectory, pore(), "liquid", bins=20)
    inside = trajectory_from(uniform_slab(-1.0, 1.0, 1000, 1))
    narrow = SlitPore(width_nm=3.5, length_nm=LENGTH_NM, base_aspect_ratio=ASPECT_RATIO, wall_selection="walls")
    with pytest.warns(PandaWarning, match="промежуток между стенками"):
        density_profile(inside, narrow, "liquid", bins=20)
    other_base = SlitPore(width_nm=WIDTH_NM, length_nm=LENGTH_NM, wall_selection="walls")
    with pytest.warns(PandaWarning, match="площадь бина"):
        density_profile(inside, other_base, "liquid", bins=20)


def test_profiles_survive_saving(tmp_path: Path) -> None:
    profiles = density_profile(trajectory_from(uniform_slab(-1.0, 1.0, 800, 4)), pore(), "liquid", block="20 ps")
    profiles.save(tmp_path / "set.npz")
    assert ProfileSet.load(tmp_path / "set.npz").model_dump_json() == profiles.model_dump_json()
    mean = profiles.mean()
    for name in ("mean.npz", "mean.csv"):
        mean.save(tmp_path / name)
        loaded = DensityProfile.load(tmp_path / name)
        np.testing.assert_allclose(loaded.density_per_nm3, mean.density_per_nm3, rtol=1e-12)
        assert loaded.model_dump(exclude={"coordinate_nm", "density_per_nm3", "density_sem_per_nm3"}) == (
            mean.model_dump(exclude={"coordinate_nm", "density_per_nm3", "density_sem_per_nm3"})
        )
    with pytest.raises(ValueError, match="формат"):
        mean.save(tmp_path / "mean.txt")


def test_centering_and_orientation_accept_strings() -> None:
    trajectory = trajectory_from(uniform_slab(1.0, 1.9, 500, 1))
    by_name = density_profile(trajectory, pore(), "liquid", centering="pore", orientation="keep", bins=20)
    by_enum = density_profile(
        trajectory, pore(), "liquid", centering=ProfileCentering.PORE, orientation=Orientation.KEEP, bins=20
    )
    np.testing.assert_allclose(by_name.block_densities_per_nm3, by_enum.block_densities_per_nm3)
    with pytest.raises(ValueError, match="none"):
        density_profile(trajectory, pore(), "liquid", centering="none")


def test_partial_block_of_several_frames_is_reported() -> None:
    trajectory = trajectory_from(uniform_slab(-1.0, 1.0, 200, 8))  # 0…70 пс, блоки по 30 пс: 60 и 70 пс — лишние
    with pytest.warns(PandaWarning, match="2 последних кадров"):
        profiles = density_profile(trajectory, pore(), "liquid", bins=20, block="30 ps")
    assert profiles.block_frame_counts == (3, 3)


def test_box_changing_along_profile_axis_is_reported_in_liquid_frame() -> None:
    generator = np.random.default_rng(SEED)
    lengths = [5.0, 5.5]
    frames = [
        np.column_stack([generator.uniform(0.0, 3.0, (400, 2)), generator.uniform(0.0, length, 400)])
        for length in lengths
    ]
    trajectory = ArrayTrajectory(
        positions_nm=np.array(frames),
        boxes_nm=np.array([[6.0, 3.0, length] for length in lengths]),
        times_ps=[0.0, 1.0],
    )
    with pytest.warns(PandaWarning, match="ящик вдоль оси профиля"):
        density_profile(trajectory, pore(), list(range(400)), centering=ProfileCentering.LIQUID, bins=20)


def test_walls_in_a_single_plane_give_the_whole_box_as_pore() -> None:
    grid = np.array([[x, y, 0.5] for x in np.arange(0.0, 6.0, 0.5) for y in np.arange(0.0, 3.0, 0.5)])
    liquid = np.column_stack([np.full(100, 1.0), np.full(100, 1.0), np.linspace(1.0, 4.0, 100)])
    trajectory = ArrayTrajectory(
        positions_nm=np.vstack([grid, liquid])[np.newaxis],
        boxes_nm=[[6.0, 3.0, 5.0]],
        times_ps=[0.0],
        groups={"walls": np.arange(len(grid)), "liquid": len(grid) + np.arange(100)},
    )
    plane = SlitPore.from_trajectory(trajectory, walls="walls")
    assert plane.width_nm == pytest.approx(5.0)
    profiles = density_profile(trajectory, plane, "liquid", bins=20)
    assert profiles.block_particle_counts[0] == 100


def test_warnings_point_at_the_caller() -> None:
    trajectory = trajectory_from(uniform_slab(-1.0, 1.0, 100, 1))
    other_base = SlitPore(width_nm=WIDTH_NM, length_nm=LENGTH_NM, wall_selection="walls")
    with pytest.warns(PandaWarning, match="площадь бина") as caught:
        density_profile(trajectory, other_base, "liquid", bins=20)
    assert caught[0].filename == __file__


def test_single_frame_with_blocks_is_reported() -> None:
    trajectory = trajectory_from(uniform_slab(-1.0, 1.0, 100, 1))
    with pytest.warns(PandaWarning, match="один кадр"):
        profiles = density_profile(trajectory, pore(), "liquid", bins=20, block="1 ns")
    assert profiles.block_count == 1


def test_profile_set_requires_npz(tmp_path: Path) -> None:
    profiles = density_profile(trajectory_from(uniform_slab(-1.0, 1.0, 100, 1)), pore(), "liquid", bins=20)
    with pytest.raises(ValueError, match=r"\.npz"):
        profiles.save(tmp_path / "set")
    with pytest.raises(ValueError, match=r"\.npz"):
        ProfileSet.load(tmp_path / "set.csv")
