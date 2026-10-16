"""Источники кадров: массивы numpy и MDAnalysis."""

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from panda_md.core.errors import InvalidTrajectoryError, MissingDependencyError
from panda_md.trajectory.arrays import ArrayTrajectory
from panda_md.trajectory.reader import MDATrajectory, load_trajectory

BOX_NM = np.array([6.0, 4.0, 10.0])
WALL_LAYERS_NM = (0.2, 0.55, 0.9)  # подложка у дна ящика; её образ сверху — граница поры через PBC
FRAME_COUNT = 5
TIME_STEP_PS = 2.0


def wall_atoms() -> np.ndarray:
    grid = np.array([[x, y] for x in np.arange(0.0, 6.0, 1.0) for y in np.arange(0.0, 4.0, 1.0)])
    return np.vstack([np.column_stack([grid, np.full(len(grid), z)]) for z in WALL_LAYERS_NM])


def slit_trajectory() -> ArrayTrajectory:
    walls = wall_atoms()
    liquid = np.column_stack([np.full(3, 1.0), np.full(3, 2.0), [3.0, 5.0, 7.0]])
    frame = np.vstack([walls, liquid])
    positions = np.repeat(frame[np.newaxis], FRAME_COUNT, axis=0)
    return ArrayTrajectory(
        positions_nm=positions,
        boxes_nm=np.tile(BOX_NM, (FRAME_COUNT, 1)),
        times_ps=TIME_STEP_PS * np.arange(FRAME_COUNT),
        groups={"walls": np.arange(len(walls)), "liquid": len(walls) + np.arange(3)},
    )


def test_array_trajectory_checks_shapes_and_order() -> None:
    with pytest.raises(ValidationError, match="ящики"):
        ArrayTrajectory(positions_nm=np.zeros((2, 3, 3)), boxes_nm=np.ones((3, 3)), times_ps=[0.0, 1.0])
    with pytest.raises(ValidationError, match="времена"):
        ArrayTrajectory(positions_nm=np.zeros((2, 3, 3)), boxes_nm=np.ones((2, 3)), times_ps=[1.0, 0.0])
    with pytest.raises(ValidationError, match="группа"):
        ArrayTrajectory(positions_nm=np.zeros((1, 3, 3)), boxes_nm=np.ones((1, 3)), times_ps=[0.0], groups={"g": [5]})


def test_selection_by_group_name_or_indices() -> None:
    trajectory = slit_trajectory()
    assert trajectory.select("liquid").tolist() == [72, 73, 74]
    assert trajectory.select([0, 1]).tolist() == [0, 1]
    with pytest.raises(InvalidTrajectoryError, match="группы"):
        trajectory.select("water")
    with pytest.raises(InvalidTrajectoryError, match="вне"):
        trajectory.select([1000])


def test_frames_are_chosen_by_time_and_stride() -> None:
    trajectory = slit_trajectory()
    assert trajectory.frame_indices() == [0, 1, 2, 3, 4]
    assert trajectory.frame_indices(start="0.002 ns", stop=6.0) == [1, 2, 3]
    assert trajectory.frame_indices(start=0.0, stride=2) == [0, 2, 4]
    assert trajectory.frame_at(3.0, [trajectory.select("liquid")]).index == 2
    with pytest.raises(InvalidTrajectoryError, match="нет кадров"):
        trajectory.frame_indices(start="1 ns")
    with pytest.raises(ValueError, match="stride"):
        trajectory.frame_indices(stride=0)


def memory_universe(dimensions: list[float]) -> Any:  # noqa: ANN401 — MDAnalysis без аннотаций типов
    mdanalysis = pytest.importorskip("MDAnalysis")
    from MDAnalysis.coordinates.memory import MemoryReader

    universe = mdanalysis.Universe.empty(4, n_residues=2, atom_resindex=[0, 0, 1, 1], trajectory=True)
    universe.add_TopologyAttr("names", ["C1", "C2", "OW", "HW"])
    universe.add_TopologyAttr("resnames", ["WALL", "WATER"])
    coordinates = np.arange(3 * 4 * 3, dtype=np.float32).reshape(3, 4, 3)  # Å
    universe.load_new(coordinates, format=MemoryReader, dimensions=dimensions, dt=2.0)
    return universe


def test_mdanalysis_trajectory_converts_angstrom_to_nm() -> None:
    trajectory = MDATrajectory(memory_universe([50.0, 40.0, 100.0, 90.0, 90.0, 90.0]))
    oxygen = trajectory.select("resname WATER and name OW")
    assert oxygen.tolist() == [2]
    np.testing.assert_allclose(trajectory.times_ps(), [0.0, 2.0, 4.0])
    frames = list(trajectory.frames([oxygen], start=2.0))
    assert [frame.index for frame in frames] == [1, 2]
    np.testing.assert_allclose(frames[0].positions_nm[0], [[1.8, 1.9, 2.0]])
    np.testing.assert_allclose(frames[0].box_nm, [5.0, 4.0, 10.0])
    with pytest.raises(InvalidTrajectoryError, match="пуста"):
        trajectory.select("resname DECAN")


def test_mdanalysis_trajectory_rejects_triclinic_box() -> None:
    trajectory = MDATrajectory(memory_universe([50.0, 40.0, 100.0, 90.0, 90.0, 60.0]))
    with pytest.raises(InvalidTrajectoryError, match="не прямоугольный"):
        list(trajectory.frames([trajectory.select("name OW")]))


def test_load_trajectory_reports_missing_files_and_package(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("MDAnalysis")
    with pytest.raises(InvalidTrajectoryError, match="не найден"):
        load_trajectory("нет.gro", "нет.xtc")
    monkeypatch.setitem(__import__("sys").modules, "MDAnalysis", None)
    with pytest.raises(MissingDependencyError, match=r"panda-md\[md\]"):
        load_trajectory("нет.gro")
