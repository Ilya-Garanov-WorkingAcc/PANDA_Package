"""Frame sources (numpy arrays and MDAnalysis) and a slit built from the wall atoms."""

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from panda_md import SlitPore
from panda_md.core.enums import Axis
from panda_md.core.errors import InvalidTrajectoryError, MissingDependencyError, PandaWarning
from panda_md.trajectory.arrays import ArrayTrajectory
from panda_md.trajectory.reader import MDATrajectory, load_trajectory

BOX_NM = np.array([6.0, 4.0, 10.0])
WALL_LAYERS_NM = (
    0.2,
    0.55,
    0.9,
)  # substrate at the bottom of the box; its image at the top bounds the pore through PBC
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
    with pytest.raises(ValidationError, match="boxes have shape"):
        ArrayTrajectory(positions_nm=np.zeros((2, 3, 3)), boxes_nm=np.ones((3, 3)), times_ps=[0.0, 1.0])
    with pytest.raises(ValidationError, match="invalid times"):
        ArrayTrajectory(positions_nm=np.zeros((2, 3, 3)), boxes_nm=np.ones((2, 3)), times_ps=[1.0, 0.0])
    with pytest.raises(ValidationError, match="group 'g' has index"):
        ArrayTrajectory(positions_nm=np.zeros((1, 3, 3)), boxes_nm=np.ones((1, 3)), times_ps=[0.0], groups={"g": [5]})


def test_selection_by_group_name_or_indices() -> None:
    trajectory = slit_trajectory()
    assert trajectory.select("liquid").tolist() == [72, 73, 74]
    assert trajectory.select([0, 1]).tolist() == [0, 1]
    with pytest.raises(InvalidTrajectoryError, match="no group"):
        trajectory.select("water")
    with pytest.raises(InvalidTrajectoryError, match="outside"):
        trajectory.select([1000])


def test_frames_are_chosen_by_time_and_stride() -> None:
    trajectory = slit_trajectory()
    assert trajectory.frame_indices() == [0, 1, 2, 3, 4]
    assert trajectory.frame_indices(start="0.002 ns", stop=6.0) == [1, 2, 3]
    assert trajectory.frame_indices(start=0.0, stride=2) == [0, 2, 4]
    assert trajectory.frame_at(3.0, [trajectory.select("liquid")]).index == 2
    with pytest.raises(InvalidTrajectoryError, match="no frames"):
        trajectory.frame_indices(start="1 ns")
    with pytest.raises(ValueError, match="stride"):
        trajectory.frame_indices(stride=0)


def test_slit_from_trajectory_finds_walls_through_box_boundary() -> None:
    pore = SlitPore.from_trajectory(slit_trajectory(), walls="walls")
    assert pore.width_nm == pytest.approx(BOX_NM[2] + WALL_LAYERS_NM[0] - WALL_LAYERS_NM[-1])
    assert pore.length_nm == pytest.approx(6.0)
    assert pore.base_aspect_ratio == pytest.approx(4.0 / 6.0)
    assert pore.wall_selection == "walls"
    assert pore.locate_center_nm(wall_atoms()[:, 2], BOX_NM[2]) == pytest.approx(
        (WALL_LAYERS_NM[-1] + BOX_NM[2] + WALL_LAYERS_NM[0]) / 2.0
    )


def test_slit_from_trajectory_rejects_wrong_axes() -> None:
    with pytest.raises(InvalidTrajectoryError, match="length_axis='x'"):
        SlitPore.from_trajectory(slit_trajectory(), walls="walls", length_axis=Axis.Y)
    with pytest.raises(InvalidTrajectoryError, match="coincides"):
        SlitPore.from_trajectory(slit_trajectory(), walls="walls", length_axis=Axis.Z)


def memory_universe(dimensions: list[float]) -> Any:  # noqa: ANN401 — MDAnalysis has no type annotations
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
    with pytest.raises(InvalidTrajectoryError, match="is empty"):
        trajectory.select("resname DECAN")


def test_mdanalysis_trajectory_rejects_triclinic_box() -> None:
    trajectory = MDATrajectory(memory_universe([50.0, 40.0, 100.0, 90.0, 90.0, 60.0]))
    with pytest.raises(InvalidTrajectoryError, match="not rectangular"):
        list(trajectory.frames([trajectory.select("name OW")]))


def test_load_trajectory_reports_missing_files_and_package(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("MDAnalysis")
    with pytest.raises(InvalidTrajectoryError, match="file not found"):
        load_trajectory("missing.gro", "missing.xtc")
    monkeypatch.setitem(__import__("sys").modules, "MDAnalysis", None)
    with pytest.raises(MissingDependencyError, match=r"panda-md\[md\]"):
        load_trajectory("missing.gro")


def vacuum_trajectory() -> ArrayTrajectory:
    """Two substrates (z ≈ 1–1.6 and 6–6.6 nm), liquid between them, vacuum above the second one up to Lz = 20 nm."""
    grid = np.array([[x, y] for x in np.arange(0.0, 6.0, 1.0) for y in np.arange(0.0, 4.0, 1.0)])
    walls = np.vstack([np.column_stack([grid, np.full(len(grid), z)]) for z in (1.0, 1.3, 1.6, 6.0, 6.3, 6.6)])
    liquid = np.column_stack([np.full(50, 2.0), np.full(50, 2.0), np.linspace(2.0, 5.5, 50)])
    return ArrayTrajectory(
        positions_nm=np.vstack([walls, liquid])[np.newaxis],
        boxes_nm=[[6.0, 4.0, 20.0]],
        times_ps=[0.0],
        groups={"walls": np.arange(len(walls)), "liquid": len(walls) + np.arange(50)},
    )


def test_vacuum_gap_is_reported_and_liquid_selects_the_pore() -> None:
    with pytest.warns(PandaWarning, match="liquid="):
        vacuum = SlitPore.from_trajectory(vacuum_trajectory(), walls="walls")
    assert vacuum.width_nm == pytest.approx(14.4)
    pore = SlitPore.from_trajectory(vacuum_trajectory(), walls="walls", liquid="liquid")
    assert pore.width_nm == pytest.approx(4.4)
    walls_z = vacuum_trajectory().positions_nm[0, : 6 * 24, 2]
    assert pore.locate_center_nm(walls_z, 20.0) == pytest.approx(3.8)  # the gap of width H, not the vacuum


def test_mdanalysis_selection_by_indices_is_range_checked() -> None:
    trajectory = MDATrajectory(memory_universe([50.0, 40.0, 100.0, 90.0, 90.0, 90.0]))
    assert trajectory.select([0, 3]).tolist() == [0, 3]
    with pytest.raises(InvalidTrajectoryError, match="outside"):
        trajectory.select([4])


class _Timestep:
    def __init__(self, frame: int, time: float) -> None:
        self.frame = frame
        self.time = time
        self.dimensions = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
        self.positions = np.full((2, 3), float(frame))


class _Reader:
    """A joined trajectory with a break: step 2 ps, then a jump in time."""

    dt = 2.0

    def __init__(self, times: list[float]) -> None:
        self.steps = [_Timestep(index, time) for index, time in enumerate(times)]

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, key: int | list[int]) -> Any:  # noqa: ANN401 — as in MDAnalysis: a frame or an iterator
        if isinstance(key, list):
            return iter([self.steps[index] for index in key])
        return self.steps[key]

    def __iter__(self) -> Any:  # noqa: ANN401
        return iter(self.steps)


class _Universe:
    def __init__(self, times: list[float]) -> None:
        self.trajectory = _Reader(times)


def test_mdanalysis_times_follow_real_frame_times_of_joined_trajectories() -> None:
    trajectory = MDATrajectory(_Universe([0.0, 2.0, 4.0, 100.0, 102.0]))
    with pytest.warns(PandaWarning, match="step"):
        times = trajectory.times_ps()
    np.testing.assert_allclose(times, [0.0, 2.0, 4.0, 100.0, 102.0])
    assert trajectory.frame_indices(start=50.0) == [3, 4]
    regular = MDATrajectory(_Universe([0.0, 2.0, 4.0]))
    np.testing.assert_allclose(regular.times_ps(), [0.0, 2.0, 4.0])


def test_overlapping_times_are_an_error() -> None:
    base = slit_trajectory()
    with pytest.warns(PandaWarning, match="frame times"):
        trajectory = MDATrajectory(_Universe([0.0, 2.0, 4.0, 2.0, 4.0]))
        times = trajectory.times_ps()
    assert times.tolist() == [0.0, 2.0, 4.0, 2.0, 4.0]
    with pytest.raises(InvalidTrajectoryError, match="trjcat"):
        trajectory.frame_indices()
    assert base.frame_indices() == [0, 1, 2, 3, 4]
