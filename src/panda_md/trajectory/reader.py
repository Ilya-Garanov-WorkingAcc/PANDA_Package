"""Чтение файлов МД через MDAnalysis (необязательная зависимость: `pip install panda-md[md]`).

MDAnalysis хранит длины в ångström; наружу отдаются нм. Единственный модуль пакета, импортирующий MDAnalysis.
"""

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from panda_md.core.errors import InvalidTrajectoryError, MissingDependencyError
from panda_md.core.trajectory import Trajectory, TrajectoryFrame, check_box
from panda_md.core.types import FloatArray, IndexArray

ANGSTROM_PER_NM = 10.0
RECTANGULAR_ANGLE_DEG = 90.0
ANGLE_TOLERANCE_DEG = 1e-3


def _import_mdanalysis() -> Any:  # noqa: ANN401 — у MDAnalysis нет аннотаций типов
    try:
        import MDAnalysis
    except ImportError as error:
        raise MissingDependencyError("для чтения файлов МД нужен MDAnalysis: pip install 'panda-md[md]'") from error
    return MDAnalysis


class MDATrajectory(Trajectory):
    """Траектория из объекта MDAnalysis.Universe."""

    universe: Any  # Any: MDAnalysis.Universe без аннотаций типов

    def __init__(self, universe: Any) -> None:  # noqa: ANN401 — MDAnalysis.Universe без аннотаций типов
        self.universe = universe

    def __repr__(self) -> str:
        return f"MDATrajectory({self.universe.atoms.n_atoms} атомов, {len(self.universe.trajectory)} кадров)"

    def select(self, selection: str | Sequence[int]) -> IndexArray:
        if isinstance(selection, str):
            try:
                indices = np.asarray(self.universe.select_atoms(selection).indices, dtype=np.int64)
            except Exception as error:  # MDAnalysis бросает разные исключения на синтаксис выборки
                raise InvalidTrajectoryError(f"выборка {selection!r} не разобрана MDAnalysis: {error}") from error
        else:
            indices = np.asarray(selection, dtype=np.int64)
        if indices.size == 0:
            raise InvalidTrajectoryError(f"выборка {selection!r} пуста — проверь имена остатков и атомов")
        return indices

    def times_ps(self) -> FloatArray:
        reader = self.universe.trajectory
        times: FloatArray = reader[0].time + reader.dt * np.arange(len(reader), dtype=np.float64)
        return times

    def _read(self, frame_indices: Sequence[int], groups: Sequence[IndexArray]) -> Iterator[TrajectoryFrame]:
        atoms = self.universe.atoms
        for timestep in self.universe.trajectory[list(frame_indices)]:
            dimensions = timestep.dimensions
            if dimensions is None or np.any(np.abs(dimensions[3:] - RECTANGULAR_ANGLE_DEG) > ANGLE_TOLERANCE_DEG):
                raise InvalidTrajectoryError(f"кадр {timestep.frame}: ящик не прямоугольный ({dimensions})")
            box_nm = np.asarray(dimensions[:3], dtype=np.float64) / ANGSTROM_PER_NM
            check_box(box_nm)
            positions = np.asarray(atoms.positions, dtype=np.float64) / ANGSTROM_PER_NM
            yield TrajectoryFrame(
                index=int(timestep.frame),
                time_ps=float(timestep.time),
                box_nm=box_nm,
                positions_nm=tuple(positions[group] for group in groups),
            )


def load_trajectory(topology: str | Path, *trajectory_files: str | Path) -> MDATrajectory:
    """Открыть траекторию МД: топология (.tpr, .gro, .pdb, …) и файлы траектории (.xtc, .trr, …)."""
    mdanalysis = _import_mdanalysis()
    paths = [str(path) for path in (topology, *trajectory_files)]
    for path in paths:
        if not Path(path).exists():
            raise InvalidTrajectoryError(f"файл {path} не найден")
    return MDATrajectory(mdanalysis.Universe(*paths))
