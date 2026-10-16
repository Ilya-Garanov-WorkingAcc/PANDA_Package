"""Траектория из массивов numpy — для синтетических данных и тестов (MDAnalysis не нужен)."""

from collections.abc import Iterator, Mapping, Sequence

import numpy as np
from pydantic import Field, model_validator

from panda_md.core.errors import InvalidTrajectoryError
from panda_md.core.trajectory import BOX_DIMENSIONS, Trajectory, TrajectoryFrame
from panda_md.core.types import Array1D, Array2D, Array3D, FloatArray, IndexArray, IndexArray1D, PandaModel


class ArrayTrajectory(PandaModel, Trajectory):
    """Кадры из массивов: позиции (кадры × атомы × 3, нм), ящики (кадры × 3, нм), времена (пс).

    Выборка — явные индексы или имя группы из `groups` (например, {"water": [...], "walls": [...]}).
    """

    positions_nm: Array3D
    boxes_nm: Array2D
    times_ps_values: Array1D = Field(alias="times_ps")
    groups: Mapping[str, IndexArray1D] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_shapes(self) -> "ArrayTrajectory":
        frame_count, atom_count, dimensions = self.positions_nm.shape
        if dimensions != BOX_DIMENSIONS or frame_count == 0:
            raise ValueError(f"позиции формы {self.positions_nm.shape}: нужно (кадры ≥ 1, атомы, 3)")
        if self.boxes_nm.shape != (frame_count, BOX_DIMENSIONS) or np.any(self.boxes_nm <= 0.0):
            raise ValueError(f"ящики формы {self.boxes_nm.shape}: нужно ({frame_count}, 3) с положительными длинами")
        if self.times_ps_values.shape != (frame_count,) or np.any(np.diff(self.times_ps_values) <= 0.0):
            raise ValueError("времена: по одному на кадр, строго по возрастанию")
        for name, indices in self.groups.items():
            if indices.size and int(np.max(indices)) >= atom_count:
                raise ValueError(f"группа {name}: индекс {int(np.max(indices))} при {atom_count} атомах")
        return self

    def select(self, selection: str | Sequence[int]) -> IndexArray:
        if isinstance(selection, str):
            if selection not in self.groups:
                raise InvalidTrajectoryError(f"группы {selection!r} нет; доступны: {list(self.groups)}")
            indices = self.groups[selection]
        else:
            indices = np.asarray(selection, dtype=np.int64)
            if indices.size and (int(np.min(indices)) < 0 or int(np.max(indices)) >= self.positions_nm.shape[1]):
                raise InvalidTrajectoryError(f"индексы выборки вне [0, {self.positions_nm.shape[1]})")
        if indices.size == 0:
            raise InvalidTrajectoryError(f"выборка {selection!r} пуста")
        return indices

    def times_ps(self) -> FloatArray:
        return self.times_ps_values

    def _read(self, frame_indices: Sequence[int], groups: Sequence[IndexArray]) -> Iterator[TrajectoryFrame]:
        for index in frame_indices:
            frame_positions = self.positions_nm[index]
            yield TrajectoryFrame(
                index=index,
                time_ps=float(self.times_ps_values[index]),
                box_nm=self.boxes_nm[index],
                positions_nm=tuple(frame_positions[group] for group in groups),
            )
