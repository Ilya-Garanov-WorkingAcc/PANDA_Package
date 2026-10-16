"""Траектория как последовательность кадров: интерфейс, общий для файлов МД и массивов numpy.

Реализации — `panda_md.trajectory` (`ArrayTrajectory`, `MDATrajectory`). Ядро и поры знают только этот интерфейс,
поэтому MDAnalysis нужен лишь для чтения файлов. Все длины — нм, время — пс, ящик прямоугольный.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence

import numpy as np
from pydantic import Field

from panda_md.core.errors import InvalidTrajectoryError
from panda_md.core.timing import TimeValue, to_picoseconds
from panda_md.core.types import Array1D, Array2D, FloatArray, IndexArray, PandaModel

BOX_DIMENSIONS = 3
TIME_TOLERANCE_PS = 1e-6  # кадр на границе интервала времени включается


class TrajectoryFrame(PandaModel):
    """Один кадр: время, ящик и позиции запрошенных групп атомов (в порядке запроса)."""

    index: int = Field(ge=0)  # номер кадра в траектории
    time_ps: float
    box_nm: Array1D  # длины рёбер прямоугольного ящика
    positions_nm: tuple[Array2D, ...]  # по одному массиву (n, 3) на группу


class Trajectory(ABC):
    """Источник кадров. Выборка атомов — строка (язык выборки бэкенда) или явные индексы."""

    @abstractmethod
    def select(self, selection: str | Sequence[int]) -> IndexArray:
        """Индексы атомов выборки. Пустая выборка — InvalidTrajectoryError."""

    @abstractmethod
    def molecules(self, indices: IndexArray) -> tuple[IndexArray, FloatArray]:
        """Для атомов выборки: номер молекулы (0 … n − 1 в порядке первого появления) и масса атома."""

    @abstractmethod
    def times_ps(self) -> FloatArray:
        """Время каждого кадра, пс."""

    @abstractmethod
    def _read(self, frame_indices: Sequence[int], groups: Sequence[IndexArray]) -> Iterator[TrajectoryFrame]:
        """Прочитать кадры по номерам (по возрастанию), отдать позиции групп."""

    def frames(
        self,
        groups: Sequence[IndexArray],
        *,
        start: TimeValue = 0.0,
        stop: TimeValue | None = None,
        stride: int = 1,
    ) -> Iterator[TrajectoryFrame]:
        """Кадры с start ≤ t ≤ stop, каждый stride-й; потоково, без загрузки всей траектории в память."""
        indices = self.frame_indices(start=start, stop=stop, stride=stride)
        return self._read(indices, groups)

    def frame_indices(self, *, start: TimeValue = 0.0, stop: TimeValue | None = None, stride: int = 1) -> list[int]:
        if stride < 1:
            raise ValueError(f"stride = {stride}: шаг по кадрам должен быть ≥ 1")
        times = self.times_ps()
        lower = to_picoseconds(start)
        upper = float(times[-1]) if stop is None else to_picoseconds(stop)
        inside = np.flatnonzero((times >= lower - TIME_TOLERANCE_PS) & (times <= upper + TIME_TOLERANCE_PS))
        if inside.size == 0:
            raise InvalidTrajectoryError(
                f"в интервале [{lower:g}, {upper:g}] пс нет кадров; траектория — [{times[0]:g}, {times[-1]:g}] пс"
            )
        return [int(index) for index in inside[::stride]]

    def frame_at(self, time: TimeValue, groups: Sequence[IndexArray]) -> TrajectoryFrame:
        """Первый кадр с t ≥ time."""
        return next(self._read(self.frame_indices(start=time)[:1], groups))


def check_box(box_nm: FloatArray) -> None:
    if box_nm.shape != (BOX_DIMENSIONS,) or np.any(box_nm <= 0.0):
        raise InvalidTrajectoryError(f"ящик {box_nm.tolist()}: нужны три положительные длины прямоугольного ящика")


def renumber(labels: IndexArray) -> IndexArray:
    """Метки молекул → 0 … n − 1 в порядке первого появления."""
    _, first, inverse = np.unique(labels, return_index=True, return_inverse=True)
    order = np.argsort(np.argsort(first))
    renumbered: IndexArray = order[inverse].astype(np.int64)
    return renumbered
