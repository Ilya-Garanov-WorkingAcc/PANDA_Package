"""Trajectory as a sequence of frames: an interface shared by MD files and numpy arrays.

Implementations — `panda_md.trajectory` (`ArrayTrajectory`, `MDATrajectory`). The core and the pores know only this
interface, so MDAnalysis is needed only for reading files. All lengths are in nm, time in ps, the box is rectangular.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence

import numpy as np
from pydantic import Field

from panda_md.core.errors import InvalidTrajectoryError
from panda_md.core.timing import TimeValue, to_picoseconds
from panda_md.core.types import Array1D, Array2D, FloatArray, IndexArray, PandaModel

BOX_DIMENSIONS = 3
TIME_TOLERANCE_PS = 1e-6  # a frame on the boundary of the time interval is included


class TrajectoryFrame(PandaModel):
    """One frame: time, box and positions of the requested atom groups (in the order of the request)."""

    index: int = Field(ge=0)  # frame index in the trajectory
    time_ps: float
    box_nm: Array1D  # edge lengths of the rectangular box
    positions_nm: tuple[Array2D, ...]  # one (n, 3) array per group


class Trajectory(ABC):
    """Source of frames. An atom selection is a string (the backend's selection language) or explicit indices."""

    @abstractmethod
    def select(self, selection: str | Sequence[int]) -> IndexArray:
        """Atom indices of the selection. An empty selection — InvalidTrajectoryError."""

    @abstractmethod
    def molecules(self, indices: IndexArray) -> tuple[IndexArray, FloatArray]:
        """For the selected atoms: molecule index (0 … n − 1 in order of first appearance) and atom mass."""

    @abstractmethod
    def times_ps(self) -> FloatArray:
        """Time of each frame, ps."""

    @abstractmethod
    def _read(self, frame_indices: Sequence[int], groups: Sequence[IndexArray]) -> Iterator[TrajectoryFrame]:
        """Read frames by index (in ascending order), yield the positions of the groups."""

    def frames(
        self,
        groups: Sequence[IndexArray],
        *,
        start: TimeValue = 0.0,
        stop: TimeValue | None = None,
        stride: int = 1,
    ) -> Iterator[TrajectoryFrame]:
        """Frames with start ≤ t ≤ stop, every stride-th; streamed, without loading the whole trajectory into memory."""
        indices = self.frame_indices(start=start, stop=stop, stride=stride)
        return self._read(indices, groups)

    def frame_indices(self, *, start: TimeValue = 0.0, stop: TimeValue | None = None, stride: int = 1) -> list[int]:
        if stride < 1:
            raise ValueError(f"frame stride must be ≥ 1: stride = {stride}")
        times = self.times_ps()
        if np.any(np.diff(times) <= 0.0):
            raise InvalidTrajectoryError(
                "frame times do not increase — concatenated trajectories overlap; remove the overlap "
                "(for example, gmx trjcat -settime)"
            )
        lower = to_picoseconds(start)
        upper = float(times[-1]) if stop is None else to_picoseconds(stop)
        inside = np.flatnonzero((times >= lower - TIME_TOLERANCE_PS) & (times <= upper + TIME_TOLERANCE_PS))
        if inside.size == 0:
            raise InvalidTrajectoryError(
                f"no frames in the interval [{lower:g}, {upper:g}] ps; "
                f"the trajectory spans [{times[0]:g}, {times[-1]:g}] ps"
            )
        return [int(index) for index in inside[::stride]]

    def frame_at(self, time: TimeValue, groups: Sequence[IndexArray]) -> TrajectoryFrame:
        """First frame with t ≥ time."""
        return next(self._read(self.frame_indices(start=time)[:1], groups))


def check_box(box_nm: FloatArray) -> None:
    if box_nm.shape != (BOX_DIMENSIONS,) or np.any(box_nm <= 0.0):
        raise InvalidTrajectoryError(
            f"invalid box {box_nm.tolist()}: three positive edge lengths of a rectangular box are required"
        )


def renumber(labels: IndexArray) -> IndexArray:
    """Molecule labels → 0 … n − 1 in order of first appearance."""
    _, first, inverse = np.unique(labels, return_index=True, return_inverse=True)
    order = np.argsort(np.argsort(first))
    renumbered: IndexArray = order[inverse].astype(np.int64)
    return renumbered
