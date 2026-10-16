"""2D-карта плотности щели: длинная сторона ячейки × ось профиля, усреднение по короткой стороне.

Карта показывает саму форму (валик, каплю, отверстие) — для проверки, что выбранная форма та, и для рисунков. Без
осевого центрирования форма, блуждающая по ящику, размазывается; с ним ось формы в каждом кадре ставится в x = 0.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from pydantic import Field, model_validator

from panda_md.core.centering import axis_position
from panda_md.core.enums import Frame, ParticleGrouping, ProfileCentering
from panda_md.core.errors import InvalidTrajectoryError
from panda_md.core.pbc import wrap_centered
from panda_md.core.profile import load_npz_arrays, save_npz_arrays
from panda_md.core.timing import TimeValue
from panda_md.core.trajectory import Trajectory, check_box
from panda_md.core.types import Array1D, Array2D, FloatArray, IndexArray, PandaModel
from panda_md.pores.slit.pore import SlitPore
from panda_md.trajectory.density import particle_positions, profile_bin_edges, profile_frame_center

DEFAULT_MAP_BINS = (100, 100)


class DensityMap(PandaModel):
    """ρ (нм⁻³) на сетке: вдоль длинной стороны × вдоль оси профиля, среднее по кадрам и по короткой стороне."""

    length_edges_nm: Array1D
    height_edges_nm: Array1D
    density_per_nm3: Array2D  # бины по длинной стороне × бины по оси профиля
    frame: Frame
    axis_centered: bool  # ось формы в каждом кадре поставлена в x = 0
    frame_count: int = Field(ge=1)
    selection: str

    @model_validator(mode="after")
    def _check_shapes(self) -> "DensityMap":
        expected = (self.length_edges_nm.size - 1, self.height_edges_nm.size - 1)
        if self.density_per_nm3.shape != expected:
            raise ValueError(f"карта формы {self.density_per_nm3.shape}, а по краям бинов — {expected}")
        return self

    def save(self, path: str | Path) -> None:
        arrays = {name: getattr(self, name) for name in ("length_edges_nm", "height_edges_nm", "density_per_nm3")}
        save_npz_arrays(Path(path), arrays, self.model_dump(mode="json", exclude=set(arrays)))

    @classmethod
    def load(cls, path: str | Path) -> "DensityMap":
        arrays, metadata = load_npz_arrays(Path(path))
        return cls(**arrays, **metadata)


def density_map(
    trajectory: Trajectory,
    pore: SlitPore,
    select: str | Sequence[int],
    *,
    shape: str | None = None,
    centering: ProfileCentering | str = ProfileCentering.PORE,
    bins: tuple[int, int] = DEFAULT_MAP_BINS,
    start: TimeValue = 0.0,
    stop: TimeValue | None = None,
    stride: int = 1,
    pore_center_nm: float | None = None,
    per: ParticleGrouping | str = ParticleGrouping.ATOM,
) -> DensityMap:
    """Карта плотности выборки в плоскости «длинная сторона × ось профиля».

    shape — имя формы поры с осью симметрии (`roll`, `doughnut`, `perforation`): тогда в каждом кадре ось формы
    ставится в x = 0 (осевое центрирование); без shape — координата вдоль длинной стороны как в траектории.
    Ось профиля центрируется так же, как в `density_profile`.
    """
    centering = ProfileCentering(centering)
    grouping = ParticleGrouping(per)
    axis = None
    if shape is not None:
        axis = pore.interface(shape).shape.symmetry_axis
        if axis is None:
            raise InvalidTrajectoryError(f"у формы {shape} нет оси симметрии — осевое центрирование не определено")
    liquid = trajectory.select(select)
    groups: list[IndexArray] = [liquid]
    if centering is ProfileCentering.PORE and pore_center_nm is None:
        if pore.wall_selection is None:
            raise InvalidTrajectoryError("центрирование по поре: задай pore_center_nm или wall_selection у поры")
        groups.append(trajectory.select(pore.wall_selection))
    molecules = trajectory.molecules(liquid) if grouping is ParticleGrouping.MOLECULE else None
    normal = pore.normal_axis.column
    length_column = pore.length_axis.column
    (width_column,) = (column for column in range(3) if column not in (normal, length_column))
    length_edges: FloatArray | None = None
    height_edges: FloatArray | None = None
    total: FloatArray | None = None
    frame_count = 0
    warned: set[str] = set()
    for frame in trajectory.frames(groups, start=start, stop=stop, stride=stride):
        check_box(frame.box_nm)
        positions = particle_positions(frame.positions_nm[0], frame.box_nm, molecules)
        period = float(frame.box_nm[normal])
        center = profile_frame_center(frame, positions[:, normal], pore, centering, pore_center_nm, period, warned)
        heights = wrap_centered(positions[:, normal] - center, period)
        length = float(frame.box_nm[length_column])
        along = positions[:, length_column]
        if axis is not None:
            lateral = positions[:, [length_column, width_column]]
            periods = frame.box_nm[[length_column, width_column]]
            along = along - axis_position(lateral, periods, axis)[0]
        along = wrap_centered(along, length)
        if length_edges is None or height_edges is None or total is None:
            length_edges = np.linspace(-0.5 * length, 0.5 * length, bins[0] + 1)
            height_edges = profile_bin_edges(pore, centering, period, bins[1], None)
            total = np.zeros((bins[0], height_edges.size - 1))
        counts, _, _ = np.histogram2d(along, heights, bins=(length_edges, height_edges))
        cell = np.outer(np.diff(length_edges), np.diff(height_edges)) * float(frame.box_nm[width_column])
        total += counts / cell
        frame_count += 1
    assert length_edges is not None and height_edges is not None and total is not None  # кадры есть
    return DensityMap(
        length_edges_nm=length_edges,
        height_edges_nm=height_edges,
        density_per_nm3=total / frame_count,
        frame=Frame.PORE if centering is ProfileCentering.PORE else Frame.LIQUID,
        axis_centered=axis is not None,
        frame_count=frame_count,
        selection=select if isinstance(select, str) else f"{len(liquid)} индексов",
    )
