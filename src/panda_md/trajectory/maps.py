"""2D-карта плотности щели: длинная сторона ячейки × ось профиля, усреднение по короткой стороне.

Карта показывает саму форму (валик, каплю, отверстие) — для проверки, что выбранная форма та, и для рисунков. Без
осевого центрирования форма, блуждающая по ящику, размазывается; с ним ось формы в каждом кадре ставится в x = 0.
У форм с вертикальной осью (пончик, перфорация) усреднение по всей короткой стороне сглаживает круглое сечение —
`width_window_nm` оставляет только полосу вокруг оси.
"""

import math
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from pydantic import Field, model_validator

from panda_md.core.centering import axis_position
from panda_md.core.enums import AxisOrientation, Frame, Orientation, ParticleGrouping, ProfileCentering
from panda_md.core.errors import InvalidTrajectoryError, NotDerivedError, PandaWarning
from panda_md.core.pbc import wrap_centered
from panda_md.core.profile import load_npz_arrays, save_npz_arrays
from panda_md.core.timing import TimeValue
from panda_md.core.trajectory import Trajectory, check_box
from panda_md.core.types import Array1D, Array2D, FloatArray, IndexArray, PandaModel
from panda_md.pores.slit.pore import SlitPore
from panda_md.trajectory.density import (
    OUTSIDE_FRACTION_WARNING,
    check_area,
    check_box_change,
    particle_positions,
    profile_bin_edges,
    profile_frame_center,
    resolve_centering,
)

DEFAULT_MAP_BINS = (100, 100)
MIN_MAP_BINS = 2


class DensityMap(PandaModel):
    """ρ (нм⁻³) на сетке: вдоль длинной стороны × вдоль оси профиля, среднее по кадрам и по короткой стороне."""

    length_edges_nm: Array1D
    height_edges_nm: Array1D
    density_per_nm3: Array2D  # бины по длинной стороне × бины по оси профиля
    frame: Frame
    axis_centered: bool  # ось формы в каждом кадре поставлена в x = 0
    flipped: bool = False  # карта отражена z → −z (жидкость была у верхней стенки), как профиль
    width_window_nm: float | None = Field(default=None, gt=0.0)  # учтена только полоса вокруг вертикальной оси
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
    centering: ProfileCentering | str | None = None,
    bins: tuple[int, int] = DEFAULT_MAP_BINS,
    start: TimeValue = 0.0,
    stop: TimeValue | None = None,
    stride: int = 1,
    pore_center_nm: float | None = None,
    per: ParticleGrouping | str = ParticleGrouping.ATOM,
    orientation: Orientation | str = Orientation.AUTO,
    width_window_nm: float | None = None,
) -> DensityMap:
    """Карта плотности выборки в плоскости «длинная сторона × ось профиля».

    shape — имя формы поры с осью симметрии (`roll`, `doughnut`, `perforation`): тогда в каждом кадре ось формы
    ставится в x = 0 (осевое центрирование); без shape — координата вдоль длинной стороны как в траектории.
    Ось профиля центрируется и при необходимости отражается так же, как в `density_profile`. width_window_nm —
    только для вертикальной оси: учитываются частицы не дальше половины окна от оси по короткой стороне.
    """
    centering = resolve_centering(centering, pore, pore_center_nm)
    orientation = Orientation(orientation)
    grouping = ParticleGrouping(per)
    if min(bins) < MIN_MAP_BINS:
        raise ValueError(f"бинов {bins}: нужно хотя бы {MIN_MAP_BINS} по каждой оси")
    axis = None
    if shape is not None:
        axis = pore.interface(shape).shape.symmetry_axis
        if axis is None:
            raise NotDerivedError(f"у формы {shape} нет оси симметрии — осевое центрирование не определено")
    if width_window_nm is not None and (axis is None or axis.orientation is not AxisOrientation.VERTICAL):
        raise ValueError("width_window_nm — полоса вокруг вертикальной оси: нужна форма с вертикальной осью (shape)")
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
    first_length = first_height = 0.0
    frame_count = 0
    warned: set[str] = set()
    for frame in trajectory.frames(groups, start=start, stop=stop, stride=stride):
        check_box(frame.box_nm)
        positions = particle_positions(frame.positions_nm[0], frame.box_nm, molecules)
        period = float(frame.box_nm[normal])
        center = profile_frame_center(frame, positions[:, normal], pore, centering, pore_center_nm, period, warned)
        heights = wrap_centered(positions[:, normal] - center, period)
        length = float(frame.box_nm[length_column])
        width = float(frame.box_nm[width_column])
        along = positions[:, length_column]
        selected = np.ones(along.size, dtype=bool)
        if axis is not None:
            lateral = positions[:, [length_column, width_column]]
            located = axis_position(lateral, frame.box_nm[[length_column, width_column]], axis)
            along = along - located[0]
            if width_window_nm is not None:
                across = wrap_centered(positions[:, width_column] - located[1], width)
                selected = np.abs(across) < 0.5 * width_window_nm
        along = wrap_centered(along, length)
        if length_edges is None or height_edges is None or total is None:
            first_length, first_height = length, period
            length_edges = np.linspace(-0.5 * length, 0.5 * length, bins[0] + 1)
            height_edges = profile_bin_edges(pore, centering, period, bins[1], None)
            total = np.zeros((bins[0], height_edges.size - 1))
        counts, _, _ = np.histogram2d(along[selected], heights[selected], bins=(length_edges, height_edges))
        check_box_change(frame.index, length, first_length, False, warned, "вдоль длинной стороны")
        if centering is ProfileCentering.LIQUID:
            check_box_change(frame.index, period, first_height, False, warned)
        elif "outside" not in warned and selected.any():
            in_pore = (heights[selected] >= height_edges[0]) & (heights[selected] < height_edges[-1])
            outside = 1.0 - float(np.count_nonzero(in_pore)) / int(np.count_nonzero(selected))
            if outside > OUTSIDE_FRACTION_WARNING:
                warned.add("outside")
                warnings.warn(
                    f"{outside:.1%} выбранных частиц вне поры — проверь выборку и стенки", PandaWarning, stacklevel=2
                )
        check_area(pore, length * width, warned)
        strip = width if width_window_nm is None else min(width_window_nm, width)
        total += counts / (np.outer(np.diff(length_edges), np.diff(height_edges)) * strip)
        frame_count += 1
    assert length_edges is not None and height_edges is not None and total is not None  # кадры есть
    density = total / frame_count
    frame_reference = Frame.PORE if centering is ProfileCentering.PORE else Frame.LIQUID
    flipped = False
    symmetric = math.isclose(height_edges[0], -height_edges[-1])
    if orientation is Orientation.AUTO and frame_reference is Frame.PORE and symmetric:
        heights_centers = 0.5 * (height_edges[:-1] + height_edges[1:])
        if float(np.sum(heights_centers * density.mean(axis=0))) > 0.0:
            density = density[:, ::-1]
            flipped = True
    return DensityMap(
        length_edges_nm=length_edges,
        height_edges_nm=height_edges,
        density_per_nm3=density,
        frame=frame_reference,
        axis_centered=axis is not None,
        flipped=flipped,
        width_window_nm=width_window_nm,
        frame_count=frame_count,
        selection=select if isinstance(select, str) else f"{len(liquid)} индексов",
    )
