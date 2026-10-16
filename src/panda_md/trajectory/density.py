"""Профиль плотности по траектории: выборка → центр в каждом кадре → координата и бины поры → средние по блокам.

Чтение потоковое: в памяти только текущий кадр и суммы по блокам.
"""

import math
import warnings
from collections.abc import Sequence

import numpy as np

from panda_md.core.enums import Frame, Orientation, ProfileCentering
from panda_md.core.errors import InvalidTrajectoryError, PandaWarning
from panda_md.core.pbc import circular_mean, wrap_centered
from panda_md.core.pore import Pore
from panda_md.core.profile import ProfileSet
from panda_md.core.timing import TimeValue, to_picoseconds
from panda_md.core.trajectory import Trajectory, TrajectoryFrame, check_box
from panda_md.core.types import FloatArray, IndexArray

DEFAULT_BIN_COUNT = 200
WALL_GAP_TOLERANCE = 0.02  # промежуток между стенками в кадре отличается от H поры больше чем на 2 % — предупреждение
BOX_AREA_TOLERANCE = 0.01  # площадь основания ящика отличается от основания поры больше чем на 1 % — предупреждение
OUTSIDE_FRACTION_WARNING = 0.01  # больше 1 % выбранных частиц вне поры — предупреждение
BOX_CHANGE_TOLERANCE = 0.01  # в системе жидкости бины — по ящику первого кадра; изменение больше 1 % — предупреждение


class _BlockAccumulator:
    """Суммы по кадрам текущего блока."""

    def __init__(self, bin_count: int) -> None:
        self.density_sum: FloatArray = np.zeros(bin_count)
        self.particle_sum: float = 0.0
        self.frame_count: int = 0
        self.start_ps: float = math.nan
        self.stop_ps: float = math.nan

    def add(self, density: FloatArray, particles: int, time_ps: float) -> None:
        if self.frame_count == 0:
            self.start_ps = time_ps
        self.density_sum += density
        self.particle_sum += particles
        self.frame_count += 1
        self.stop_ps = time_ps


def density_profile(
    trajectory: Trajectory,
    pore: Pore,
    select: str | Sequence[int],
    *,
    centering: ProfileCentering | str = ProfileCentering.PORE,
    bins: int = DEFAULT_BIN_COUNT,
    bin_width_nm: float | None = None,
    start: TimeValue = 0.0,
    stop: TimeValue | None = None,
    block: TimeValue | None = None,
    stride: int = 1,
    pore_center_nm: float | None = None,
    orientation: Orientation | str = Orientation.AUTO,
) -> ProfileSet:
    """Профили плотности выбранных частиц вдоль оси поры, усреднённые по блокам времени длины `block`.

    centering=PORE: 0 — центр поры; он берётся из `pore_center_nm` или в каждом кадре из атомов стенок
    (`pore.wall_selection`); бины выровнены по стенкам поры. centering=LIQUID: 0 — центр выборки (среднее
    по атомам без весов) с учётом PBC; бины покрывают весь ящик вдоль оси профиля. Неполный последний блок
    отбрасывается.
    """
    centering = ProfileCentering(centering)
    orientation = Orientation(orientation)
    liquid = trajectory.select(select)
    groups: list[IndexArray] = [liquid]
    use_walls = centering is ProfileCentering.PORE and pore_center_nm is None
    if use_walls:
        if pore.wall_selection is None:
            raise InvalidTrajectoryError(
                "центрирование по поре: задай pore_center_nm или выборку стенок (wall_selection у поры)"
            )
        groups.append(trajectory.select(pore.wall_selection))
    frame_indices = trajectory.frame_indices(start=start, stop=stop, stride=stride)
    times = trajectory.times_ps()[frame_indices]
    block_ps = None if block is None else to_picoseconds(block)
    block_of_frame, block_count = _assign_blocks(times, block_ps)

    axis = pore.normal_axis.column
    edges: FloatArray | None = None
    first_period = 0.0
    blocks: list[_BlockAccumulator] = []
    warned: set[str] = set()
    for frame, block_index in zip(
        trajectory.frames(groups, start=start, stop=stop, stride=stride), block_of_frame, strict=True
    ):
        if block_index < 0:
            continue
        check_box(frame.box_nm)
        period = float(frame.box_nm[axis])
        center = _frame_center(frame, pore, centering, pore_center_nm, period, warned)
        coordinate = wrap_centered(frame.positions_nm[0][:, axis] - center, period)
        if edges is None:
            first_period = period
            edges = _bin_edges(pore, centering, period, bins, bin_width_nm)
            blocks = [_BlockAccumulator(edges.size - 1) for _ in range(block_count)]
        counts, _ = np.histogram(coordinate, bins=edges)
        if centering is ProfileCentering.LIQUID:
            _check_box_change(frame.index, period, first_period, int(counts.sum()) < coordinate.size, warned)
        area = pore.bin_cross_section_nm2(frame.box_nm)
        _check_area(pore, area, warned)
        density = counts / (area * np.diff(edges))
        inside = int(counts.sum()) if centering is ProfileCentering.PORE else coordinate.size
        if centering is ProfileCentering.PORE and "outside" not in warned:
            outside_fraction = 1.0 - inside / coordinate.size
            if outside_fraction > OUTSIDE_FRACTION_WARNING:
                warned.add("outside")
                warnings.warn(
                    f"{outside_fraction:.1%} выбранных частиц вне поры — проверь выборку и стенки",
                    PandaWarning,
                    stacklevel=2,
                )
        blocks[block_index].add(density, inside, frame.time_ps)

    assert edges is not None  # frame_indices не пуст
    densities = np.array([accumulator.density_sum / accumulator.frame_count for accumulator in blocks])
    frame_reference = Frame.PORE if centering is ProfileCentering.PORE else Frame.LIQUID
    flipped = False
    if orientation is Orientation.AUTO and frame_reference is Frame.PORE and math.isclose(edges[0], -edges[-1]):
        centers = 0.5 * (edges[:-1] + edges[1:])
        if float(np.sum(centers * densities.mean(axis=0))) > 0.0:
            densities = densities[:, ::-1]
            flipped = True
    return ProfileSet(
        bin_edges_nm=edges,
        block_densities_per_nm3=densities,
        block_start_ps=np.array([accumulator.start_ps for accumulator in blocks]),
        block_stop_ps=np.array([accumulator.stop_ps for accumulator in blocks]),
        block_frame_counts=tuple(accumulator.frame_count for accumulator in blocks),
        block_particle_counts=np.array([accumulator.particle_sum / accumulator.frame_count for accumulator in blocks]),
        frame=frame_reference,
        centering=centering,
        selection=select if isinstance(select, str) else f"{len(liquid)} индексов",
        pore_volume_nm3=pore.volume_nm3(),
        flipped=flipped,
    )


def _assign_blocks(times: FloatArray, block_ps: float | None) -> tuple[list[int], int]:
    """Номер блока каждого кадра (−1 — кадр неполного последнего блока) и число полных блоков."""
    if block_ps is None:
        return [0] * times.size, 1
    if block_ps <= 0.0:
        raise ValueError(f"длина блока {block_ps} пс должна быть положительной")
    spacing = float(np.median(np.diff(times))) if times.size > 1 else block_ps
    elapsed = times - times[0]
    block_count = math.floor((elapsed[-1] + spacing) / block_ps + 1e-9)
    if block_count == 0:
        raise InvalidTrajectoryError(
            f"интервал {elapsed[-1] + spacing:g} пс короче одного блока {block_ps:g} пс — уменьши block"
        )
    assignment = [math.floor(value / block_ps + 1e-9) for value in elapsed]
    dropped = sum(1 for index in assignment if index >= block_count)
    if dropped > 1:  # один кадр ровно на границе — начало следующего блока, а не неполный блок
        warnings.warn(f"{dropped} последних кадров не составляют полного блока и не учтены", PandaWarning, stacklevel=3)
    return [index if index < block_count else -1 for index in assignment], block_count


def _frame_center(
    frame: TrajectoryFrame,
    pore: Pore,
    centering: ProfileCentering,
    pore_center_nm: float | None,
    period: float,
    warned: set[str],
) -> float:
    axis = pore.normal_axis.column
    if centering is ProfileCentering.LIQUID:
        return circular_mean(frame.positions_nm[0][:, axis], period)
    if pore_center_nm is not None:
        return pore_center_nm
    walls = frame.positions_nm[1][:, axis]
    center = pore.locate_center_nm(walls, period)
    gap = _wall_gap(walls, period, center)
    expected = pore.coordinate_bounds_nm()[1] - pore.coordinate_bounds_nm()[0]
    if abs(gap - expected) > WALL_GAP_TOLERANCE * expected and "gap" not in warned:
        warned.add("gap")
        warnings.warn(
            f"кадр {frame.index}: промежуток между стенками {gap:.3f} нм, а размер поры {expected:.3f} нм — "
            f"стенки сместились или пора задана неверно",
            PandaWarning,
            stacklevel=3,
        )
    return center


def _wall_gap(walls: FloatArray, period: float, center: float) -> float:
    """Ширина пустого промежутка вокруг center: ближайшие атомы стенок снизу и сверху."""
    offsets = wrap_centered(walls - center, period)
    above, below = offsets[offsets > 0], offsets[offsets <= 0]
    if above.size == 0 or below.size == 0:
        return period  # стенки в одной плоскости: пора — весь ящик
    return float(np.min(above) - np.max(below))


def _bin_edges(
    pore: Pore, centering: ProfileCentering, period: float, bins: int, bin_width_nm: float | None
) -> FloatArray:
    if centering is ProfileCentering.PORE:
        lower, upper = pore.coordinate_bounds_nm()
        if upper - lower > period:
            raise InvalidTrajectoryError(
                f"пора ({upper - lower:.3f} нм) шире ящика вдоль оси профиля ({period:.3f} нм)"
            )
    else:
        lower, upper = -0.5 * period, 0.5 * period
    count = bins if bin_width_nm is None else max(1, round((upper - lower) / bin_width_nm))
    if count < 2:
        raise ValueError(f"бинов {count}: нужно хотя бы 2")
    edges: FloatArray = np.linspace(lower, upper, count + 1)
    return edges


def _check_box_change(frame_index: int, period: float, first_period: float, lost: bool, warned: set[str]) -> None:
    """В системе жидкости бины построены по ящику первого кадра: ящик заметно изменился — предупредить один раз."""
    if "box" in warned:
        return
    if abs(period - first_period) > BOX_CHANGE_TOLERANCE * first_period or lost:
        warned.add("box")
        warnings.warn(
            f"кадр {frame_index}: ящик вдоль оси профиля {period:.3f} нм, а бины построены по первому кадру "
            f"({first_period:.3f} нм) — часть частиц вне бинов или крайние бины недозаполнены",
            PandaWarning,
            stacklevel=3,
        )


def _check_area(pore: Pore, area: float, warned: set[str]) -> None:
    base = pore.base_area_nm2()
    if base is None:
        return
    if abs(area - base) > BOX_AREA_TOLERANCE * base and "area" not in warned:
        warned.add("area")
        warnings.warn(
            f"площадь бина {area:.3f} нм² отличается от основания поры {base:.3f} нм² — проверь размеры поры",
            PandaWarning,
            stacklevel=4,
        )
