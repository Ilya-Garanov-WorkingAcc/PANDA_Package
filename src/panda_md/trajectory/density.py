"""Профиль плотности по траектории: выборка → центр в каждом кадре → координата и бины поры → средние по блокам.

Чтение потоковое: в памяти только текущий кадр и суммы по блокам.
"""

import math
import warnings
from collections.abc import Sequence

import numpy as np

from panda_md.core.enums import Frame, Orientation, ParticleGrouping, ProfileCentering
from panda_md.core.errors import InvalidTrajectoryError, PandaWarning
from panda_md.core.pbc import circular_mean, molecule_centers, wrap_centered
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
    centering: ProfileCentering | str | None = None,
    bins: int = DEFAULT_BIN_COUNT,
    bin_width_nm: float | None = None,
    start: TimeValue = 0.0,
    stop: TimeValue | None = None,
    block: TimeValue | None = None,
    stride: int = 1,
    pore_center_nm: float | None = None,
    orientation: Orientation | str = Orientation.AUTO,
    per: ParticleGrouping | str = ParticleGrouping.ATOM,
) -> ProfileSet:
    """Профили плотности выбранных частиц вдоль оси поры, усреднённые по блокам времени длины `block`.

    centering=None: по поре, если известны стенки (`pore.wall_selection`) или `pore_center_nm`, иначе по жидкости
    с предупреждением. centering=PORE: 0 — центр поры; он берётся из `pore_center_nm` или в каждом кадре из атомов
    стенок
    (`pore.wall_selection`); бины выровнены по стенкам поры. centering=LIQUID: 0 — центр выборки (среднее
    по атомам без весов) с учётом PBC; бины покрывают весь ящик вдоль оси профиля. Неполный последний блок
    отбрасывается. per=MOLECULE: частица — центр масс выбранных атомов молекулы (с учётом PBC; «name OW» даст
    кислород, а не центр масс воды), N — число молекул.
    """
    centering = resolve_centering(centering, pore, pore_center_nm)
    orientation = Orientation(orientation)
    grouping = ParticleGrouping(per)
    liquid = trajectory.select(select)
    molecules = trajectory.molecules(liquid) if grouping is ParticleGrouping.MOLECULE else None
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
        heights = particle_positions(frame.positions_nm[0], frame.box_nm, molecules)[:, axis]
        center = profile_frame_center(frame, heights, pore, centering, pore_center_nm, period, warned)
        coordinate = wrap_centered(heights - center, period)
        if edges is None:
            first_period = period
            edges = profile_bin_edges(pore, centering, period, bins, bin_width_nm)
            blocks = [_BlockAccumulator(edges.size - 1) for _ in range(block_count)]
        counts, _ = np.histogram(coordinate, bins=edges)
        if centering is ProfileCentering.LIQUID:
            check_box_change(frame.index, period, first_period, int(counts.sum()) < coordinate.size, warned)
        area = pore.bin_cross_section_nm2(frame.box_nm)
        check_area(pore, area, warned)
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
    empty = sum(1 for accumulator in blocks if accumulator.frame_count == 0)
    if empty:
        warnings.warn(
            f"{empty} блоков без кадров (разрыв во времени траектории) — пропущены", PandaWarning, stacklevel=2
        )
        blocks = [accumulator for accumulator in blocks if accumulator.frame_count > 0]
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


def resolve_centering(
    centering: ProfileCentering | str | None, pore: Pore, pore_center_nm: float | None
) -> ProfileCentering:
    """Явное центрирование — как задано; без него — по поре, если её центр можно найти, иначе по жидкости."""
    if centering is not None:
        return ProfileCentering(centering)
    if pore.wall_selection is not None or pore_center_nm is not None:
        return ProfileCentering.PORE
    warnings.warn(
        "стенки поры не заданы (wall_selection) и центр поры не известен (pore_center_nm): профиль центрирован по "
        "жидкости — для форм, касающихся обеих стенок, задай стенки",
        PandaWarning,
        stacklevel=3,
    )
    return ProfileCentering.LIQUID


def _assign_blocks(times: FloatArray, block_ps: float | None) -> tuple[list[int], int]:
    """Номер блока каждого кадра (−1 — кадр неполного последнего блока) и число полных блоков."""
    if block_ps is None:
        return [0] * times.size, 1
    if block_ps <= 0.0:
        raise ValueError(f"длина блока {block_ps} пс должна быть положительной")
    if times.size == 1:
        warnings.warn(
            f"один кадр: длину блока {block_ps:g} пс проверить не по чему — профиль из одного блока",
            PandaWarning,
            stacklevel=3,
        )
        return [0], 1
    spacing = float(np.median(np.diff(times)))
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


def particle_positions(
    positions: FloatArray, box: FloatArray, molecules: tuple[IndexArray, FloatArray] | None
) -> FloatArray:
    """Позиции частиц (n × 3): атомы или центры масс молекул (molecules — номер молекулы и масса каждого атома)."""
    if molecules is None:
        return positions
    molecule_indices, masses = molecules
    centers: FloatArray = molecule_centers(positions, box, molecule_indices, masses)
    return centers


def profile_frame_center(
    frame: TrajectoryFrame,
    heights: FloatArray,
    pore: Pore,
    centering: ProfileCentering,
    pore_center_nm: float | None,
    period: float,
    warned: set[str],
) -> float:
    axis = pore.normal_axis.column
    if centering is ProfileCentering.LIQUID:
        return circular_mean(heights, period)
    if pore_center_nm is not None:
        return pore_center_nm
    walls = frame.positions_nm[1][:, axis]
    center = pore.locate_center_nm(walls, period)
    gap = _wall_gap(walls, period, center)
    expected = pore.expected_wall_gap_nm()
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


def profile_bin_edges(
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


def check_box_change(
    frame_index: int, period: float, first_period: float, lost: bool, warned: set[str], what: str = "вдоль оси профиля"
) -> None:
    """Бины построены по ящику первого кадра: ящик заметно изменился — предупредить один раз."""
    key = f"box {what}"
    if key in warned:
        return
    if abs(period - first_period) > BOX_CHANGE_TOLERANCE * first_period or lost:
        warned.add(key)
        warnings.warn(
            f"кадр {frame_index}: ящик {what} {period:.3f} нм, а бины построены по первому кадру "
            f"({first_period:.3f} нм) — часть частиц вне бинов или крайние бины недозаполнены",
            PandaWarning,
            stacklevel=3,
        )


def check_area(pore: Pore, area: float, warned: set[str]) -> None:
    base = pore.base_area_nm2()
    if base is None:
        return
    if abs(area - base) > BOX_AREA_TOLERANCE * base and "area" not in warned:
        warned.add("area")
        warnings.warn(
            f"площадь бина {area:.3f} нм² отличается от основания поры {base:.3f} нм² — проверь размеры поры",
            PandaWarning,
            stacklevel=3,
        )
