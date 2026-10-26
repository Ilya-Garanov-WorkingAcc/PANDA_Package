"""Density profile over a trajectory: selection → centre in each frame → pore coordinate and bins → block averages.

Reading is streamed: only the current frame and the block sums are kept in memory.
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
WALL_GAP_TOLERANCE = 0.02  # the gap between the walls in a frame differs from the pore H by more than 2 % — warning
BOX_AREA_TOLERANCE = 0.01  # the box base area differs from the pore base by more than 1 % — warning
OUTSIDE_FRACTION_WARNING = 0.01  # more than 1 % of the selected particles outside the pore — warning
BOX_CHANGE_TOLERANCE = 0.01  # in the liquid frame the bins follow the first frame box; a change over 1 % — warning


class _BlockAccumulator:
    """Sums over the frames of the current block."""

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
    """Density profiles of the selected particles along the pore axis, averaged over time blocks of length `block`.

    centering=None: pore centering if the walls (`pore.wall_selection`) or `pore_center_nm` are known, otherwise
    liquid centering with a warning. centering=PORE: 0 is the pore centre; it is taken from `pore_center_nm` or, in
    each frame, from the wall atoms
    (`pore.wall_selection`); the bins are aligned with the pore walls. centering=LIQUID: 0 is the centre of the
    selection (unweighted mean over atoms) with PBC; the bins cover the whole box along the profile axis. An
    incomplete last block is dropped. per=MOLECULE: a particle is the centre of mass of the selected atoms of a
    molecule (with PBC; “name OW” gives the oxygen, not the centre of mass of water), N is the number of molecules.
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
                "pore centering needs the pore centre; set pore_center_nm or a wall selection "
                "(wall_selection of the pore)"
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
                    f"{outside_fraction:.1%} of the selected particles are outside the pore; check the selection "
                    f"and the walls",
                    PandaWarning,
                    stacklevel=2,
                )
        blocks[block_index].add(density, inside, frame.time_ps)

    assert edges is not None  # frame_indices is not empty
    empty = sum(1 for accumulator in blocks if accumulator.frame_count == 0)
    if empty:
        warnings.warn(
            f"{empty} block(s) without frames (a time gap in the trajectory); skipped", PandaWarning, stacklevel=2
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
        selection=select if isinstance(select, str) else f"{len(liquid)} indices",
        pore_volume_nm3=pore.volume_nm3(),
        flipped=flipped,
    )


def resolve_centering(
    centering: ProfileCentering | str | None, pore: Pore, pore_center_nm: float | None
) -> ProfileCentering:
    """Explicit centering is used as given; without it — pore centering if the pore centre can be found, else liquid."""
    if centering is not None:
        return ProfileCentering(centering)
    if pore.wall_selection is not None or pore_center_nm is not None:
        return ProfileCentering.PORE
    warnings.warn(
        "pore walls are not set (wall_selection) and the pore centre is unknown (pore_center_nm): the profile is "
        "centred on the liquid; set the walls for shapes that touch both walls",
        PandaWarning,
        stacklevel=3,
    )
    return ProfileCentering.LIQUID


def _assign_blocks(times: FloatArray, block_ps: float | None) -> tuple[list[int], int]:
    """Block number of each frame (−1 for a frame of the incomplete last block) and the number of complete blocks."""
    if block_ps is None:
        return [0] * times.size, 1
    if block_ps <= 0.0:
        raise ValueError(f"block length must be positive: {block_ps} ps")
    if times.size == 1:
        warnings.warn(
            f"a single frame: nothing to check the block length {block_ps:g} ps against; the profile has one block",
            PandaWarning,
            stacklevel=3,
        )
        return [0], 1
    spacing = float(np.median(np.diff(times)))
    elapsed = times - times[0]
    block_count = math.floor((elapsed[-1] + spacing) / block_ps + 1e-9)
    if block_count == 0:
        raise InvalidTrajectoryError(
            f"interval {elapsed[-1] + spacing:g} ps is shorter than one block of {block_ps:g} ps; reduce block"
        )
    assignment = [math.floor(value / block_ps + 1e-9) for value in elapsed]
    dropped = sum(1 for index in assignment if index >= block_count)
    if dropped > 1:  # a single frame exactly on the boundary starts the next block, it is not an incomplete block
        warnings.warn(
            f"{dropped} last frame(s) do not make a complete block and are ignored", PandaWarning, stacklevel=3
        )
    return [index if index < block_count else -1 for index in assignment], block_count


def particle_positions(
    positions: FloatArray, box: FloatArray, molecules: tuple[IndexArray, FloatArray] | None
) -> FloatArray:
    """Particle positions (n × 3): atoms or molecular centres of mass (molecules: molecule number, mass per atom)."""
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
    walls = frame.positions_nm[1]
    center = pore.locate_center_nm(walls[:, axis], period)
    measured, expected = pore.wall_size_check_nm(walls, frame.box_nm, center)
    if abs(measured - expected) > WALL_GAP_TOLERANCE * expected and "gap" not in warned:
        warned.add("gap")
        warnings.warn(
            f"frame {frame.index}: the {pore.wall_size_label} is {measured:.3f} nm, the pore size is "
            f"{expected:.3f} nm; the walls have moved or the pore is set incorrectly",
            PandaWarning,
            stacklevel=3,
        )
    return center


def profile_bin_edges(
    pore: Pore, centering: ProfileCentering, period: float, bins: int, bin_width_nm: float | None
) -> FloatArray:
    if centering is ProfileCentering.PORE:
        lower, upper = pore.coordinate_bounds_nm()
        if upper - lower > period:
            raise InvalidTrajectoryError(
                f"the pore ({upper - lower:.3f} nm) is wider than the box along the profile axis ({period:.3f} nm)"
            )
    else:
        lower, upper = -0.5 * period, 0.5 * period
    count = bins if bin_width_nm is None else max(1, round((upper - lower) / bin_width_nm))
    if count < 2:
        raise ValueError(f"too few bins: {count}; at least 2 are needed")
    edges: FloatArray = np.linspace(lower, upper, count + 1)
    return edges


def check_box_change(
    frame_index: int,
    period: float,
    first_period: float,
    lost: bool,
    warned: set[str],
    what: str = "along the profile axis",
) -> None:
    """The bins are built from the first frame box: if the box changed noticeably, warn once."""
    key = f"box {what}"
    if key in warned:
        return
    if abs(period - first_period) > BOX_CHANGE_TOLERANCE * first_period or lost:
        warned.add(key)
        warnings.warn(
            f"frame {frame_index}: the box {what} is {period:.3f} nm, but the bins follow the first frame "
            f"({first_period:.3f} nm); some particles are outside the bins or the edge bins are underfilled",
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
            f"bin area {area:.3f} nm² differs from the pore base {base:.3f} nm²; check the pore dimensions",
            PandaWarning,
            stacklevel=3,
        )
