"""2D density map of a slit: long side of the cell × profile axis, averaged over the short side.

The map shows the shape itself (roll, droplet, hole) — to check that the chosen shape is right, and for figures.
Without axial centering a shape wandering through the box is smeared out; with it the shape axis is put at x = 0 in
each frame. For shapes with a vertical axis (doughnut, perforation) averaging over the whole short side smooths the
circular cross-section — `width_window_nm` keeps only a strip around the axis.
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

BULK_CORE_FRACTION = 0.5  # bulk density: the middle half of the map height, away from the wall layering
BULK_LEVEL = 0.5  # and the bins denser than half the densest bin there (the core of the phase)


class DensityMap(PandaModel):
    """ρ (nm⁻³) on a grid: along the long side × along the profile axis, averaged over frames and the short side."""

    length_edges_nm: Array1D
    height_edges_nm: Array1D
    density_per_nm3: Array2D  # bins along the long side × bins along the profile axis
    frame: Frame
    axis_centered: bool  # the shape axis is put at x = 0 in each frame
    flipped: bool = False  # the map is mirrored z → −z (the liquid was at the upper wall), like the profile
    width_window_nm: float | None = Field(default=None, gt=0.0)  # only a strip around the vertical axis is counted
    frame_count: int = Field(ge=1)
    selection: str

    @model_validator(mode="after")
    def _check_shapes(self) -> "DensityMap":
        expected = (self.length_edges_nm.size - 1, self.height_edges_nm.size - 1)
        if self.density_per_nm3.shape != expected:
            raise ValueError(f"map has shape {self.density_per_nm3.shape}, the bin edges give {expected}")
        return self

    def bulk_density(self, core_fraction: float = BULK_CORE_FRACTION, level: float = BULK_LEVEL) -> float:
        """Bulk density of the mapped liquid, nm⁻³: the median over the core of its phase.
        The core is the middle `core_fraction` of the map height (away from the layering at the walls) and, in it,
        the bins denser than `level` × the densest bin there. Gives ρ_bulk for the fit (`density_bulk_per_nm3`) of a
        liquid that crosses the middle of the slit (a roll, a layer, the continuous phase around another liquid). A
        droplet or a worm on one wall does not reach the middle, and the map averages it over the short side: take
        ρ_bulk from the other phase or from a bulk simulation.
        """
        if not 0.0 < core_fraction <= 1.0 or not 0.0 <= level < 1.0:
            raise ValueError(
                f"invalid core_fraction = {core_fraction} or level = {level}; "
                "need 0 < core_fraction ≤ 1 and 0 ≤ level < 1"
            )
        heights = 0.5 * (self.height_edges_nm[:-1] + self.height_edges_nm[1:])
        middle = 0.5 * (self.height_edges_nm[0] + self.height_edges_nm[-1])
        half_band = 0.5 * core_fraction * (self.height_edges_nm[-1] - self.height_edges_nm[0])
        band = self.density_per_nm3[:, np.abs(heights - middle) <= half_band]
        if band.size == 0 or float(np.max(band)) <= 0.0:
            raise ValueError("no liquid in the middle of the map; increase core_fraction or check the selection")
        return float(np.median(band[band > level * float(np.max(band))]))

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
    """Density map of the selection in the “long side × profile axis” plane.

    shape is the name of a pore shape with a symmetry axis (`roll`, `doughnut`, `perforation`): then the shape axis
    is put at x = 0 in each frame (axial centering); without shape the coordinate along the long side is as in the
    trajectory. The profile axis is centred and, if needed, mirrored the same way as in `density_profile`.
    width_window_nm — only for a vertical axis: particles no farther than half the window from the axis along the
    short side are counted.
    """
    centering = resolve_centering(centering, pore, pore_center_nm)
    orientation = Orientation(orientation)
    grouping = ParticleGrouping(per)
    if min(bins) < MIN_MAP_BINS:
        raise ValueError(f"too few bins: {bins}; at least {MIN_MAP_BINS} are needed along each axis")
    axis = None
    if shape is not None:
        axis = pore.interface(shape).shape.symmetry_axis
        if axis is None:
            raise NotDerivedError(f"shape '{shape}' has no symmetry axis; axial centering is not defined")
    if width_window_nm is not None and (axis is None or axis.orientation is not AxisOrientation.VERTICAL):
        raise ValueError("width_window_nm is a strip around a vertical axis; set shape to a shape with a vertical axis")
    liquid = trajectory.select(select)
    groups: list[IndexArray] = [liquid]
    if centering is ProfileCentering.PORE and pore_center_nm is None:
        if pore.wall_selection is None:
            raise InvalidTrajectoryError(
                "pore centering needs the pore centre; set pore_center_nm or wall_selection of the pore"
            )
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
        check_box_change(frame.index, length, first_length, False, warned, "along the long side")
        if centering is ProfileCentering.LIQUID:
            check_box_change(frame.index, period, first_height, False, warned)
        elif "outside" not in warned and selected.any():
            in_pore = (heights[selected] >= height_edges[0]) & (heights[selected] < height_edges[-1])
            outside = 1.0 - float(np.count_nonzero(in_pore)) / int(np.count_nonzero(selected))
            if outside > OUTSIDE_FRACTION_WARNING:
                warned.add("outside")
                warnings.warn(
                    f"{outside:.1%} of the selected particles are outside the pore; check the selection and the walls",
                    PandaWarning,
                    stacklevel=2,
                )
        check_area(pore, length * width, warned)
        strip = width if width_window_nm is None else min(width_window_nm, width)
        total += counts / (np.outer(np.diff(length_edges), np.diff(height_edges)) * strip)
        frame_count += 1
    assert length_edges is not None and height_edges is not None and total is not None  # there are frames
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
        selection=select if isinstance(select, str) else f"{len(liquid)} indices",
    )
