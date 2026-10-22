"""Axial centering: the symmetry axis of a shape in the plane of the walls and how to find its position in a frame.

Centering along the profile axis (pore centering or liquid centering) — `trajectory/density.py`. This is the second,
independent part: a shift in the plane of the walls so that the shape axis sits at the origin. It does not affect the
one-dimensional slit profile; it is needed for 2D maps and point clouds. The axis is declared in the shape class
(`InterfaceShape.symmetry_axis`).

The position is found from the particle histogram in the plane of the walls (for a horizontal axis — along the long
side), periodically smoothed with a Gaussian of width L / 16: the most pronounced extremum is taken — a liquid maximum
or a void minimum (then the liquid centre is L/2 further) — and refined by a weighted mean around it. A plain circular
mean of all particles loses the axis when the liquid almost fills the cell (a small perforation hole, a wide roll): the
mean vector is then of the order of the sampling noise.
"""

import warnings

import numpy as np

from panda_md.core.enums import AxisLocator, AxisOrientation
from panda_md.core.errors import PandaWarning
from panda_md.core.types import FloatArray, PandaModel

LATERAL_BINS = 64  # histogram bins along a side of the cell
SMOOTHING_FRACTION = 1.0 / 16.0  # smoothing σ as a fraction of the cell side
REGION_LEVEL = 0.5  # boundary of the region around the extremum — half its height (full width at half maximum)
WEAK_CONTRAST = 0.3  # contrast of the smoothed histogram below 30 % of the mean — the shape is almost uniform


class SymmetryAxis(PandaModel):
    orientation: AxisOrientation
    locate: AxisLocator


def _smooth_periodic(counts: FloatArray, sigma_bins: float) -> FloatArray:
    """Periodic Gaussian smoothing along each axis of the array (convolution via FFT)."""
    smoothed = counts.astype(np.float64)
    for axis_index, size in enumerate(smoothed.shape):
        offsets = np.fft.fftfreq(size) * size  # 0, 1, …, −1 in bins
        kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
        kernel /= kernel.sum()
        shape = [1] * smoothed.ndim
        shape[axis_index] = size
        transformed = np.fft.fft(smoothed, axis=axis_index) * np.fft.fft(kernel).reshape(shape)
        smoothed = np.real(np.fft.ifft(transformed, axis=axis_index))
    result: FloatArray = smoothed
    return result


def _region_center(deviation: FloatArray, index: int, period: float) -> float | None:
    """Centre along one axis: weighted mean over the connected region above half the extremum around index.
    For a symmetric shape the region is symmetric, and its midpoint is the axis for any shape width: both for a
    narrow dip (a small hole) and for a wide plateau (a shape of half a cell), where the extremum sits at a random
    point and a fixed-width window around it did not find the axis (pre-release audit). None — the region covers the
    whole axis (the row does not cross the shape), the centre along it is undefined.
    """
    size = deviation.size
    level = REGION_LEVEL * float(deviation[index])  # region above half the extremum: distant noise does not enter it
    left = 0
    while left < size - 1 and deviation[(index - left - 1) % size] > level:
        left += 1
    right = 0
    while left + right < size - 1 and deviation[(index + right + 1) % size] > level:
        right += 1
    if left + right >= size - 1:
        return None
    offsets = np.arange(-left, right + 1)
    weights = np.clip(deviation[(index + offsets) % size] - level, 0.0, None)
    total = float(np.sum(weights))
    shift = float(np.sum(weights * offsets)) / total if total > 0.0 else 0.0
    return ((index + 0.5 + shift) * period / size) % period


def structure_center(values: FloatArray, periods: FloatArray) -> tuple[FloatArray, float]:
    """Liquid centre (along each axis, in [0, L)) and the contrast of the smoothed histogram divided by the mean.
    values — particle coordinates (n × d, d = 1 or 2), periods — cell sides (d). The extremum of the smoothed
    histogram is an excess (liquid) or a deficit (void: perforation, liquid in almost the whole cell); along each
    axis the centre is the midpoint of the connected deviation region along the row through the extremum, and if the
    row lies entirely in the region — the midpoint of the region of the marginal distribution along this axis. A void
    centre is converted to the liquid centre (+ L/2).
    """
    dimensions = values.shape[1]
    counts, _ = np.histogramdd(
        np.mod(values, periods), bins=LATERAL_BINS, range=[(0.0, float(period)) for period in periods]
    )
    smoothed = _smooth_periodic(counts, SMOOTHING_FRACTION * LATERAL_BINS)
    mean = float(np.mean(smoothed))
    if mean == 0.0:
        return np.zeros(dimensions), 0.0
    excess, deficit = float(np.max(smoothed)) - mean, mean - float(np.min(smoothed))
    liquid_peak = excess >= deficit
    deviation = smoothed - mean if liquid_peak else mean - smoothed
    peak = np.unravel_index(int(np.argmax(deviation)), deviation.shape)
    center = np.empty(dimensions)
    for axis_index in range(dimensions):
        period = float(periods[axis_index])
        index: list[int | slice] = [int(position) for position in peak]
        index[axis_index] = slice(None)
        found = _region_center(deviation[tuple(index)], int(peak[axis_index]), period)
        if found is None:
            other_axes = tuple(axis for axis in range(dimensions) if axis != axis_index)
            marginal = deviation.sum(axis=other_axes)
            marginal = marginal - float(np.mean(marginal))
            found = _region_center(marginal, int(np.argmax(marginal)), period)
        center[axis_index] = found if found is not None else 0.0
    if not liquid_peak:
        center = np.mod(center + 0.5 * periods, periods)  # void centre → liquid centre
    return center, max(excess, deficit) / mean


def axis_position(lateral_positions: FloatArray, lateral_periods: FloatArray, axis: SymmetryAxis) -> FloatArray:
    """Axis position in the plane of the walls: (along the long side, along the short side), in [0, L) along each.

    lateral_positions — particle coordinates (n × 2): along the long side of the cell and along the short one. For a
    horizontal axis (along the short side) the position along the short side is undefined and is returned as 0.
    """
    if axis.orientation is AxisOrientation.HORIZONTAL:
        along, contrast = structure_center(lateral_positions[:, :1], lateral_periods[:1])
        _, across_contrast = structure_center(lateral_positions[:, 1:], lateral_periods[1:])
        if across_contrast > contrast:
            warnings.warn(
                "shape is less uniform along the short side than along the long one: the axis seems to lie along the "
                "long side, so a map along the long side will not show it (the profile does not depend on this)",
                PandaWarning,
                stacklevel=3,
            )
        position = np.array([along[0], 0.0])
    else:
        position, contrast = structure_center(lateral_positions, lateral_periods)
    if contrast < WEAK_CONTRAST:
        warnings.warn(
            "shape is almost uniform in the plane of the walls: the axis position is poorly determined",
            PandaWarning,
            stacklevel=3,
        )
    if axis.locate is AxisLocator.VOID_CENTER:
        shift = 0.5 * lateral_periods
        if axis.orientation is AxisOrientation.HORIZONTAL:
            shift = np.array([shift[0], 0.0])
        position = np.mod(position + shift, lateral_periods)
    centered: FloatArray = position
    return centered
