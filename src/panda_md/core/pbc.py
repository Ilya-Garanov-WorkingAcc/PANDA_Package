"""Periodic boundary conditions along one box axis: wrapping, circular mean, largest empty gap."""

import math
import warnings

import numpy as np

from panda_md.core.errors import InvalidTrajectoryError, PandaWarning
from panda_md.core.types import FloatArray, IndexArray

LONG_MOLECULE_FRACTION = 0.45  # atom farther than 0.45 L from the reference atom — molecule likely longer than L/2


def wrap_centered(values: FloatArray, period: float | FloatArray) -> FloatArray:
    """Wrap coordinates into [−L/2, L/2) — the nearest image relative to zero (L is a number or one per column)."""
    wrapped: FloatArray = values - period * np.floor(values / period + 0.5)
    return wrapped


def circular_mean(values: FloatArray, period: float) -> float:
    """Centre of a point cloud on a circle of length L: the mean direction, result in [0, L).

    This way the centre of a liquid that crosses the box boundary does not land in the middle of the void (as in
    the old PANDA, `utils.get_center_pbc`).
    """
    if values.size == 0:
        raise InvalidTrajectoryError("centre of an empty selection is undefined; check the selection")
    angles = 2.0 * math.pi * values / period
    mean_angle = math.atan2(-float(np.mean(np.sin(angles))), -float(np.mean(np.cos(angles)))) + math.pi
    center = (period * mean_angle / (2.0 * math.pi)) % period
    return center if center < period else 0.0  # −1e−17 % L gives exactly L


def largest_gap(values: FloatArray, period: float) -> tuple[float, float]:
    """Largest gap without points on a circle of length L: (start, end), end > start, end < start + L.

    The slit walls are substrate atoms: the pore is the widest empty gap between them, including one across the
    box boundary (one substrate and its image).
    """
    return empty_gaps(values, period)[0]


def empty_gaps(values: FloatArray, period: float) -> list[tuple[float, float]]:
    """All gaps between neighbouring points on a circle of length L, from widest to narrowest: (start, end)."""
    if values.size == 0:
        raise InvalidTrajectoryError("wall selection contains no atoms; check the wall selection")
    ordered = np.sort(np.mod(values, period))
    widths = np.diff(np.append(ordered, ordered[0] + period))
    order = np.argsort(-widths, kind="stable")
    return [(float(ordered[index]), float(ordered[index] + widths[index])) for index in order]


def molecule_centers(
    positions: FloatArray, box: FloatArray, molecule_indices: IndexArray, masses: FloatArray
) -> FloatArray:
    """Centres of mass of the molecules (n_molecules × 3), inside the box. The atoms of each molecule are first
    gathered to one of its atoms by the nearest image — a molecule cut by the box boundary is not 'smeared' over
    the whole box. A molecule longer than half the box cannot be gathered this way — a warning.

    molecule_indices — the molecule index of each atom (0 … n_molecules − 1, atoms of one molecule anywhere).
    """
    count = int(np.max(molecule_indices)) + 1
    if np.any(np.bincount(molecule_indices, minlength=count) == 0):
        raise InvalidTrajectoryError("molecule indices have gaps; they must run 0 … n − 1 without gaps (see renumber)")
    first_atom = np.full(count, -1, dtype=np.int64)
    first_atom[molecule_indices[::-1]] = np.arange(molecule_indices.size)[::-1]
    reference = positions[first_atom[molecule_indices]]
    offsets = wrap_centered(positions - reference, box)
    if np.any(np.abs(offsets) > LONG_MOLECULE_FRACTION * box):
        warnings.warn(
            "molecule is longer than half the box: the centre of mass by the nearest image may be wrong",
            PandaWarning,
            stacklevel=2,
        )
    unwrapped = reference + offsets
    total_mass = np.bincount(molecule_indices, weights=masses, minlength=count)
    if np.any(total_mass <= 0.0):
        raise InvalidTrajectoryError("molecule has zero mass; check the atom masses")
    centers = np.column_stack(
        [np.bincount(molecule_indices, weights=masses * unwrapped[:, axis], minlength=count) for axis in range(3)]
    )
    wrapped: FloatArray = np.mod(centers / total_mass[:, np.newaxis], box)
    return wrapped
