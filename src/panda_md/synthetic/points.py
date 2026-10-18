"""Synthetic trajectory from point clouds: a shape in a slit → liquid particles and wall atoms per frame.

Liquid particles fill the three-dimensional shape (`contains`) uniformly at density ρ_bulk; in each frame the shape
axis is shifted randomly in the wall plane (as a droplet or roll wanders in MD), and the slit crosses the box boundary
along the profile axis. This is the input for the end-to-end check "shape → trajectory → centering → profile → fit"
and for 2D maps.
"""

from collections.abc import Mapping

import numpy as np

from panda_md.core.enums import ParameterName
from panda_md.core.errors import NotDerivedError
from panda_md.core.model import InterfaceModel
from panda_md.core.types import FloatArray
from panda_md.pores.slit.pore import SlitPore
from panda_md.trajectory.arrays import ArrayTrajectory

WALL_SPACING_NM = 0.25  # lattice spacing of the substrate atoms
OVERSAMPLING = 1.3  # more than 1/φ candidates per accepted particle — fewer repeated attempts
MAX_ATTEMPTS = 100


def synthetic_trajectory(
    model: InterfaceModel,
    values: Mapping[str, float],
    *,
    density_bulk_per_nm3: float,
    frame_count: int = 10,
    time_step_ps: float = 10.0,
    substrate_nm: float = 1.0,
    seed: int | None = None,
) -> ArrayTrajectory:
    """Return a slit trajectory with the shape of `model` (user units in values): groups "liquid" and "walls".

    The box is L × kL × (H + substrate_nm); the substrate occupies z ∈ [0, substrate_nm), its periodic image on top
    closes the slit. The number of liquid particles N = φ ρ_bulk V is constant; in each frame the shape axis is at a
    random point of the wall plane (the true positions are the trajectory's axis_positions_nm, for checking the
    axis search).
    """
    if substrate_nm <= 0.0:
        raise ValueError(f"substrate thickness is not positive: substrate_nm = {substrate_nm}; pass substrate_nm > 0")
    if frame_count < 1:
        raise ValueError(f"no frames requested: frame_count = {frame_count}; pass frame_count >= 1")
    pore = model.pore
    if not isinstance(pore, SlitPore) or not model.shape.has_contains():
        raise NotDerivedError(f"{model}: a point cloud is derived only for slit shapes that define contains")
    internal = model.supported_values(values)
    geometry = pore.geometry()
    if not model.shape.feasible(geometry, internal):
        raise NotDerivedError(f"{model}: the shape does not exist at {dict(values)}")
    generator = np.random.default_rng(seed)
    width = pore.width_nm
    box = np.array([pore.length_nm, pore.base_aspect_ratio * pore.length_nm, width + substrate_nm])
    walls = _substrate(box, substrate_nm)
    volume_fraction = internal[ParameterName.VOLUME_FRACTION]
    count = round(volume_fraction * density_bulk_per_nm3 * pore.volume_nm3())
    cell = np.array([geometry.pore_length_ratio, geometry.short_side_ratio(), 1.0])
    frames = []
    axes = []
    for _ in range(frame_count):
        points = _sample_inside(model, internal, cell, count, volume_fraction, generator)
        axis = generator.uniform(0.0, 1.0, size=2) * box[:2]
        axes.append(axis)
        liquid = points * width + np.array([axis[0], axis[1], substrate_nm + width / 2.0])
        frames.append(np.mod(np.vstack([walls, liquid]), box))
    return ArrayTrajectory(
        positions_nm=np.array(frames),
        boxes_nm=np.tile(box, (frame_count, 1)),
        times_ps=time_step_ps * np.arange(frame_count),
        groups={"walls": np.arange(len(walls)), "liquid": len(walls) + np.arange(count)},
        axis_positions_nm=np.array(axes),
    )


def _substrate(box: FloatArray, substrate_nm: float) -> FloatArray:
    """Return the substrate atom lattice: layers every WALL_SPACING_NM from z = 0 to z = substrate_nm (both faces)."""
    xs = np.arange(0.0, box[0], WALL_SPACING_NM)
    ys = np.arange(0.0, box[1], WALL_SPACING_NM)
    zs = np.linspace(0.0, substrate_nm, max(2, round(substrate_nm / WALL_SPACING_NM) + 1))
    grid = np.array(np.meshgrid(xs, ys, zs, indexing="ij")).reshape(3, -1).T
    walls: FloatArray = grid
    return walls


def _sample_inside(
    model: InterfaceModel,
    internal: Mapping[str, float],
    cell: FloatArray,
    count: int,
    volume_fraction: float,
    generator: np.random.Generator,
) -> FloatArray:
    """Return count points uniformly distributed in the liquid (dimensionless, shape axis at the origin)."""
    geometry = model.pore.geometry()
    accepted: list[FloatArray] = []
    total = 0
    for _ in range(MAX_ATTEMPTS):
        batch = max(1000, int(OVERSAMPLING * (count - total) / max(volume_fraction, 1e-6)))
        candidates = generator.uniform(-0.5, 0.5, size=(batch, 3)) * cell
        inside = candidates[model.shape.contains(candidates, geometry, dict(internal))]
        accepted.append(inside)
        total += len(inside)
        if total >= count:
            points: FloatArray = np.vstack(accepted)[:count]
            return points
    raise NotDerivedError(f"{model}: could not place {count} points in the shape in {MAX_ATTEMPTS} attempts")
