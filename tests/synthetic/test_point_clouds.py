"""Point clouds: shape in a slit → trajectory with a wandering shape → walls, centering, profile, fit and 2D maps.

Specification criterion (§9): “point cloud → full data path → fit” — |Δθ| < 2°.
"""

from pathlib import Path

import numpy as np
import pytest

from panda_md import ArrayTrajectory, InterfaceModel, SlitPore, SquarePore, density_profile
from panda_md.core.centering import axis_position
from panda_md.core.errors import NotDerivedError, PandaWarning
from panda_md.core.types import FloatArray
from panda_md.synthetic.points import synthetic_trajectory
from panda_md.trajectory.maps import DensityMap, density_map

DENSITY_BULK = 33.0
PORE = SlitPore(width_nm=4.0, length_nm=8.0, base_aspect_ratio=0.5)
MAX_ANGLE_ERROR_DEG = 2.0


@pytest.mark.parametrize(
    ("shape_name", "truth"),
    [
        ("roll", {"theta": 120.0, "delta": 0.03, "phi": 0.4}),
        ("droplet", {"theta": 130.0, "delta": 0.03, "phi": 0.05}),
        ("doughnut", {"theta": 130.0, "delta": 0.03, "phi": 0.2}),
    ],
)
def test_angle_recovered_from_point_clouds(shape_name: str, truth: dict[str, float]) -> None:
    model = PORE.interface(shape_name)
    trajectory = synthetic_trajectory(model, truth, density_bulk_per_nm3=DENSITY_BULK, frame_count=100, seed=1)
    found = SlitPore.from_trajectory(trajectory, walls="walls")
    assert found.width_nm == pytest.approx(PORE.width_nm, abs=1e-9)
    assert found.base_aspect_ratio == pytest.approx(PORE.base_aspect_ratio)
    profiles = density_profile(trajectory, found, "liquid", bins=100)
    result = found.interface(shape_name).fit(profiles.mean(), density_bulk_per_nm3=DENSITY_BULK)
    assert result.values["theta"] == pytest.approx(truth["theta"], abs=MAX_ANGLE_ERROR_DEG)
    assert result.values["phi"] == pytest.approx(truth["phi"], rel=0.03)


def mirror_agreement(density: np.ndarray) -> float:
    """Correlation of the map with its mirror x → −x: 1 means the shape is symmetric about the axis at x = 0."""
    return float(np.corrcoef(density.ravel(), density[::-1].ravel())[0, 1])


def region_mean(density_map_: DensityMap, length_window: tuple[float, float], height_limit: float) -> float:
    """Mean map density in the band |z| < height_limit and length_window[0] ≤ |x| < length_window[1] (nm)."""
    along = 0.5 * (density_map_.length_edges_nm[:-1] + density_map_.length_edges_nm[1:])
    height = 0.5 * (density_map_.height_edges_nm[:-1] + density_map_.height_edges_nm[1:])
    rows = (np.abs(along) >= length_window[0]) & (np.abs(along) < length_window[1])
    columns = np.abs(height) < height_limit
    return float(np.mean(density_map_.density_per_nm3[np.ix_(rows, columns)]))


def test_axis_centering_sharpens_the_map_of_a_wandering_roll() -> None:
    roll = PORE.interface("roll")
    trajectory = synthetic_trajectory(
        roll, {"theta": 120.0, "delta": 0.03, "phi": 0.3}, density_bulk_per_nm3=DENSITY_BULK, frame_count=20, seed=2
    )
    found = SlitPore.from_trajectory(trajectory, walls="walls")
    centered = density_map(trajectory, found, "liquid", shape="roll", bins=(40, 40))
    smeared = density_map(trajectory, found, "liquid", bins=(40, 40))
    assert centered.axis_centered and not smeared.axis_centered
    assert mirror_agreement(centered.density_per_nm3) > 0.95
    assert region_mean(centered, (0.0, 0.5), 1.0) == pytest.approx(DENSITY_BULK, rel=0.1)  # roll core
    assert region_mean(centered, (3.5, 4.0), 1.0) == pytest.approx(0.0, abs=1.0)  # no liquid at the cell edge
    assert np.max(smeared.density_per_nm3) < 0.8 * np.max(centered.density_per_nm3)


def test_perforation_axis_is_the_center_of_the_hole() -> None:
    perforation = PORE.interface("perforation")
    trajectory = synthetic_trajectory(
        perforation,
        {"theta": 130.0, "delta": 0.03, "phi": 0.8},
        density_bulk_per_nm3=DENSITY_BULK,
        frame_count=50,
        seed=3,
    )
    found = SlitPore.from_trajectory(trajectory, walls="walls")
    centered = density_map(trajectory, found, "liquid", shape="perforation", bins=(40, 40))
    hole = region_mean(centered, (0.0, 0.5), 1.0)
    bulk = region_mean(centered, (3.0, 4.0), 1.0)
    assert bulk == pytest.approx(DENSITY_BULK, rel=0.1)
    assert hole < 0.6 * bulk


def test_map_needs_a_shape_with_an_axis_and_survives_saving(tmp_path: Path) -> None:
    roll = PORE.interface("roll")
    trajectory = synthetic_trajectory(
        roll, {"theta": 120.0, "delta": 0.03, "phi": 0.3}, density_bulk_per_nm3=DENSITY_BULK, frame_count=2, seed=4
    )
    found = SlitPore.from_trajectory(trajectory, walls="walls")
    with pytest.raises(NotDerivedError, match="no symmetry axis"):
        density_map(trajectory, found, "liquid", shape="layer")
    density = density_map(trajectory, found, "liquid", shape="roll", bins=(10, 12))
    density.save(tmp_path / "map.npz")
    assert DensityMap.load(tmp_path / "map.npz").model_dump_json() == density.model_dump_json()


def test_point_clouds_exist_only_for_slit_shapes() -> None:
    square = SquarePore(side_nm=6.0, length_nm=12.0, layer_width_nm=6.0)
    with pytest.raises(NotDerivedError, match="only for slit shapes"):
        synthetic_trajectory(
            square.interface("droplet_w1c0"), {"theta": 140.0, "delta": 0.0, "phi": 0.05}, density_bulk_per_nm3=30.0
        )


@pytest.mark.parametrize(
    ("shape_name", "values"),
    [
        ("perforation", {"theta": 130.0, "delta": 0.03, "phi": 0.92}),  # small hole (φ_max = 0.932)
        ("roll", {"theta": 120.0, "delta": 0.03, "phi": 0.85}),  # the roll almost fills the cell
    ],
)
def test_axis_is_found_when_the_liquid_almost_fills_the_cell(shape_name: str, values: dict[str, float]) -> None:
    model = PORE.interface(shape_name)
    axis = model.shape.symmetry_axis
    assert axis is not None
    for seed in range(10):
        trajectory, true_positions = synthetic_trajectory_with_axes(model, values, seed)
        liquid = trajectory.positions_nm[0][trajectory.select("liquid")]
        found = axis_position(liquid[:, :2], trajectory.boxes_nm[0][:2], axis)[0]
        error = abs((found - true_positions[0] + 0.5 * PORE.length_nm) % PORE.length_nm - 0.5 * PORE.length_nm)
        assert error < 0.15, (seed, found, true_positions[0])


def synthetic_trajectory_with_axes(
    model: InterfaceModel, values: dict[str, float], seed: int
) -> tuple[ArrayTrajectory, FloatArray]:
    trajectory = synthetic_trajectory(model, values, density_bulk_per_nm3=DENSITY_BULK, frame_count=1, seed=seed)
    return trajectory, trajectory_axes(trajectory)


def trajectory_axes(trajectory: ArrayTrajectory) -> FloatArray:
    assert trajectory.axis_positions_nm is not None
    axes: FloatArray = trajectory.axis_positions_nm[0]
    return axes


def test_roll_lying_along_the_long_side_is_reported() -> None:
    square = SlitPore(width_nm=4.0, length_nm=6.0)  # k = 1: the roll may lie along either side
    roll = square.interface("roll")
    trajectory = synthetic_trajectory(
        roll, {"theta": 120.0, "delta": 0.03, "phi": 0.4}, density_bulk_per_nm3=DENSITY_BULK, frame_count=2, seed=5
    )
    rotated = ArrayTrajectory(
        positions_nm=trajectory.positions_nm[:, :, [1, 0, 2]],
        boxes_nm=trajectory.boxes_nm[:, [1, 0, 2]],
        times_ps=trajectory.times_ps(),
        groups=trajectory.groups,
    )
    found = SlitPore.from_trajectory(rotated, walls="walls")
    with pytest.warns(PandaWarning) as caught:  # the roll is uniform along the long side — this is reported too
        density_map(rotated, found, "liquid", shape="roll", bins=(20, 20))
    assert any("axis seems to lie along the long side" in str(warning.message) for warning in caught)


def test_map_reports_a_changing_box_and_wrong_bins() -> None:
    roll = PORE.interface("roll")
    trajectory = synthetic_trajectory(
        roll, {"theta": 120.0, "delta": 0.03, "phi": 0.3}, density_bulk_per_nm3=DENSITY_BULK, frame_count=2, seed=6
    )
    boxes = trajectory.boxes_nm.copy()
    boxes[1, 0] *= 1.2
    grown = ArrayTrajectory(
        positions_nm=trajectory.positions_nm, boxes_nm=boxes, times_ps=trajectory.times_ps(), groups=trajectory.groups
    )
    found = SlitPore.from_trajectory(trajectory, walls="walls")
    with pytest.warns(PandaWarning) as caught:  # the base area changes too — also a warning
        density_map(grown, found, "liquid", bins=(20, 20))
    messages = [str(warning.message) for warning in caught]
    assert any("the box along the long side" in message for message in messages)
    assert not any("outside the pore" in message for message in messages)  # all particles are inside the pore in height
    with pytest.raises(ValueError, match="too few bins"):
        density_map(trajectory, found, "liquid", bins=(0, 20))


def test_map_integral_is_the_particle_count_and_follows_profile_orientation() -> None:
    worm = PORE.interface("worm")
    trajectory = synthetic_trajectory(
        worm, {"theta": 130.0, "delta": 0.03, "phi": 0.1}, density_bulk_per_nm3=DENSITY_BULK, frame_count=3, seed=7
    )
    flipped_positions = trajectory.positions_nm.copy()
    flipped_positions[:, :, 2] = np.mod(
        -flipped_positions[:, :, 2], trajectory.boxes_nm[0, 2]
    )  # worm — on the upper wall
    upside_down = ArrayTrajectory(
        positions_nm=flipped_positions,
        boxes_nm=trajectory.boxes_nm,
        times_ps=trajectory.times_ps(),
        groups=trajectory.groups,
    )
    found = SlitPore.from_trajectory(upside_down, walls="walls")
    density = density_map(upside_down, found, "liquid", bins=(40, 40))
    assert density.flipped
    cell = np.outer(np.diff(density.length_edges_nm), np.diff(density.height_edges_nm)) * PORE.base_aspect_ratio * 8.0
    assert float(np.sum(density.density_per_nm3 * cell)) == pytest.approx(len(trajectory.select("liquid")))
    profile = density_profile(upside_down, found, "liquid", bins=40).mean()
    np.testing.assert_allclose(density.density_per_nm3.mean(axis=0), profile.density_per_nm3, rtol=1e-9)


def test_map_with_liquid_centering_per_molecule_and_axis_window() -> None:
    doughnut = PORE.interface("doughnut")
    trajectory = synthetic_trajectory(
        doughnut, {"theta": 130.0, "delta": 0.03, "phi": 0.2}, density_bulk_per_nm3=DENSITY_BULK, frame_count=5, seed=8
    )
    atom_count = trajectory.positions_nm.shape[1]
    molecular = ArrayTrajectory(
        positions_nm=trajectory.positions_nm,
        boxes_nm=trajectory.boxes_nm,
        times_ps=trajectory.times_ps(),
        groups=trajectory.groups,
        molecule_ids=np.arange(atom_count),
        masses=np.ones(atom_count),
    )
    found = SlitPore.from_trajectory(molecular, walls="walls")
    liquid_frame = density_map(molecular, found, "liquid", shape="doughnut", centering="liquid", per="molecule")
    assert liquid_frame.frame.value == "liquid"
    whole = density_map(molecular, found, "liquid", shape="doughnut", bins=(40, 40))
    sliced = density_map(molecular, found, "liquid", shape="doughnut", bins=(40, 40), width_window_nm=0.8)
    assert sliced.density_per_nm3[20, 20] > 1.3 * whole.density_per_nm3[20, 20]  # slice through the axis, no averaging
    with pytest.raises(NotDerivedError, match="no symmetry axis"):
        density_map(molecular, found, "liquid", shape="layer", bins=(10, 10))


def test_droplet_map_with_axial_centering_is_symmetric() -> None:
    """Pre-release audit: the droplet had no axis — a droplet map could not be aligned across frames."""
    droplet = PORE.interface("droplet")
    trajectory = synthetic_trajectory(
        droplet, {"theta": 130.0, "delta": 0.03, "phi": 0.05}, density_bulk_per_nm3=DENSITY_BULK, frame_count=20, seed=5
    )
    found = SlitPore.from_trajectory(trajectory, walls="walls")
    density = density_map(trajectory, found, "liquid", shape="droplet", bins=(40, 20)).density_per_nm3
    np.testing.assert_allclose(density, density[::-1, :], atol=0.35 * DENSITY_BULK)  # symmetry x → −x
    assert float(np.max(density[18:22, :])) > 0.6 * DENSITY_BULK  # liquid at the axis, not a smeared cloud


@pytest.mark.parametrize(
    ("shape_name", "values"),
    [("roll", {"theta": 120.0, "delta": 0.03, "phi": 0.4}), ("layer", {"delta": 0.03, "phi": 0.4})],
)
def test_bulk_density_is_estimated_from_the_map(shape_name: str, values: dict[str, float]) -> None:
    """ρ_bulk for the fit straight from the 2D map: the core of the phase in the middle of the slit."""
    model = PORE.interface(shape_name)
    trajectory = synthetic_trajectory(model, values, density_bulk_per_nm3=DENSITY_BULK, frame_count=40, seed=6)
    found = SlitPore.from_trajectory(trajectory, walls="walls")
    axis = shape_name if model.shape.symmetry_axis is not None else None
    density = density_map(trajectory, found, "liquid", shape=axis, bins=(40, 20))
    assert density.bulk_density() == pytest.approx(DENSITY_BULK, rel=0.1)


def test_bulk_density_ignores_dense_layers_at_the_walls() -> None:
    edges = np.linspace(-2.0, 2.0, 21)
    rows = np.full((10, 20), 3.0)  # bulk liquid 3 nm⁻³
    rows[:, :3] = rows[:, -3:] = 16.0  # adsorbed layers, as decane on graphite
    density = DensityMap(length_edges_nm=np.linspace(0.0, 8.0, 11), height_edges_nm=edges, density_per_nm3=rows,
                         frame="pore", axis_centered=False, frame_count=1, selection="test")  # fmt: skip
    assert density.bulk_density() == pytest.approx(3.0)
    with pytest.raises(ValueError, match="core_fraction"):
        density.bulk_density(core_fraction=0.0)
