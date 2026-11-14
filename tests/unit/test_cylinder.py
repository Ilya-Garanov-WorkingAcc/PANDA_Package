"""Cylindrical pore and its capsule shape (formulas: alpha.py rollx)."""

import json
import math
import subprocess
import sys
import warnings
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.integrate import trapezoid

import panda_md as panda
from panda_md import CylinderPore, TrianglePore
from panda_md.cli import main
from panda_md.cli.options import CommandLineError, load_pore, pore_json
from panda_md.cli.pipeline import trajectory_pore
from panda_md.core.enums import Axis
from panda_md.core.errors import InvalidTrajectoryError, PandaWarning, UnsupportedRegimeError
from panda_md.core.pore import field_as_float
from panda_md.core.profile import DensityProfile
from panda_md.pores.cylinder.capsule import CylinderCapsule
from panda_md.pores.registry import pore_class
from panda_md.synthetic.profiles import synthetic_profile
from panda_md.trajectory.arrays import ArrayTrajectory
from panda_md.trajectory.density import density_profile

TUBE = {"diameter_nm": 8.0, "length_nm": 16.0, "layer_width_nm": 11.345}


def test_cylinder_pore_geometry() -> None:
    pore = CylinderPore(**TUBE)
    geometry = pore.geometry()
    assert geometry.pore_length_ratio == 2.0
    assert geometry.layer_width_ratio == pytest.approx(1.418125)
    assert geometry.coordinate_bounds() == (-0.5, 0.5)
    assert geometry.profile_integral() == pytest.approx(math.pi / 4.0 / 1.418125)
    assert geometry.layer_area() == pytest.approx(2.0 * 1.418125)
    assert pore.characteristic_size_nm() == 8.0
    assert pore.volume_nm3() == pytest.approx(math.pi * 64.0 * 16.0 / 4.0)
    assert pore.bin_cross_section_nm2(np.array([16.0, 11.345, 11.345])) == pytest.approx(16.0 * 11.345)
    assert pore.layer_axis is Axis.Y


def test_cylinder_pore_rejects_bad_dimensions() -> None:
    with pytest.raises(ValidationError, match="layer width"):
        CylinderPore(diameter_nm=8.0, length_nm=16.0, layer_width_nm=7.0)
    with pytest.raises(ValidationError, match="axis"):
        CylinderPore(**TUBE, axis="z")


def test_cylinder_pore_is_registered(tmp_path: Path) -> None:
    assert pore_class("cylinder") is CylinderPore
    assert panda.CylinderPore is CylinderPore
    pore = CylinderPore(**TUBE, wall_offset_nm=0.1, wall_selection="resname GRAPH*")
    path = tmp_path / "pore.json"
    path.write_text(pore_json(pore), encoding="utf-8")
    assert load_pore(None, path, []) == pore


CAPSULE_PORE = CylinderPore(diameter_nm=1.0, length_nm=3.0, layer_width_nm=1.42)


def capsule_values(theta: float, delta: float) -> dict[str, float]:
    """Mid-range φ of the feasible interval at this θ and δ."""
    lower, upper = CAPSULE_PORE.interface("capsule").phi_limits(theta=theta, delta=delta, phi=0.5)
    return {"theta": theta, "delta": delta, "phi": 0.5 * (max(lower, 0.0) + upper)}


@pytest.mark.parametrize("theta", [95.0, 120.0, 150.0, 179.0])
@pytest.mark.parametrize("delta", [0.0, 0.04])
def test_capsule_conserves_volume(theta: float, delta: float) -> None:
    model = CAPSULE_PORE.interface("capsule")
    values = capsule_values(theta, delta)
    grid = np.linspace(-0.5, 0.5, 200_001)
    integral = trapezoid(model.profile(grid, **values), grid)
    assert integral == pytest.approx(values["phi"] * CAPSULE_PORE.geometry().profile_integral(), rel=1e-4)


@pytest.mark.parametrize("theta", [100.0, 140.0, 175.0])
def test_capsule_touches_its_periodic_image_at_phi_max(theta: float) -> None:
    model = CAPSULE_PORE.interface("capsule")
    geometry = CAPSULE_PORE.geometry()
    _, upper = model.phi_limits(theta=theta, delta=0.02, phi=0.5)
    values = model.supported_values({"theta": theta, "delta": 0.02, "phi": upper - 1e-9})
    shape = CylinderCapsule()
    radius = shape.meniscus_radius(values)
    cap_height = radius * (1.0 - math.sin(math.radians(theta)))
    assert shape.plug_length(geometry, values) + 2.0 * cap_height == pytest.approx(3.0, abs=1e-6)


def test_capsule_wetting_regime_is_unsupported() -> None:
    model = CAPSULE_PORE.interface("capsule")
    with pytest.raises(UnsupportedRegimeError):
        model.profile(np.zeros(3), theta=80.0, delta=0.0, phi=0.3)
    assert not model.is_feasible(theta=80.0, delta=0.0, phi=0.3)


def test_capsule_contains_matches_profile() -> None:
    model = CAPSULE_PORE.interface("capsule")
    geometry = CAPSULE_PORE.geometry()
    values = model.supported_values(capsule_values(130.0, 0.03))
    generator = np.random.default_rng(3)
    count = 400_000
    points = np.column_stack(
        [
            (generator.random(count) - 0.5) * geometry.pore_length_ratio,
            (generator.random(count) - 0.5) * geometry.layer_width_ratio,
            generator.random(count) - 0.5,
        ]
    )
    inside = model.shape.contains(points, geometry, values)
    edges = np.linspace(-0.5, 0.5, 21)
    slab = np.digitize(points[:, 2], edges) - 1
    fine = np.linspace(-0.5, 0.5, 20_001)
    profile = model.shape.profile(fine, geometry, values)
    for index in range(20):
        in_slab = slab == index
        expected = profile[(fine >= edges[index]) & (fine < edges[index + 1])].mean()
        assert inside[in_slab].mean() == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize("theta", [140.0, 175.0])
def test_capsule_fit_recovers_parameters(theta: float) -> None:
    model = CAPSULE_PORE.interface("capsule")
    values = {"theta": theta, "delta": 0.03, "phi": capsule_values(theta, 0.03)["phi"]}
    profile = synthetic_profile(model, values, density_bulk_per_nm3=33.0)
    result = model.fit(profile, optimize=["theta", "delta"], fixed={"phi": values["phi"]}, density_bulk_per_nm3=33.0)
    assert result.values["theta"] == pytest.approx(theta, abs=0.1)
    assert result.values["delta"] == pytest.approx(0.03, abs=1e-3)


TUBE_BOX_NM = (8.0, 6.0, 6.0)
POINT_DENSITY_PER_NM3 = 3000.0  # sampling density of the synthetic liquid


def ring_walls(shells: tuple[float, ...], center: tuple[float, float], length: float) -> np.ndarray:
    """Wall atoms on rings around an axis along x: 60 atoms per ring, a ring every 0.25 nm, wrapped into the box."""
    angles = np.linspace(0.0, 2.0 * np.pi, 60, endpoint=False)
    rings = [
        np.column_stack([np.full(60, x), center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)])
        for radius in shells
        for x in np.arange(0.0, length, 0.25)
    ]
    return np.vstack(rings)


def capsule_liquid(values: dict[str, float], center: tuple[float, float], seed: int) -> np.ndarray:
    """Liquid points of a capsule in the tube of diameter 4 nm: uniform points kept where the capsule contains them."""
    pore = CylinderPore(diameter_nm=4.0, length_nm=TUBE_BOX_NM[0], layer_width_nm=TUBE_BOX_NM[1])
    model = pore.interface("capsule")
    generator = np.random.default_rng(seed)
    count = int(POINT_DENSITY_PER_NM3 * TUBE_BOX_NM[0] * 4.0 * 4.0)
    points = np.column_stack([generator.random(count) * TUBE_BOX_NM[0], (generator.random((count, 2)) - 0.5) * 4.0])
    reduced = np.column_stack([points[:, 0] - 0.5 * TUBE_BOX_NM[0], points[:, 1:]]) / 4.0
    inside = model.shape.contains(reduced, pore.geometry(), model.supported_values(values))
    liquid = points[inside]
    liquid[:, 1] += center[0]
    liquid[:, 2] += center[1]
    return liquid


def cylinder_trajectory(
    shells: tuple[float, ...] = (2.0,),
    center: tuple[float, float] = (3.0, 3.0),
    liquid_values: dict[str, float] | None = None,
    frames: int = 1,
    liquid_outside: bool = False,
) -> ArrayTrajectory:
    """A tube along x (wall rings of the given radii) with liquid: a capsule, a uniform rod, or a rod outside."""
    walls = ring_walls(shells, center, TUBE_BOX_NM[0])
    liquids = []
    for seed in range(frames):
        if liquid_values is not None:
            liquids.append(capsule_liquid(liquid_values, center, seed))
            continue
        generator = np.random.default_rng(seed)
        radius = 1.6 * np.sqrt(generator.random(200))
        angle = 2.0 * np.pi * generator.random(200)
        offset = (0.0, 0.0) if not liquid_outside else (2.9, 2.9)
        liquids.append(
            np.column_stack(
                [
                    TUBE_BOX_NM[0] * generator.random(200),
                    center[0] + offset[0] + 0.3 * radius * np.cos(angle),
                    center[1] + offset[1] + 0.3 * radius * np.sin(angle),
                ]
            )
        )
    liquid_count = min(len(liquid) for liquid in liquids)
    positions = [np.vstack([walls, liquid[:liquid_count]]) % np.array(TUBE_BOX_NM) for liquid in liquids]
    return ArrayTrajectory(
        positions_nm=np.array(positions),
        boxes_nm=[TUBE_BOX_NM] * frames,
        times_ps=np.arange(frames, dtype=float),
        groups={"walls": np.arange(len(walls)), "liquid": np.arange(len(walls), len(walls) + liquid_count)},
    )


def test_tube_from_trajectory() -> None:
    pore = CylinderPore.from_trajectory(cylinder_trajectory(), walls="walls", liquid="liquid")
    assert pore.diameter_nm == pytest.approx(4.0, abs=1e-9)
    assert (pore.length_nm, pore.layer_width_nm) == (8.0, 6.0)
    assert pore.axis is Axis.X and pore.normal_axis is Axis.Z and pore.wall_selection == "walls"
    with_offset = CylinderPore.from_trajectory(cylinder_trajectory(), walls="walls", wall_offset_nm=0.1)
    assert with_offset.diameter_nm == pytest.approx(3.8, abs=1e-9)


def test_multishell_tube_uses_the_inner_shell() -> None:
    pore = CylinderPore.from_trajectory(cylinder_trajectory(shells=(2.0, 2.34, 2.68)), walls="walls")
    assert pore.diameter_nm == pytest.approx(4.0, abs=1e-9)


def test_tube_across_the_box_edge() -> None:
    values = {"theta": 130.0, "delta": 0.05, "phi": 0.4}
    trajectory = cylinder_trajectory(center=(0.3, 0.3), liquid_values=values)
    pore = CylinderPore.from_trajectory(trajectory, walls="walls", liquid="liquid")
    assert pore.diameter_nm == pytest.approx(4.0, abs=1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("error", PandaWarning)
        profile = density_profile(trajectory, pore, "liquid", bins=40).mean()
    center = np.sum(profile.coordinate_nm * profile.density_per_nm3) / np.sum(profile.density_per_nm3)
    assert center == pytest.approx(0.0, abs=0.05)


def test_slit_is_not_a_tube() -> None:
    grid = np.stack(np.meshgrid(np.arange(0.0, 8.0, 0.25), np.arange(0.0, 6.0, 0.25)), axis=-1).reshape(-1, 2)
    planes = np.vstack([np.column_stack([grid, np.full(len(grid), z)]) for z in (1.0, 5.0)])
    liquid = np.column_stack([np.full(50, 4.0), np.full(50, 3.0), np.linspace(2.0, 4.0, 50)])
    slit = ArrayTrajectory(
        positions_nm=np.vstack([planes, liquid])[np.newaxis],
        boxes_nm=[TUBE_BOX_NM],
        times_ps=[0.0],
        groups={"walls": np.arange(len(planes)), "liquid": np.arange(len(planes), len(planes) + 50)},
    )
    with pytest.raises(InvalidTrajectoryError, match="not a tube"):
        CylinderPore.from_trajectory(slit, walls="walls")
    with pytest.raises(InvalidTrajectoryError, match="not a tube"):
        CylinderPore.from_trajectory(cylinder_trajectory(liquid_outside=True), walls="walls", liquid="liquid")
    with pytest.raises(InvalidTrajectoryError, match="axis"):
        CylinderPore.from_trajectory(cylinder_trajectory(), walls="walls", axis="z")


def test_capsule_profile_from_a_tube_trajectory() -> None:
    values = {"theta": 130.0, "delta": 0.05, "phi": 0.4}
    trajectory = cylinder_trajectory(liquid_values=values, frames=3)
    pore = CylinderPore.from_trajectory(trajectory, walls="walls", liquid="liquid")
    with warnings.catch_warnings():
        warnings.simplefilter("error", PandaWarning)
        profile = density_profile(trajectory, pore, "liquid", bins=40).mean()
    result = pore.interface("capsule").fit(
        profile, optimize=["theta", "delta", "phi"], density_bulk_per_nm3=POINT_DENSITY_PER_NM3
    )
    assert result.values["theta"] == pytest.approx(130.0, abs=5.0)
    assert result.values["phi"] == pytest.approx(0.4, abs=0.03)


def test_command_line_measures_a_tube_without_dimensions() -> None:
    tube = trajectory_pore(CylinderPore, {"wall_selection": "walls"}, cylinder_trajectory(), select="liquid", start=0.0)
    assert isinstance(tube, CylinderPore)
    assert tube.diameter_nm == pytest.approx(4.0, abs=1e-9)
    with pytest.raises(CommandLineError, match="missing dimensions of pore 'triangle'"):
        trajectory_pore(TrianglePore, {"wall_selection": "walls"}, cylinder_trajectory(), select="liquid", start=0.0)
    with pytest.raises(CommandLineError, match="unused"):
        trajectory_pore(
            CylinderPore,
            {"wall_selection": "walls", "length_axis": "x"},
            cylinder_trajectory(),
            select="liquid",
            start=0.0,
        )


def test_command_line_lists_the_capsule(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["shapes", "--pore", "cylinder"]) == 0
    assert "capsule" in capsys.readouterr().out


def walls_only_trajectory(walls: np.ndarray) -> ArrayTrajectory:
    return ArrayTrajectory(
        positions_nm=(walls % np.array(TUBE_BOX_NM))[np.newaxis],
        boxes_nm=[TUBE_BOX_NM],
        times_ps=[0.0],
        groups={"walls": np.arange(len(walls))},
    )


def test_a_tube_that_does_not_run_through_the_box_is_an_error() -> None:
    """L, l and φ all use the box edge along the axis: a finite tube would silently get the wrong length."""
    finite = walls_only_trajectory(ring_walls((2.0,), (3.0, 3.0), length=4.0))
    with pytest.raises(InvalidTrajectoryError, match="does not run through the box"):
        CylinderPore.from_trajectory(finite, walls="walls")


@pytest.mark.parametrize(
    ("shells", "diameter"),
    [
        (tuple(0.8 + 0.34 * shell for shell in range(7)), 1.6),  # multi-walled tube with a narrow core
        ((0.41, 0.75), 0.82),  # double-walled (6,6)@(11,11)
    ],
)
def test_multiwalled_and_thin_tubes_are_tubes(shells: tuple[float, ...], diameter: float) -> None:
    pore = CylinderPore.from_trajectory(walls_only_trajectory(ring_walls(shells, (3.0, 3.0), 8.0)), walls="walls")
    assert pore.diameter_nm == pytest.approx(diameter, abs=1e-9)


def test_profile_help_says_any_measurable_pore_is_measured() -> None:
    completed = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "panda_md", "profile", "--help"],
        capture_output=True,
        encoding="utf-8",
        check=False,
    )
    help_text = " ".join(completed.stdout.split())
    assert "the slit is measured" not in help_text
    assert "cylinder" in help_text


def test_liquid_partly_outside_the_tube_is_reported() -> None:
    """Liquid outside the tube still falls into the profile layer L × L_y: the share is reported once, radially."""
    generator = np.random.default_rng(5)
    radius, angle = 1.6 * np.sqrt(generator.random(180)), 2.0 * np.pi * generator.random(180)
    inside = np.column_stack([8.0 * generator.random(180), 3.0 + radius * np.cos(angle), 3.0 + radius * np.sin(angle)])
    corners = np.column_stack([8.0 * generator.random(20), np.full(20, 0.2), np.full(20, 0.2)])
    walls = ring_walls((2.0,), (3.0, 3.0), 8.0)
    liquid = np.vstack([inside, corners])
    trajectory = ArrayTrajectory(
        positions_nm=np.vstack([walls, liquid])[np.newaxis],
        boxes_nm=[TUBE_BOX_NM],
        times_ps=[0.0],
        groups={"walls": np.arange(len(walls)), "liquid": np.arange(len(walls), len(walls) + len(liquid))},
    )
    with pytest.warns(PandaWarning, match="10% of the liquid lies outside the tube"):
        CylinderPore.from_trajectory(trajectory, walls="walls", liquid="liquid")


def test_command_line_fit_of_a_cylinder_pore_file(tmp_path: Path) -> None:
    model = CAPSULE_PORE.interface("capsule")
    values = capsule_values(130.0, 0.03)
    profile_path, pore_path, output = tmp_path / "profile.npz", tmp_path / "pore.json", tmp_path / "fit.json"
    synthetic_profile(model, values, density_bulk_per_nm3=33.0).save(profile_path)
    pore_path.write_text(pore_json(CAPSULE_PORE), encoding="utf-8")
    arguments = ["fit", str(profile_path), "--pore-file", str(pore_path), "--shape", "capsule"]
    code = main([*arguments, "--optimize", "theta,delta", "--rho-bulk", "33", "-o", str(output)])
    assert code == 0
    expected = model.fit(
        DensityProfile.load(profile_path), optimize=["theta", "delta"], density_bulk_per_nm3=33.0
    ).values["theta"]
    assert json.loads(output.read_text())["fit"]["values"]["theta"] == pytest.approx(expected, abs=1e-9)


def test_a_pore_declares_both_trajectory_hooks_or_neither() -> None:
    with pytest.raises(TypeError, match="trajectory_fields"):

        class HalfMeasuredPore(CylinderPore):
            trajectory_fields: ClassVar[tuple[str, ...]] = ()


def test_field_as_float_reads_numbers_and_strings() -> None:
    assert field_as_float({"wall_offset_nm": "0.1"}, "wall_offset_nm", 0.0) == 0.1
    assert field_as_float({"wall_offset_nm": 2}, "wall_offset_nm", 0.0) == 2.0
    assert field_as_float({}, "wall_offset_nm", 0.0) == 0.0
    with pytest.raises(ValueError, match="wall_offset_nm"):
        field_as_float({"wall_offset_nm": [1.0]}, "wall_offset_nm", 0.0)
