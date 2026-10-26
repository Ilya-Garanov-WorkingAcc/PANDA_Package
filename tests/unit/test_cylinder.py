"""Cylindrical pore and its capsule shape (formulas: alpha.py rollx)."""

import math
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.integrate import trapezoid

import panda_md as panda
from panda_md import CylinderPore
from panda_md.cli.options import load_pore, pore_json
from panda_md.core.enums import Axis
from panda_md.core.errors import UnsupportedRegimeError
from panda_md.pores.cylinder.capsule import CylinderCapsule
from panda_md.pores.registry import pore_class
from panda_md.synthetic.profiles import synthetic_profile

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
