"""Synthetic validation: shape → noisy profile → fit → recovered parameters.

Thresholds and the findings are described in tests/README.md (the synthetic data, O2 and O3 sections).
"""

import numpy as np
import pytest

from panda_md import Frame, SlitPore, SquarePore, TrianglePore
from panda_md.core.model import InterfaceModel
from panda_md.synthetic.profiles import synthetic_profile

DENSITY_BULK = 30.0
NOISE_STD = 0.005  # noise 0.5 % of ρ_bulk per bin, 200 bins
SEEDS = range(5)
MAX_ANGLE_ERROR_DEG = 1.5  # in every run
MEAN_ANGLE_ERROR_DEG = 1.0  # mean over the runs (requirement |Δθ| < 1°)

SLIT = SlitPore(width_nm=8.0, length_nm=16.0)
SQUARE = SquarePore(side_nm=6.0, length_nm=12.0, layer_width_nm=6.0)
TRIANGLE = TrianglePore(side_nm=6.0, length_nm=12.0, layer_width_nm=6.0)
RECTANGULAR_SLIT = SlitPore(width_nm=8.0, length_nm=32.0, base_aspect_ratio=0.4)
CASES: list[tuple[InterfaceModel, dict[str, float]]] = [
    (SLIT.interface("roll"), {"theta": 120.0, "delta": 0.03, "phi": 0.4}),
    (SLIT.interface("droplet"), {"theta": 130.0, "delta": 0.03, "phi": 0.05}),
    (SQUARE.interface("droplet_w1c0"), {"theta": 140.0, "delta": 0.02, "phi": 0.05}),
    (SQUARE.interface("filament_w1c0"), {"theta": 130.0, "delta": 0.02, "phi": 0.1}),
    (SQUARE.interface("filament_w1c0"), {"theta": 60.0, "delta": 0.02, "phi": 0.05}),
    (SLIT.interface("doughnut"), {"theta": 130.0, "delta": 0.03, "phi": 0.3372}),
    (SLIT.interface("worm"), {"theta": 130.0, "delta": 0.03, "phi": 0.226}),
    (SLIT.interface("perforation"), {"theta": 130.0, "delta": 0.03, "phi": 0.6493}),
    (RECTANGULAR_SLIT.interface("droplet"), {"theta": 130.0, "delta": 0.03, "phi": 0.0561}),
    (RECTANGULAR_SLIT.interface("roll"), {"theta": 130.0, "delta": 0.03, "phi": 0.4848}),
    (TRIANGLE.interface("droplet_w1c0"), {"theta": 140.0, "delta": 0.02, "phi": 0.05}),
    (TRIANGLE.interface("filament_w3c0"), {"theta": 150.0, "delta": 0.02, "phi": 0.5}),
    (TRIANGLE.interface("capsule_w3c0"), {"theta": 150.0, "delta": 0.02, "phi": 0.4}),
    (SQUARE.interface("droplet_w2c0"), {"theta": 150.0, "delta": 0.02, "phi": 0.05}),
    (SQUARE.interface("droplet_w2c1"), {"theta": 115.0, "delta": 0.02, "phi": 0.1}),
    (SQUARE.interface("capsule_w4c0"), {"theta": 160.0, "delta": 0.02, "phi": 0.4}),
]


def case_id(case: tuple[InterfaceModel, dict[str, float]]) -> str:
    model, truth = case
    aspect_ratio = getattr(model.pore, "base_aspect_ratio", 1.0)
    suffix = "" if aspect_ratio == 1.0 else f"(k={aspect_ratio})"
    return f"{model.pore.name}{suffix}/{model.shape.name}/θ={truth['theta']:.0f}"


def angle_errors(model: InterfaceModel, truth: dict[str, float], density_bulk_per_nm3: float) -> list[float]:
    errors = []
    for seed in SEEDS:
        profile = synthetic_profile(model, truth, density_bulk_per_nm3=DENSITY_BULK, noise_std=NOISE_STD, seed=seed)
        result = model.fit(profile, density_bulk_per_nm3=density_bulk_per_nm3)
        errors.append(abs(result.values["theta"] - truth["theta"]))
    return errors


def assert_angle_recovered(errors: list[float]) -> None:
    assert max(errors) < MAX_ANGLE_ERROR_DEG, errors
    assert float(np.mean(errors)) < MEAN_ANGLE_ERROR_DEG, errors


@pytest.mark.slow
@pytest.mark.parametrize("case", CASES, ids=case_id)
def test_angle_and_wall_layer_recovered_with_bulk_density(case: tuple[InterfaceModel, dict[str, float]]) -> None:
    model, truth = case
    assert_angle_recovered(angle_errors(model, truth, DENSITY_BULK))


@pytest.mark.slow
@pytest.mark.parametrize(
    ("optimize", "fixed"),
    [(["phi"], {"theta": 120.0, "delta": 0.03}), (["delta", "phi"], {"theta": 120.0}), (["theta", "delta", "phi"], {})],
    ids=["phi", "delta-phi", "theta-delta-phi"],
)
def test_task_modes_recover_volume_fraction(optimize: list[str], fixed: dict[str, float]) -> None:
    model, truth = CASES[0]
    for seed in SEEDS:
        profile = synthetic_profile(model, truth, density_bulk_per_nm3=DENSITY_BULK, noise_std=NOISE_STD, seed=seed)
        result = model.fit(profile, optimize=optimize, fixed=fixed, density_bulk_per_nm3=DENSITY_BULK)
        assert result.values["phi"] == pytest.approx(truth["phi"], abs=0.01)
        assert result.values["theta"] == pytest.approx(truth["theta"], abs=MAX_ANGLE_ERROR_DEG)


@pytest.mark.slow
@pytest.mark.parametrize("case", CASES[1:4], ids=case_id)
def test_mean_normalization_recovers_compact_shapes(case: tuple[InterfaceModel, dict[str, float]]) -> None:
    """O2: without ρ_bulk φ and θ are determined together — for droplets and filaments (not the roll, see README)."""
    model, truth = case
    for seed in SEEDS:
        profile = synthetic_profile(model, truth, density_bulk_per_nm3=DENSITY_BULK, noise_std=NOISE_STD, seed=seed)
        result = model.fit(profile, optimize=["theta", "delta", "phi"])
        assert result.values["phi"] == pytest.approx(truth["phi"], abs=0.01)
        assert result.values["theta"] == pytest.approx(truth["theta"], abs=3.0)
        assert result.density_bulk_derived_per_nm3 == pytest.approx(DENSITY_BULK, rel=0.15)


def test_mean_normalization_with_known_phi_recovers_roll() -> None:
    """O2: the roll in mean normalization needs a known φ (from the system construction)."""
    model, truth = CASES[0]
    profile = synthetic_profile(model, truth, density_bulk_per_nm3=DENSITY_BULK)
    result = model.fit(profile, fixed={"phi": truth["phi"]})
    assert result.values["theta"] == pytest.approx(truth["theta"], abs=0.5)
    assert result.density_bulk_derived_per_nm3 == pytest.approx(DENSITY_BULK, rel=1e-3)


WALL_OFFSET = 0.05  # wall position error: 5 % of the pore size


@pytest.mark.parametrize("case", CASES[1:4], ids=case_id)
def test_wall_offset_goes_into_wall_layer_for_one_wall_shapes(case: tuple[InterfaceModel, dict[str, float]]) -> None:
    """O3: for shapes on one wall the wall shift is absorbed by δ, θ does not change — pore centering."""
    model, truth = case
    profile = synthetic_profile(model, truth, density_bulk_per_nm3=DENSITY_BULK, offset=WALL_OFFSET)
    result = model.fit(profile, density_bulk_per_nm3=DENSITY_BULK)
    assert result.values["theta"] == pytest.approx(truth["theta"], abs=0.1)


def test_wall_offset_needs_free_shift_for_roll() -> None:
    """O3: the roll touches both walls — the shift is absorbed only by the shift parameter."""
    model, truth = CASES[0]
    profile = synthetic_profile(model, truth, density_bulk_per_nm3=DENSITY_BULK, offset=WALL_OFFSET)
    with_shift = model.fit(profile, optimize=["theta", "delta", "shift"], density_bulk_per_nm3=DENSITY_BULK)
    without_shift = model.fit(profile, density_bulk_per_nm3=DENSITY_BULK)
    assert with_shift.values["theta"] == pytest.approx(truth["theta"], abs=0.5)
    assert abs(without_shift.values["theta"] - truth["theta"]) > 2.0


@pytest.mark.parametrize("case", CASES[1:4], ids=case_id)
def test_liquid_frame_recovers_angle_with_known_wall_layer(case: tuple[InterfaceModel, dict[str, float]]) -> None:
    model, truth = case
    profile = synthetic_profile(model, truth, density_bulk_per_nm3=DENSITY_BULK, frame=Frame.LIQUID)
    result = model.fit(profile, optimize=["theta"], fixed={"delta": truth["delta"]}, density_bulk_per_nm3=DENSITY_BULK)
    assert result.values["theta"] == pytest.approx(truth["theta"], abs=0.1)
