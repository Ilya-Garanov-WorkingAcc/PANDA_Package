"""Common fit: barrier, evaluation counter per regime, error messages, shapes without θ and without φ."""

from collections.abc import Iterator
from typing import ClassVar

import numpy as np
import pytest

from panda_md import DensityProfile, LossFunction, Normalization, Regime, SlitPore, SquarePore, TrianglePore
from panda_md.core.enums import Frame
from panda_md.core.errors import (
    InfeasibleParametersError,
    InvalidProfileError,
    NotDerivedError,
    UnsupportedRegimeError,
)
from panda_md.core.fitting import _Objective
from panda_md.core.parameters import CONTACT_ANGLE, Parameter, ParameterValues
from panda_md.core.shape import InterfaceShape
from panda_md.core.types import FloatArray
from panda_md.pores.slit import SlitGeometry
from panda_md.synthetic.profiles import synthetic_profile

DENSITY_BULK = 30.0
SLIT = SlitPore(width_nm=8.0, length_nm=16.0)
SQUARE = SquarePore(side_nm=6.0, length_nm=12.0, layer_width_nm=6.0)


def test_barrier_is_above_any_feasible_loss_and_grows_with_violation() -> None:
    roll = SLIT.interface("roll")
    truth = {"theta": 120.0, "delta": 0.03, "phi": 0.4}
    profile = synthetic_profile(roll, truth, density_bulk_per_nm3=DENSITY_BULK)
    reduced = SLIT.reduce(profile, Normalization.BULK, DENSITY_BULK)
    free_spec = roll.parameters.resolve(["phi"], {"theta": 120.0, "delta": 0.03}, {}, {}, {})
    objective = _Objective(roll, free_spec, reduced, np.ones(reduced.coordinate.shape, dtype=bool), LossFunction.L1)
    _, upper = roll.phi_limits(**truth)
    feasible_loss = objective(np.array([0.5 * upper]))
    slightly_outside = objective(np.array([upper + 0.01]))
    far_outside = objective(np.array([upper + 0.05]))
    assert feasible_loss < objective.barrier_level < slightly_outside < far_outside


def test_evaluation_count_belongs_to_its_regime() -> None:
    filament = SQUARE.interface("filament_w1c0")
    profile = synthetic_profile(
        filament, {"theta": 60.0, "delta": 0.02, "phi": 0.05}, density_bulk_per_nm3=DENSITY_BULK
    )
    both = filament.fit(profile, fixed={"phi": 0.05}, density_bulk_per_nm3=DENSITY_BULK)
    wetting_only = filament.fit(profile, fixed={"phi": 0.05}, regime=Regime.WETTING, density_bulk_per_nm3=DENSITY_BULK)
    assert both.regime is Regime.WETTING
    assert both.evaluation_count == wetting_only.evaluation_count
    assert both.alternatives[0].evaluation_count > 0


def test_infeasible_start_error_lists_ranges() -> None:
    roll = SLIT.interface("roll")
    profile = synthetic_profile(roll, {"theta": 120.0, "delta": 0.03, "phi": 0.4}, density_bulk_per_nm3=DENSITY_BULK)
    with pytest.raises(InfeasibleParametersError, match=r"theta ∈ \[90\.5, 180\].*phi = 0\.999"):
        roll.fit(profile, fixed={"phi": 0.999}, density_bulk_per_nm3=DENSITY_BULK)


def test_requested_regime_for_shape_without_contact_angle_is_an_error() -> None:
    layer = SLIT.interface("layer")
    profile = synthetic_profile(layer, {"delta": 0.03, "phi": 0.4}, density_bulk_per_nm3=DENSITY_BULK)
    with pytest.raises(UnsupportedRegimeError, match="has no contact angle"):
        layer.fit(profile, regime=Regime.WETTING, density_bulk_per_nm3=DENSITY_BULK)


def test_layer_volume_fraction_has_no_systematic_bias() -> None:
    """φ of the layer is a step: resolved to within a bin (0.005), but without a systematic shift.

    Without a start from ∫ρ/ρ_bulk Nelder–Mead stopped at the lower edge of the plateau: mean −0.006 (stage 2 review).
    """
    layer = SLIT.interface("layer")
    errors = []
    for volume_fraction in np.linspace(0.05, 0.9, 35):
        profile = synthetic_profile(
            layer, {"delta": 0.03, "phi": float(volume_fraction)}, density_bulk_per_nm3=DENSITY_BULK
        )
        errors.append(layer.fit(profile, density_bulk_per_nm3=DENSITY_BULK).values["phi"] - volume_fraction)
    assert max(abs(error) for error in errors) <= 0.005 + 1e-9
    assert abs(float(np.mean(errors))) < 0.0025


class ContactAngleOnly(InterfaceShape[SlitGeometry]):
    """A shape without the φ parameter — only to check that mean normalization is refused."""

    name: ClassVar[str] = "test_contact_angle_only"
    parameters: ClassVar[tuple[Parameter, ...]] = (CONTACT_ANGLE,)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: (90.0, 180.0)}
    frame: ClassVar[Frame] = Frame.PORE
    default_optimize: ClassVar[tuple[str, ...]] = ("theta",)

    def feasible(self, geometry: SlitGeometry, values: ParameterValues) -> bool:
        return True

    def profile(self, coordinate: FloatArray, geometry: SlitGeometry, values: ParameterValues) -> FloatArray:
        return np.full_like(coordinate, 0.5)


@pytest.fixture
def contact_angle_only() -> Iterator[None]:
    SlitPore.register(ContactAngleOnly)
    yield
    del SlitPore.own_registry()["test_contact_angle_only"]


@pytest.mark.usefixtures("contact_angle_only")
def test_mean_normalization_needs_volume_fraction_parameter() -> None:
    profile = DensityProfile(
        coordinate_nm=np.linspace(-3.98, 3.98, 200),
        density_per_nm3=np.full(200, 10.0),
        particle_count=1000.0,
        pore_volume_nm3=SLIT.volume_nm3(),
    )
    with pytest.raises(InvalidProfileError, match="φ"):
        SLIT.interface("test_contact_angle_only").fit(profile)


def test_fit_of_shape_without_profile_fails_before_optimizing() -> None:
    triangle = TrianglePore(side_nm=6.0, length_nm=12.0, layer_width_nm=6.0)
    profile = synthetic_profile(
        triangle.interface("droplet_w1c0"),
        {"theta": 140.0, "delta": 0.02, "phi": 0.1},
        density_bulk_per_nm3=DENSITY_BULK,
    )
    corner_droplet = triangle.interface("droplet_w2c0")
    with pytest.raises(NotDerivedError, match=r"shape 'droplet_w2c0' cannot be fitted.*droplet_w1c0"):
        corner_droplet.fit(profile, density_bulk_per_nm3=DENSITY_BULK)
    with pytest.raises(NotDerivedError, match=r"profile is not derived.*droplet_w2c0"):
        synthetic_profile(
            corner_droplet, {"theta": 140.0, "delta": 0.02, "phi": 0.1}, density_bulk_per_nm3=DENSITY_BULK
        )


def test_volume_fraction_from_histogram_is_exact_on_a_coarse_grid() -> None:
    """Pre-release audit: ∫ρ by trapezoids over bin centres lost half a bin at each wall — φ too low, θ shifted."""
    from panda_md import SlitPore
    from panda_md.core.enums import ProfileSampling
    from panda_md.synthetic.profiles import synthetic_profile

    worm = SlitPore(width_nm=4.0, length_nm=10.0).interface("worm")
    truth = {"theta": 120.0, "delta": 0.0, "phi": 0.3}
    profile = synthetic_profile(
        worm, truth, density_bulk_per_nm3=30.0, bin_count=20, sampling=ProfileSampling.BIN_AVERAGE
    )
    result = worm.fit(profile, density_bulk_per_nm3=30.0)
    assert result.volume_fraction_from_data == pytest.approx(0.3, abs=1e-4)  # synthetic data: 16 points per bin
    assert result.values["theta"] == pytest.approx(120.0, abs=0.5)


def test_wall_layer_at_zero_is_a_physical_bound_not_a_problem() -> None:
    """Pre-release audit: δ = 0 (no layer) gave the flag "at bound: delta" and exit code 2 of panda fit."""
    from panda_md import SlitPore
    from panda_md.synthetic.profiles import synthetic_profile

    roll = SlitPore(width_nm=8.0, length_nm=16.0).interface("roll")
    profile = synthetic_profile(roll, {"theta": 120.0, "delta": 0.0, "phi": 0.4}, density_bulk_per_nm3=30.0)
    assert "delta" not in roll.fit(profile, density_bulk_per_nm3=30.0).at_bounds
    narrowed = roll.fit(profile, density_bulk_per_nm3=30.0, bounds={"delta": (0.01, 0.1)})
    assert "delta" in narrowed.at_bounds  # a bound narrowed by the user is still flagged


def test_barrier_stays_above_feasible_loss_for_mean_normalization_and_small_phi() -> None:
    """Pre-release audit: mean normalization, L2, φ = 1e−4 — feasible loss 83.9 with a barrier of 22.9."""
    worm = SLIT.interface("worm")
    profile = synthetic_profile(worm, {"theta": 120.0, "delta": 0.0, "phi": 0.1}, density_bulk_per_nm3=DENSITY_BULK)
    reduced = SLIT.reduce(profile, Normalization.MEAN, None)
    free_spec = worm.parameters.resolve(["theta"], {"delta": 0.0, "phi": 1e-4}, {}, {}, {})
    objective = _Objective(worm, free_spec, reduced, np.ones(reduced.coordinate.shape, dtype=bool), LossFunction.L2)
    feasible_loss = objective(np.array([np.radians(91.0)]))
    assert worm.is_feasible(theta=91.0, delta=0.0, phi=1e-4)
    assert feasible_loss < objective.barrier_level
