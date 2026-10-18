"""Common optimizer: parameter rules, input checks, result flags, wetting regimes.

The profiles are noise-free synthetic data: exact recovery shows that the fitting machinery is correct.
"""

import numpy as np
import pytest

from panda_md import DensityProfile, FitResult, Frame, Regime, SlitPore, SquarePore
from panda_md.core.errors import (
    EmptyWindowError,
    InfeasibleParametersError,
    InvalidBoundsError,
    InvalidProfileError,
    ProfileMismatchError,
    UnidentifiableParameterError,
    UnsupportedRegimeError,
)
from panda_md.synthetic.profiles import synthetic_profile

DENSITY_BULK = 30.0
ROLL_TRUTH = {"theta": 120.0, "delta": 0.03, "phi": 0.4}


def roll_profile(slit_pore: SlitPore, **changes: float) -> DensityProfile:
    values = {**ROLL_TRUTH, **changes}
    return synthetic_profile(slit_pore.interface("roll"), values, density_bulk_per_nm3=DENSITY_BULK)


@pytest.mark.parametrize(
    ("optimize", "fixed"),
    [
        (["phi"], {"theta": 120.0, "delta": 0.03}),
        (["delta", "phi"], {"theta": 120.0}),
        (["theta", "delta", "phi"], {}),
    ],
)
def test_one_optimizer_handles_one_two_three_free_parameters(
    slit_pore: SlitPore, optimize: list[str], fixed: dict[str, float]
) -> None:
    result = slit_pore.interface("roll").fit(
        roll_profile(slit_pore), optimize=optimize, fixed=fixed, density_bulk_per_nm3=DENSITY_BULK
    )
    assert result.free_parameters == tuple(name for name in ("theta", "delta", "phi") if name in optimize)
    for name, value in fixed.items():
        assert result.values[name] == value  # fixed values do not change
    # δ is set by the profile edge and is determined to within half a bin (1/200/2 = 0.0025),
    assert result.values["theta"] == pytest.approx(120.0, abs=0.5)
    assert result.values["delta"] == pytest.approx(0.03, abs=0.0025)
    assert result.values["phi"] == pytest.approx(0.4, abs=0.003)


def test_default_optimize_is_theta_and_delta(slit_pore: SlitPore) -> None:
    result = slit_pore.interface("roll").fit(roll_profile(slit_pore), density_bulk_per_nm3=DENSITY_BULK)
    assert result.free_parameters == ("theta", "delta")
    assert result.volume_fraction_from_data == pytest.approx(0.4, abs=1e-3)
    assert result.wall_layer_thickness_nm == pytest.approx(0.03 * 8.0, abs=1e-3)


def test_result_flags_parameter_at_bound(slit_pore: SlitPore) -> None:
    result = slit_pore.interface("roll").fit(
        roll_profile(slit_pore, theta=170.0),
        bounds={"theta": (100.0, 150.0)},
        density_bulk_per_nm3=DENSITY_BULK,
    )
    assert result.values["theta"] == pytest.approx(150.0, abs=0.1)
    assert "theta" in result.at_bounds


def test_result_flags_regime_boundary(square_pore: SquarePore) -> None:
    filament = square_pore.interface("filament_w1c0")
    profile = synthetic_profile(
        filament, {"theta": 89.0, "delta": 0.02, "phi": 0.05}, density_bulk_per_nm3=DENSITY_BULK
    )
    result = filament.fit(profile, fixed={"phi": 0.05}, regime=Regime.NON_WETTING, density_bulk_per_nm3=DENSITY_BULK)
    assert result.values["theta"] == pytest.approx(90.5, abs=0.01)  # θ ∈ (89.5°, 90.5°) is not computed
    assert result.at_regime_boundary
    assert result.at_bounds == ()


def test_summary_mentions_flags(slit_pore: SlitPore) -> None:
    result = slit_pore.interface("roll").fit(
        roll_profile(slit_pore, theta=170.0), bounds={"theta": (100.0, 150.0)}, density_bulk_per_nm3=DENSITY_BULK
    )
    assert "at bound: theta" in str(result)


def test_result_round_trips_through_json(slit_pore: SlitPore) -> None:
    result = slit_pore.interface("roll").fit(roll_profile(slit_pore), density_bulk_per_nm3=DENSITY_BULK)
    restored = FitResult.model_validate_json(result.model_dump_json())
    assert restored.values == result.values
    assert restored.spec == result.spec


def test_empty_window_is_an_error(slit_pore: SlitPore) -> None:
    wrong_pore = SlitPore(width_nm=0.08, length_nm=0.16)  # the pore is 100 times narrower than the data
    with pytest.raises(EmptyWindowError, match="fraction of the pore size"):
        wrong_pore.interface("roll").fit(roll_profile(slit_pore), density_bulk_per_nm3=DENSITY_BULK)


def test_mean_normalization_requires_phi(slit_pore: SlitPore) -> None:
    with pytest.raises(InvalidProfileError, match="ρ_bulk"):  # the cause and what to set — pre-release audit
        slit_pore.interface("roll").fit(roll_profile(slit_pore))


def test_infeasible_fixed_values_are_reported(slit_pore: SlitPore) -> None:
    with pytest.raises(InfeasibleParametersError, match="existence domain"):
        slit_pore.interface("roll").fit(
            roll_profile(slit_pore), fixed={"phi": 0.999}, density_bulk_per_nm3=DENSITY_BULK
        )


def test_fixed_angle_in_undeclared_regime_is_an_error(slit_pore: SlitPore) -> None:
    with pytest.raises(UnsupportedRegimeError, match="wetting"):
        slit_pore.interface("roll").fit(
            roll_profile(slit_pore), optimize=["delta"], fixed={"theta": 60.0}, density_bulk_per_nm3=DENSITY_BULK
        )


def test_parameter_without_influence_is_rejected(slit_pore: SlitPore) -> None:
    droplet = slit_pore.interface("droplet")
    profile = synthetic_profile(
        droplet, {"theta": 130.0, "delta": 0.02, "phi": 0.05}, density_bulk_per_nm3=DENSITY_BULK, frame=Frame.LIQUID
    )
    with pytest.raises(UnidentifiableParameterError, match="delta"):
        droplet.fit(profile, density_bulk_per_nm3=DENSITY_BULK, fixed={"phi": 0.05})


def test_angle_range_across_90_is_split_by_regime(square_pore: SquarePore) -> None:
    filament = square_pore.interface("filament_w1c0")
    profile = synthetic_profile(
        filament, {"theta": 60.0, "delta": 0.02, "phi": 0.05}, density_bulk_per_nm3=DENSITY_BULK
    )
    result = filament.fit(profile, density_bulk_per_nm3=DENSITY_BULK, fixed={"phi": 0.05})
    assert result.regime is Regime.WETTING
    assert result.values["theta"] == pytest.approx(60.0, abs=0.05)
    assert [alternative.regime for alternative in result.alternatives] == [Regime.NON_WETTING]
    assert result.alternatives[0].loss > result.loss


def test_requested_regime_restricts_search(square_pore: SquarePore) -> None:
    filament = square_pore.interface("filament_w1c0")
    profile = synthetic_profile(
        filament, {"theta": 60.0, "delta": 0.02, "phi": 0.05}, density_bulk_per_nm3=DENSITY_BULK
    )
    result = filament.fit(profile, density_bulk_per_nm3=DENSITY_BULK, fixed={"phi": 0.05}, regime=Regime.NON_WETTING)
    assert result.regime is Regime.NON_WETTING
    assert result.alternatives == ()
    assert 90.0 < result.values["theta"] <= 180.0


def test_synthetic_profile_refuses_nonexistent_shape(square_pore: SquarePore) -> None:
    with pytest.raises(InfeasibleParametersError):
        synthetic_profile(
            square_pore.interface("filament_w1c0"),
            {"theta": 60.0, "delta": 0.02, "phi": 0.2},
            density_bulk_per_nm3=DENSITY_BULK,
        )


def test_user_initial_point_is_used(slit_pore: SlitPore) -> None:
    result = slit_pore.interface("roll").fit(
        roll_profile(slit_pore), initial={"theta": 125.0}, starts_per_parameter=1, density_bulk_per_nm3=DENSITY_BULK
    )
    assert result.start_count == 2
    assert np.isclose(result.values["theta"], 120.0, atol=0.05)


def test_fixed_angle_inside_regime_gap_is_an_error(slit_pore: SlitPore) -> None:
    with pytest.raises(UnsupportedRegimeError, match="inside the gap"):
        slit_pore.interface("roll").fit(
            roll_profile(slit_pore), optimize=["delta"], fixed={"theta": 90.0}, density_bulk_per_nm3=DENSITY_BULK
        )


def test_fixed_value_outside_parameter_bounds_is_rejected(slit_pore: SlitPore) -> None:
    with pytest.raises(InvalidBoundsError, match="delta"):
        slit_pore.interface("roll").fit(
            roll_profile(slit_pore), optimize=["theta"], fixed={"delta": -0.05}, density_bulk_per_nm3=DENSITY_BULK
        )


def test_volume_fraction_from_data_outside_bounds_names_likely_causes(slit_pore: SlitPore) -> None:
    with pytest.raises(InvalidBoundsError, match="ρ_bulk"):
        slit_pore.interface("roll").fit(roll_profile(slit_pore), density_bulk_per_nm3=DENSITY_BULK / 10.0)


def test_pore_wider_than_data_is_rejected() -> None:
    data_pore = SlitPore(width_nm=8.0, length_nm=16.0)
    wrong_pore = SlitPore(width_nm=10.0, length_nm=20.0)  # the same l = L/H, H is 25 % larger
    with pytest.raises(ProfileMismatchError, match="do not cover the pore"):
        wrong_pore.interface("roll").fit(roll_profile(data_pore), density_bulk_per_nm3=DENSITY_BULK)


def test_profile_in_other_units_is_rejected(slit_pore: SlitPore) -> None:
    profile = roll_profile(slit_pore)
    in_angstrom = profile.model_copy(update={"coordinate_nm": profile.coordinate_nm * 10.0})
    with pytest.raises(ProfileMismatchError, match="outside the pore"):
        slit_pore.interface("roll").fit(in_angstrom, density_bulk_per_nm3=DENSITY_BULK)


def test_liquid_frame_profile_wider_than_pore_is_rejected(square_pore: SquarePore) -> None:
    droplet = square_pore.interface("droplet_w1c0")
    values = {"theta": 140.0, "delta": 0.02, "phi": 0.05}
    profile = synthetic_profile(droplet, values, density_bulk_per_nm3=DENSITY_BULK, frame=Frame.LIQUID)
    stretched = profile.model_copy(update={"coordinate_nm": profile.coordinate_nm * 3.0})
    with pytest.raises(ProfileMismatchError, match="wider than the pore"):
        droplet.fit(
            stretched, optimize=["theta"], fixed={"delta": 0.02, "phi": 0.05}, density_bulk_per_nm3=DENSITY_BULK
        )


def test_shape_without_contact_angle_fits_volume_fraction(slit_pore: SlitPore) -> None:
    layer = slit_pore.interface("layer")
    profile = synthetic_profile(layer, {"delta": 0.03, "phi": 0.4}, density_bulk_per_nm3=DENSITY_BULK)
    result = layer.fit(profile, optimize=["phi"], density_bulk_per_nm3=DENSITY_BULK)
    assert layer.default_optimize == ("phi",)
    assert result.regime is None
    assert result.values["phi"] == pytest.approx(0.4, abs=0.006)  # the step edge — to within a bin


def test_profile_of_a_pore_with_another_volume_is_rejected() -> None:
    """Pre-release audit: the profile of a 4 × 10 nm pore, fitted in a 4 × 12 nm pore, gave θ = 105.6° without flags."""
    from panda_md import SlitPore
    from panda_md.core.errors import ProfileMismatchError
    from panda_md.synthetic.profiles import synthetic_profile

    measured = SlitPore(width_nm=4.0, length_nm=10.0)
    profile = synthetic_profile(
        measured.interface("worm"), {"theta": 120.0, "delta": 0.0, "phi": 0.3}, density_bulk_per_nm3=30.0
    )
    other = SlitPore(width_nm=4.0, length_nm=12.0)
    with pytest.raises(ProfileMismatchError, match="pore volume"):
        other.interface("worm").fit(profile, density_bulk_per_nm3=30.0)
    measured.interface("worm").fit(profile, density_bulk_per_nm3=30.0)  # its own pore — no error
