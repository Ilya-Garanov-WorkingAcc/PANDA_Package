"""Shape contract — checked automatically for EVERY registered shape of every pore.

A new shape registered with the @Pore.register decorator lands here without editing the tests. The slit is
checked with both a square and a rectangular base (k = 0.4). Profile checks are skipped for shapes whose
profile is not derived (such shapes take part only in the stability diagram).
"""

import itertools
import math

import numpy as np
import pytest
from scipy.integrate import trapezoid

from panda_md import SlitPore, SquarePore, TrianglePore
from panda_md.core.enums import ParameterName, Regime
from panda_md.core.errors import NotDerivedError
from panda_md.core.model import InterfaceModel
from panda_md.core.pore import Pore
from panda_md.core.shape import InterfaceShape, regime_of

PORES: tuple[Pore, ...] = (
    SlitPore(width_nm=8.0, length_nm=16.0),
    SlitPore(width_nm=8.0, length_nm=32.0, base_aspect_ratio=0.4),  # rectangular base (RectCell)
    SquarePore(side_nm=6.0, length_nm=12.0, layer_width_nm=6.0),
    TrianglePore(side_nm=6.0, length_nm=12.0, layer_width_nm=6.0),
)
MODELS = [model for pore in PORES for model in pore.available_interfaces()]
INTEGRATION_POINTS = 200_001
OUTSIDE_OFFSETS = (0.01, 0.4)  # points outside the pore: just past the wall and far past it
INTEGRAL_TOLERANCE = 1e-4  # the step at the profile edge gives a trapezoid error ~ grid step
CENTER_OF_MASS_TOLERANCE = 1e-4
SAMPLE_LIMIT = 12
WALL_LAYERS = (0.0, 0.03)
VOLUME_FRACTIONS = np.linspace(0.005, 0.995, 400)  # fine: the triangle droplet_w3c0 exists in a ±1.5 % window in φ


def has_contact_angle(model: InterfaceModel) -> bool:
    return ParameterName.CONTACT_ANGLE in model.parameters.names


def regime_angles(model: InterfaceModel) -> list[float]:
    """Three angles inside each declared regime (degrees)."""
    angles: list[float] = []
    for lower, upper in model.shape.regimes.values():
        angles.extend(np.linspace(lower, upper, 5)[1:-1].tolist())
    return angles


def feasible_samples(model: InterfaceModel) -> list[dict[str, float]]:
    """Feasible points (θ, δ, φ) — at most SAMPLE_LIMIT, evenly over the grid; (δ, φ) for a shape without θ."""
    grid: list[dict[str, float]]
    if has_contact_angle(model):
        grid = [
            {"theta": theta, "delta": delta, "phi": float(phi)}
            for theta, delta, phi in itertools.product(regime_angles(model), WALL_LAYERS, VOLUME_FRACTIONS)
        ]
    else:
        grid = [{"delta": delta, "phi": float(phi)} for delta, phi in itertools.product(WALL_LAYERS, VOLUME_FRACTIONS)]
    samples = [values for values in grid if model.is_feasible(**values)]
    step = max(1, len(samples) // SAMPLE_LIMIT)
    return samples[::step]


def integration_grid(model: InterfaceModel) -> np.ndarray:
    lower, upper = model.pore.geometry().coordinate_bounds()
    return np.linspace(lower, upper, INTEGRATION_POINTS)


def outside_points(model: InterfaceModel) -> np.ndarray:
    lower, upper = model.pore.geometry().coordinate_bounds()
    return np.array(
        [lower - OUTSIDE_OFFSETS[1], lower - OUTSIDE_OFFSETS[0], upper + OUTSIDE_OFFSETS[0], upper + OUTSIDE_OFFSETS[1]]
    )


def require_profile(model: InterfaceModel) -> None:
    if not model.shape.has_profile():
        pytest.skip(f"{model_id(model)}: profile not derived")


def model_id(model: InterfaceModel) -> str:
    aspect_ratio = getattr(model.pore, "base_aspect_ratio", 1.0)
    suffix = "" if aspect_ratio == 1.0 else f"(k={aspect_ratio})"
    return f"{model.pore.name}{suffix}/{model.shape.name}"


@pytest.mark.parametrize("model", MODELS, ids=model_id)
def test_shape_declares_contract_attributes(model: InterfaceModel) -> None:
    shape = model.shape
    assert isinstance(shape, InterfaceShape)
    assert shape.name in type(model.pore).own_registry()
    if has_contact_angle(model):
        assert shape.regimes, "a shape with θ must declare at least one wetting regime"
    else:
        assert shape.regimes == {}, "a shape without θ has no wetting regimes"
    for regime, (lower, upper) in shape.regimes.items():
        assert isinstance(regime, Regime)
        assert 0.0 <= lower < upper <= 180.0
    assert model.default_optimize, "a shape must have at least one free parameter by default"
    assert set(model.default_optimize) <= set(model.parameters.names)


@pytest.mark.parametrize("model", MODELS, ids=model_id)
def test_shape_exists_somewhere_in_each_regime(model: InterfaceModel) -> None:
    samples = feasible_samples(model)
    assert samples, "no feasible point on the grid; check feasible"
    if has_contact_angle(model):
        covered = {regime_of(sample["theta"]) for sample in samples}
        assert covered == set(model.shape.regimes), "some regime has no feasible points; check feasible"


@pytest.mark.parametrize("model", MODELS, ids=model_id)
def test_profile_integrates_to_volume_fraction(model: InterfaceModel) -> None:
    require_profile(model)
    grid = integration_grid(model)
    integral_per_phi = model.pore.geometry().profile_integral()
    for values in feasible_samples(model):
        density = model.profile(grid, **values)
        integral = trapezoid(density, grid)
        assert integral == pytest.approx(values["phi"] * integral_per_phi, rel=INTEGRAL_TOLERANCE), values


@pytest.mark.parametrize("model", MODELS, ids=model_id)
def test_profile_is_finite_bounded_and_zero_outside_pore(model: InterfaceModel) -> None:
    require_profile(model)
    grid = integration_grid(model)
    outside = outside_points(model)
    for values in feasible_samples(model):
        density = model.profile(grid, **values)
        assert np.all(np.isfinite(density)), values
        assert np.all(density >= 0.0), values
        assert np.all(density <= 1.0 + 1e-12), values
        assert np.all(model.profile(outside, **values) == 0.0), values


@pytest.mark.parametrize("model", MODELS, ids=model_id)
def test_center_of_mass_matches_numerical_value(model: InterfaceModel) -> None:
    require_profile(model)
    geometry = model.pore.geometry()
    for values in feasible_samples(model):
        internal = model.internal_values(values)
        declared = model.shape.center_of_mass(geometry, internal)
        numerical = InterfaceShape.center_of_mass(model.shape, geometry, internal)
        assert declared == pytest.approx(numerical, abs=CENTER_OF_MASS_TOLERANCE), values


@pytest.mark.parametrize("model", MODELS, ids=model_id)
def test_surface_terms_are_non_negative_where_derived(model: InterfaceModel) -> None:
    for values in feasible_samples(model):
        try:
            terms = model.surface_terms(**values)
        except NotDerivedError:
            pytest.skip(f"{model_id(model)}: areas not derived")
        assert terms.liquid_liquid_area > 0.0
        assert terms.liquid_solid_area >= 0.0
        assert math.isfinite(model.capillary_energy(**values))
