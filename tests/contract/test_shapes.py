"""Контракт формы — проверяется автоматически для КАЖДОЙ зарегистрированной формы каждой поры.

Новая форма, подключённая декоратором @Pore.register, попадает сюда без правки тестов.
"""

import itertools
import math

import numpy as np
import pytest
from scipy.integrate import trapezoid

from panda_md import SlitPore
from panda_md.core.enums import ParameterName, Regime
from panda_md.core.errors import NotDerivedError
from panda_md.core.model import InterfaceModel
from panda_md.core.pore import Pore
from panda_md.core.shape import InterfaceShape, regime_of

PORES: tuple[Pore, ...] = (SlitPore(width_nm=8.0, length_nm=16.0),)
MODELS = [model for pore in PORES for model in pore.available_interfaces()]
INTEGRATION_GRID = np.linspace(-0.5, 0.5, 200_001)
INTEGRAL_TOLERANCE = 1e-4  # ступенька на краю профиля даёт ошибку трапеций ~ шаг сетки
CENTER_OF_MASS_TOLERANCE = 1e-4
SAMPLE_LIMIT = 12
WALL_LAYERS = (0.0, 0.03)
VOLUME_FRACTIONS = np.linspace(0.02, 0.98, 25)


def regime_angles(model: InterfaceModel) -> list[float]:
    """Три угла внутри каждого объявленного режима (градусы)."""
    angles: list[float] = []
    for lower, upper in model.shape.regimes.values():
        angles.extend(np.linspace(lower, upper, 5)[1:-1].tolist())
    return angles


def feasible_samples(model: InterfaceModel) -> list[dict[str, float]]:
    """Допустимые точки (θ, δ, φ) — не больше SAMPLE_LIMIT, равномерно по сетке."""
    samples = [
        {"theta": theta, "delta": delta, "phi": float(phi)}
        for theta, delta, phi in itertools.product(regime_angles(model), WALL_LAYERS, VOLUME_FRACTIONS)
        if model.is_feasible(theta=theta, delta=delta, phi=float(phi))
    ]
    step = max(1, len(samples) // SAMPLE_LIMIT)
    return samples[::step]


def model_id(model: InterfaceModel) -> str:
    return f"{model.pore.name}/{model.shape.name}"


@pytest.mark.parametrize("model", MODELS, ids=model_id)
def test_shape_declares_contract_attributes(model: InterfaceModel) -> None:
    shape = model.shape
    assert isinstance(shape, InterfaceShape)
    assert shape.name in type(model.pore).own_registry()
    assert shape.regimes, "форма должна объявить хотя бы один режим смачивания"
    for regime, (lower, upper) in shape.regimes.items():
        assert isinstance(regime, Regime)
        assert 0.0 <= lower < upper <= 180.0
    assert ParameterName.CONTACT_ANGLE in model.parameters.names


@pytest.mark.parametrize("model", MODELS, ids=model_id)
def test_shape_exists_somewhere_in_each_regime(model: InterfaceModel) -> None:
    covered = {regime_of(sample["theta"]) for sample in feasible_samples(model)}
    assert covered == set(model.shape.regimes), "в каком-то режиме нет ни одной допустимой точки — проверь feasible"


@pytest.mark.parametrize("model", MODELS, ids=model_id)
def test_profile_integrates_to_volume_fraction(model: InterfaceModel) -> None:
    integral_per_phi = model.pore.geometry().profile_integral()
    for values in feasible_samples(model):
        density = model.profile(INTEGRATION_GRID, **values)
        integral = trapezoid(density, INTEGRATION_GRID)
        assert integral == pytest.approx(values["phi"] * integral_per_phi, rel=INTEGRAL_TOLERANCE), values


@pytest.mark.parametrize("model", MODELS, ids=model_id)
def test_profile_is_finite_bounded_and_zero_outside_pore(model: InterfaceModel) -> None:
    outside = np.array([-0.9, -0.51, 0.51, 0.9])
    for values in feasible_samples(model):
        density = model.profile(INTEGRATION_GRID, **values)
        assert np.all(np.isfinite(density)), values
        assert np.all(density >= 0.0), values
        assert np.all(density <= 1.0 + 1e-12), values
        assert np.all(model.profile(outside, **values) == 0.0), values


@pytest.mark.parametrize("model", MODELS, ids=model_id)
def test_center_of_mass_matches_numerical_value(model: InterfaceModel) -> None:
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
            pytest.skip(f"{model_id(model)}: площади не выведены")
        assert terms.liquid_liquid_area > 0.0
        assert terms.liquid_solid_area >= 0.0
        assert math.isfinite(model.capillary_energy(**values))
