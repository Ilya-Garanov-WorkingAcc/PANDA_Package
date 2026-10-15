"""Общий фит: барьер, счётчик вычислений по режимам, сообщения об ошибках, формы без θ и без φ."""

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
    with pytest.raises(UnsupportedRegimeError, match="нет контактного угла"):
        layer.fit(profile, regime=Regime.WETTING, density_bulk_per_nm3=DENSITY_BULK)


def test_layer_volume_fraction_has_no_systematic_bias() -> None:
    """φ слоя — ступенька: различима с точностью до бина (0.005), но без систематического сдвига.

    Без старта из ∫ρ/ρ_bulk Nelder–Mead вставал на нижний край плато: среднее −0.006 (ревью стадии 2).
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
    """Форма без параметра φ — только чтобы проверить отказ нормировки mean."""

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
    with pytest.raises(NotDerivedError, match=r"фит формы droplet_w2c0 невозможен.*droplet_w1c0"):
        corner_droplet.fit(profile, density_bulk_per_nm3=DENSITY_BULK)
    with pytest.raises(NotDerivedError, match=r"droplet_w2c0.*профиль"):
        synthetic_profile(
            corner_droplet, {"theta": 140.0, "delta": 0.02, "phi": 0.1}, density_bulk_per_nm3=DENSITY_BULK
        )
