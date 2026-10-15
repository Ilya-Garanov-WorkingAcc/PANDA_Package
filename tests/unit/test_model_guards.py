"""InterfaceModel: формулы не вызываются вне объявленных режимов; энергия формы без θ; неизвестная форма."""

from collections.abc import Iterator
from typing import ClassVar

import numpy as np
import pytest

from panda_md.core.enums import Frame, Regime
from panda_md.core.errors import PandaError, UnknownShapeError, UnsupportedRegimeError
from panda_md.core.parameters import VOLUME_FRACTION, Parameter, ParameterValues
from panda_md.core.shape import InterfaceShape, SurfaceTerms
from panda_md.pores.slit import SlitGeometry, SlitPore
from panda_md.synthetic.profiles import synthetic_profile

SLIT = SlitPore(width_nm=8.0, length_nm=16.0)


def test_profile_outside_declared_regime_is_an_error() -> None:
    """Формулы валика выведены только для θ > 90°: при 60° они дают ∫ρ ≠ φ (O1) — считать нельзя."""
    with pytest.raises(UnsupportedRegimeError, match="wetting"):
        SLIT.interface("roll").profile(np.array([0.0]), theta=60.0, phi=0.4)


def test_phi_limits_inside_regime_gap_are_an_error() -> None:
    with pytest.raises(UnsupportedRegimeError, match="зазора"):
        SLIT.interface("roll").phi_limits(theta=90.0, phi=0.4)


@pytest.mark.parametrize("contact_angle", [0.0, 60.0, 90.0])
def test_shape_does_not_exist_outside_its_regimes(contact_angle: float) -> None:
    """Раньше капля при θ = 0 падала с ZeroDivisionError внутри формулы."""
    assert not SLIT.interface("droplet").is_feasible(theta=contact_angle, phi=0.1)


def test_synthetic_profile_refuses_undeclared_regime() -> None:
    with pytest.raises(UnsupportedRegimeError):
        synthetic_profile(SLIT.interface("roll"), {"theta": 60.0, "phi": 0.4}, density_bulk_per_nm3=30.0)


def test_unknown_shape_is_panda_error_and_key_error() -> None:
    with pytest.raises(UnknownShapeError, match="доступны") as error:
        SLIT.interface("no_such_shape")
    assert isinstance(error.value, PandaError)
    assert isinstance(error.value, KeyError)


class TouchingWithoutAngle(InterfaceShape[SlitGeometry]):
    """Форма без θ, касающаяся стенок, — энергию без угла не посчитать."""

    name: ClassVar[str] = "test_touching_without_angle"
    parameters: ClassVar[tuple[Parameter, ...]] = (VOLUME_FRACTION,)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {}
    frame: ClassVar[Frame] = Frame.PORE
    default_optimize: ClassVar[tuple[str, ...]] = ("phi",)

    def feasible(self, geometry: SlitGeometry, values: ParameterValues) -> bool:
        return True

    def surface_terms(self, geometry: SlitGeometry, values: ParameterValues) -> SurfaceTerms:
        return SurfaceTerms(liquid_liquid_area=1.0, liquid_solid_area=2.0)


@pytest.fixture
def touching_without_angle() -> Iterator[None]:
    SlitPore.register(TouchingWithoutAngle)
    yield
    del SlitPore.own_registry()["test_touching_without_angle"]


@pytest.mark.usefixtures("touching_without_angle")
def test_energy_of_shape_without_angle_requires_no_wall_contact() -> None:
    with pytest.raises(PandaError, match="касается стенок"):
        SLIT.interface("test_touching_without_angle").capillary_energy(phi=0.3)


def test_energy_of_layer_is_liquid_liquid_area() -> None:
    layer = SLIT.interface("layer")
    assert layer.capillary_energy(phi=0.3) == pytest.approx(layer.surface_terms(phi=0.3).liquid_liquid_area)


def test_layer_filling_whole_accessible_width_does_not_exist() -> None:
    """φ = 1 − 2δ — слой касается слоёв δ у обеих стенок; как у остальных форм, граница не включается."""
    layer = SLIT.interface("layer")
    assert layer.is_feasible(delta=0.05, phi=0.89)
    assert not layer.is_feasible(delta=0.05, phi=0.9)
