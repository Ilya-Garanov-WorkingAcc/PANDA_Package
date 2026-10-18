"""InterfaceModel: formulas are not called outside the declared regimes; energy of a shape without θ; unknown shape."""

from collections.abc import Iterator
from typing import ClassVar

import numpy as np
import pytest
from scipy.integrate import trapezoid

from panda_md.core.enums import Frame, Regime
from panda_md.core.errors import PandaError, UnknownShapeError, UnsupportedRegimeError
from panda_md.core.parameters import VOLUME_FRACTION, Parameter, ParameterValues
from panda_md.core.shape import InterfaceShape, SurfaceTerms
from panda_md.pores.slit import SlitGeometry, SlitPore
from panda_md.synthetic.profiles import synthetic_profile

SLIT = SlitPore(width_nm=8.0, length_nm=16.0)


def test_profile_outside_declared_regime_is_an_error() -> None:
    """The roll formulas are derived only for θ > 90°: at 60° they give ∫ρ ≠ φ (O1), so they must not be evaluated."""
    with pytest.raises(UnsupportedRegimeError, match="wetting"):
        SLIT.interface("roll").profile(np.array([0.0]), theta=60.0, phi=0.4)


def test_phi_limits_inside_regime_gap_are_an_error() -> None:
    with pytest.raises(UnsupportedRegimeError, match="inside the gap"):
        SLIT.interface("roll").phi_limits(theta=90.0, phi=0.4)


@pytest.mark.parametrize("contact_angle", [0.0, 60.0, 90.0])
def test_shape_does_not_exist_outside_its_regimes(contact_angle: float) -> None:
    """Previously the droplet at θ = 0 failed with ZeroDivisionError inside the formula."""
    assert not SLIT.interface("droplet").is_feasible(theta=contact_angle, phi=0.1)


def test_synthetic_profile_refuses_undeclared_regime() -> None:
    with pytest.raises(UnsupportedRegimeError):
        synthetic_profile(SLIT.interface("roll"), {"theta": 60.0, "phi": 0.4}, density_bulk_per_nm3=30.0)


def test_unknown_shape_is_panda_error_and_key_error() -> None:
    with pytest.raises(UnknownShapeError, match="available:") as error:
        SLIT.interface("no_such_shape")
    assert isinstance(error.value, PandaError)
    assert isinstance(error.value, KeyError)


class TouchingWithoutAngle(InterfaceShape[SlitGeometry]):
    """A shape without θ that touches the walls: the energy cannot be computed without the angle."""

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
    with pytest.raises(PandaError, match="touches the walls"):
        SLIT.interface("test_touching_without_angle").capillary_energy(phi=0.3)


def test_energy_of_layer_is_liquid_liquid_area() -> None:
    layer = SLIT.interface("layer")
    assert layer.capillary_energy(phi=0.3) == pytest.approx(layer.surface_terms(phi=0.3).liquid_liquid_area)


def test_layer_filling_whole_accessible_width_does_not_exist() -> None:
    """φ = 1 − 2δ: the layer touches the δ layers at both walls; as for the other shapes, the bound is excluded."""
    layer = SLIT.interface("layer")
    assert layer.is_feasible(delta=0.05, phi=0.89)
    assert not layer.is_feasible(delta=0.05, phi=0.9)


def test_surface_terms_outside_existence_region_are_a_clear_error() -> None:
    from panda_md import TrianglePore
    from panda_md.core.errors import InfeasibleParametersError

    droplet = TrianglePore(side_nm=1.0, length_nm=2.0, layer_width_nm=1.0).interface("droplet_w1c0")
    values = {"theta": 150.0, "delta": 0.0, "phi": 0.95}  # the droplet does not fit
    assert not droplet.is_feasible(**values)
    with pytest.raises(InfeasibleParametersError, match="does not exist"):
        droplet.surface_terms(**values)
    with pytest.raises(InfeasibleParametersError, match="does not exist"):
        droplet.capillary_energy(**values)


@pytest.mark.parametrize("shape_name", ["capsule_w3c0", "droplet_w2c0", "droplet_w3c0", "filament_w2c0"])
def test_round_regime_bound_survives_degree_radian_round_trip(shape_name: str) -> None:
    """Pre-release audit: degrees(radians(120)) = 119.999…, and θ = 120° was outside the regime [120, 180]."""
    from panda_md import TrianglePore

    model = TrianglePore(side_nm=4.0, length_nm=8.0, layer_width_nm=6.0).interface(shape_name)
    model.supported_values({"theta": 120.0, "delta": 0.02, "phi": 0.3})  # no UnsupportedRegimeError


def test_regime_gap_bound_has_the_same_tolerance() -> None:
    roll = SlitPore(width_nm=8.0, length_nm=16.0).interface("roll")
    roll.shape.require_supported_contact_angle(90.5 - 1e-12)  # the bound of the gap around 90° also has the tolerance


@pytest.mark.parametrize(
    ("length_nm", "aspect_ratio", "theta", "delta"),
    [
        (2.0, 1.0, 180.0, 0.0),
        (0.9, 1.0, 163.2, 0.1),
        (4.264, 0.086, 164.3, 0.0),
        (20.0, 0.1, 180.0, 0.0),
        (0.4, 1.0, 150.0, 0.0),
    ],
)
def test_perforation_exists_only_where_the_hole_fits_the_cell(
    length_nm: float, aspect_ratio: float, theta: float, delta: float
) -> None:
    """Pre-release audit: at k·l ≲ 0.55 the lower φ bound was a spurious root, the hole wider than the cell, ρ < 0."""
    pore = SlitPore(width_nm=4.0, length_nm=length_nm, base_aspect_ratio=aspect_ratio)
    perforation = pore.interface("perforation")
    lower, upper = perforation.phi_limits(theta=theta, delta=delta, phi=0.5)
    assert lower >= 0.0
    coordinate = np.linspace(-0.5, 0.5, 4001)
    for phi in np.linspace(lower, upper, 9)[1:-1]:
        values = {"theta": theta, "delta": delta, "phi": float(phi)}
        if not perforation.is_feasible(**values):
            continue
        density = perforation.profile(coordinate, **values)
        assert float(np.min(density)) >= -1e-9
        assert trapezoid(density, coordinate) == pytest.approx(phi, rel=2e-3)
