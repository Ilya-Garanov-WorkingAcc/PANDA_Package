"""InterfaceModel — форма в конкретной поре; значения параметров в единицах пользователя."""

import math

import numpy as np
import pytest
from pydantic import ValidationError

from panda_md.core.errors import MissingParameterValueError, NotDerivedError, UnknownParameterError
from panda_md.pores.slit import SlitPore
from panda_md.pores.square import SquarePore

SLIT = SlitPore(width_nm=8.0, length_nm=16.0)


def test_model_lists_shape_parameters_and_shift() -> None:
    roll = SLIT.interface("roll")
    assert roll.parameters.names == ("theta", "delta", "phi", "shift")
    assert roll.default_optimize == ("theta", "delta")


def test_internal_values_convert_degrees_and_fill_defaults() -> None:
    values = SLIT.interface("roll").internal_values({"theta": 180.0, "phi": 0.4})
    assert values == {"theta": pytest.approx(math.pi), "delta": 0.0, "phi": 0.4, "shift": 0.0}


def test_missing_volume_fraction_is_reported() -> None:
    with pytest.raises(MissingParameterValueError, match="phi"):
        SLIT.interface("roll").profile([0.0], theta=120.0)


def test_unknown_parameter_is_reported() -> None:
    with pytest.raises(UnknownParameterError, match="angle"):
        SLIT.interface("roll").profile([0.0], theta=120.0, phi=0.4, angle=1.0)


def test_roll_profile_matches_closed_form_at_pore_center() -> None:
    theta, delta, phi = math.radians(120.0), 0.03, 0.4
    length_ratio = 2.0
    accessible = 1.0 - 2.0 * delta
    spot_width = (
        phi * length_ratio / accessible
        + accessible * (math.pi - 2 * theta) / (4 * math.cos(theta) ** 2)
        - 0.5 * accessible * math.tan(theta)
    )
    expected = (spot_width + accessible / abs(math.cos(theta)) + accessible * math.tan(theta)) / length_ratio
    profile = SLIT.interface("roll").profile(np.array([0.0, 0.49, 0.6]), theta=120.0, delta=delta, phi=phi)
    assert profile[0] == pytest.approx(expected, rel=1e-12)
    assert profile[1] == 0.0  # внутри слоя δ у стенки
    assert profile[2] == 0.0  # за стенкой


def test_feasibility_follows_volume_fraction_limits() -> None:
    roll = SLIT.interface("roll")
    lower, upper = roll.phi_limits(theta=120.0, delta=0.03, phi=0.5)
    assert 0.0 < lower < upper < 1.0
    assert roll.is_feasible(theta=120.0, delta=0.03, phi=0.5 * (lower + upper))
    assert not roll.is_feasible(theta=120.0, delta=0.03, phi=upper + 0.01)


def test_surface_terms_not_derived_for_slit_roll() -> None:
    with pytest.raises(NotDerivedError):
        SLIT.interface("roll").surface_terms(theta=120.0, phi=0.4)


def test_square_pore_exposes_its_shapes() -> None:
    assert set(SquarePore.shape_names()) == {"filament_w1c0", "droplet_w1c0"}
    assert set(SlitPore.shape_names()) == {"roll", "droplet"}


def test_square_droplet_energy_is_capillary_functional() -> None:
    square = SquarePore(side_nm=6.0, length_nm=12.0, layer_width_nm=6.0)
    droplet = square.interface("droplet_w1c0")
    values = {"theta": 140.0, "delta": 0.02, "phi": 0.05}
    terms = droplet.surface_terms(**values)
    expected = terms.liquid_liquid_area - math.cos(math.radians(140.0)) * terms.liquid_solid_area
    assert droplet.capillary_energy(**values) == pytest.approx(expected, rel=1e-12)


def test_square_pore_rejects_layer_narrower_than_pore() -> None:
    with pytest.raises(ValidationError, match="меньше стороны"):
        SquarePore(side_nm=6.0, length_nm=12.0, layer_width_nm=5.0)
