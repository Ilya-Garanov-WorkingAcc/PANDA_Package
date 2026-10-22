"""Parameters: declaration checks, units, parsing of optimize / fixed / bounds / initial."""

import math

import numpy as np
import pytest
from pydantic import ValidationError

from panda_md.core.enums import Unit
from panda_md.core.errors import (
    InvalidBoundsError,
    MissingParameterValueError,
    ParameterConflictError,
    UnknownParameterError,
)
from panda_md.core.parameters import (
    CONTACT_ANGLE,
    PROFILE_SHIFT,
    VOLUME_FRACTION,
    WALL_LAYER_THICKNESS,
    Parameter,
    ParameterSet,
)


def make_set() -> ParameterSet:
    return ParameterSet([CONTACT_ANGLE, WALL_LAYER_THICKNESS, VOLUME_FRACTION, PROFILE_SHIFT])


def test_parameter_rejects_reversed_bounds() -> None:
    with pytest.raises(ValidationError):
        Parameter(name="x", default=None, bounds=(1.0, 0.0))


def test_parameter_rejects_default_outside_bounds() -> None:
    with pytest.raises(ValidationError):
        Parameter(name="x", default=2.0, bounds=(0.0, 1.0))


def test_parameter_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        Parameter(name="x", default=None, bounds=(0.0, 1.0), bound=(0.0, 1.0))  # type: ignore[call-arg]


def test_parameter_is_immutable() -> None:
    with pytest.raises(ValidationError):
        CONTACT_ANGLE.default = 90.0  # type: ignore[misc]


def test_degree_parameter_converts_to_radians_and_back() -> None:
    assert CONTACT_ANGLE.unit is Unit.DEGREE
    assert CONTACT_ANGLE.to_internal(180.0) == pytest.approx(math.pi)
    assert CONTACT_ANGLE.to_user(math.pi / 2) == pytest.approx(90.0)
    assert WALL_LAYER_THICKNESS.to_internal(0.05) == 0.05


def test_resolve_splits_free_and_fixed_in_parameter_order() -> None:
    spec = make_set().resolve(["delta", "theta"], {"phi": 0.3}, {}, {}, {})
    assert spec.free_names == ("theta", "delta")
    assert spec.fixed_values == {"phi": 0.3, "shift": 0.0}
    assert spec.lower_bounds.tolist() == pytest.approx([0.0, 0.0])
    assert spec.upper_bounds.tolist() == pytest.approx([math.pi, 0.1])


def test_pack_unpack_round_trip_keeps_fixed_values() -> None:
    spec = make_set().resolve(["theta", "delta"], {"phi": 0.3}, {}, {}, {})
    values = spec.unpack(np.array([2.0, 0.04]))
    assert list(values) == ["theta", "delta", "phi", "shift"]
    assert values == {"theta": 2.0, "delta": 0.04, "phi": 0.3, "shift": 0.0}
    assert spec.pack(values).tolist() == [2.0, 0.04]


def test_resolve_rejects_unknown_name() -> None:
    with pytest.raises(UnknownParameterError, match="available:"):
        make_set().resolve(["thetta"], {"phi": 0.3}, {}, {}, {})


def test_resolve_rejects_parameter_both_free_and_fixed() -> None:
    with pytest.raises(ParameterConflictError):
        make_set().resolve(["theta"], {"theta": 120.0, "phi": 0.3}, {}, {}, {})


def test_resolve_requires_value_for_fixed_phi() -> None:
    with pytest.raises(MissingParameterValueError, match="phi"):
        make_set().resolve(["theta"], {}, {}, {}, {})


def test_resolve_takes_phi_from_data_when_not_given() -> None:
    spec = make_set().resolve(["theta"], {}, {}, {}, {"phi": 0.25})
    assert spec.fixed_values["phi"] == 0.25


def test_resolve_intersects_user_bounds_with_parameter_bounds() -> None:
    spec = make_set().resolve(["theta"], {"phi": 0.3}, {"theta": (100.0, 200.0)}, {}, {})
    assert spec.lower_bounds[0] == pytest.approx(math.radians(100.0))
    assert spec.upper_bounds[0] == pytest.approx(math.pi)


def test_resolve_rejects_initial_value_for_fixed_parameter() -> None:
    with pytest.raises(ParameterConflictError):
        make_set().resolve(["theta"], {"phi": 0.3}, {}, {"delta": 0.01}, {})


def test_parameter_set_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="declared twice"):
        ParameterSet([CONTACT_ANGLE, CONTACT_ANGLE])


def test_resolve_rejects_bounds_outside_parameter_range() -> None:
    with pytest.raises(InvalidBoundsError, match="do not intersect"):
        make_set().resolve(["theta"], {"phi": 0.3}, {"theta": (190.0, 200.0)}, {}, {})


def test_resolve_rejects_bounds_for_fixed_parameter() -> None:
    with pytest.raises(ParameterConflictError, match="not free"):
        make_set().resolve(["theta"], {"phi": 0.3}, {"phi": (0.5, 0.6)}, {}, {})


def test_resolve_rejects_non_finite_bounds() -> None:
    with pytest.raises(InvalidBoundsError, match="not finite"):
        make_set().resolve(["theta"], {"phi": 0.3}, {"theta": (math.nan, 150.0)}, {}, {})


def test_resolve_rejects_initial_value_outside_bounds() -> None:
    with pytest.raises(InvalidBoundsError, match="starting value"):
        make_set().resolve(["theta"], {"phi": 0.3}, {"theta": (100.0, 150.0)}, {"theta": 170.0}, {})
