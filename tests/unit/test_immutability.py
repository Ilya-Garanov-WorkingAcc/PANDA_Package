"""Результат и настройки фита действительно неизменяемы и сравнимы — их можно хранить и класть в множества."""

import pytest

from panda_md import FitResult, FitSpec, SlitPore
from panda_md.synthetic.profiles import synthetic_profile

DENSITY_BULK = 30.0
SLIT = SlitPore(width_nm=8.0, length_nm=16.0)


def fitted() -> FitResult:
    roll = SLIT.interface("roll")
    profile = synthetic_profile(roll, {"theta": 120.0, "delta": 0.03, "phi": 0.4}, density_bulk_per_nm3=DENSITY_BULK)
    return roll.fit(profile, fixed={"phi": 0.4}, bounds={"theta": (100.0, 170.0)}, density_bulk_per_nm3=DENSITY_BULK)


def test_result_values_cannot_be_changed() -> None:
    result = fitted()
    with pytest.raises(TypeError):
        result.values["theta"] = 0.0  # type: ignore[index]


def test_spec_dictionaries_cannot_be_changed() -> None:
    spec = fitted().spec
    with pytest.raises(TypeError):
        spec.fixed["phi"] = 0.1  # type: ignore[index]
    with pytest.raises(TypeError):
        spec.bounds["theta"] = (0.0, 1.0)  # type: ignore[index]


def test_equal_specs_are_hashable_and_equal() -> None:
    first = FitSpec(fixed={"phi": 0.4}, bounds={"theta": (100.0, 170.0)})
    second = FitSpec(fixed={"phi": 0.4}, bounds={"theta": (100.0, 170.0)})
    assert first == second
    assert hash(first) == hash(second)
    assert len({first, second}) == 1


def test_frozen_result_round_trips_through_json() -> None:
    result = fitted()
    restored = FitResult.model_validate_json(result.model_dump_json())
    assert restored.values == result.values
    assert restored.spec == result.spec
    assert dict(restored.spec.fixed) == {"phi": 0.4}
