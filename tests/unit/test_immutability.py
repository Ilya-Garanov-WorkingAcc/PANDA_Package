"""The fit result and settings are truly immutable and comparable — they can be stored and put into sets."""

import pytest

from panda_md import FitResult, FitSpec, SlitPore
from panda_md.core.profile import DensityProfile
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


def _profile(density: list[float]) -> DensityProfile:

    return DensityProfile(coordinate_nm=[0.0, 1.0, 2.0], density_per_nm3=density)


def test_models_with_arrays_compare_and_hash_by_value() -> None:
    """Pre-release audit: == and in failed (truth value of an array is ambiguous), hash — unhashable ndarray."""
    first, same, other = _profile([1.0, 2.0, 3.0]), _profile([1.0, 2.0, 3.0]), _profile([1.0, 2.0, 4.0])
    assert first == same and first != other
    assert first in [other, same]
    assert hash(first) == hash(same)
    assert len({first, same, other}) == 2


def test_copies_keep_arrays_read_only() -> None:
    import copy
    import pickle

    original = _profile([1.0, 2.0, 3.0])
    for duplicate in (copy.deepcopy(original), pickle.loads(pickle.dumps(original)), original.model_copy(deep=True)):
        assert duplicate == original
        with pytest.raises(ValueError, match="read-only"):
            duplicate.density_per_nm3[0] = 123.0


def test_model_copy_with_update_is_validated() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="density is negative"):
        _profile([1.0, 2.0, 3.0]).model_copy(update={"density_per_nm3": [-1.0, 5.0, 6.0]})
