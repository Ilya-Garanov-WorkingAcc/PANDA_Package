"""Профиль плотности и его приведение к безразмерному виду."""

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from panda_md.core.enums import Frame, Normalization
from panda_md.core.errors import InvalidProfileError
from panda_md.core.profile import DensityProfile
from panda_md.pores.slit.pore import SlitPore

SLIT = SlitPore(width_nm=8.0, length_nm=16.0)


def make_profile(**changes: object) -> DensityProfile:
    fields: dict[str, object] = {
        "coordinate_nm": np.linspace(-4.0, 4.0, 50),
        "density_per_nm3": np.full(50, 10.0),
        "particle_count": 1000.0,
        "pore_volume_nm3": 2048.0,
    }
    fields.update(changes)
    return DensityProfile.model_validate(fields)


def test_profile_rejects_length_mismatch() -> None:
    with pytest.raises(ValidationError, match="длины"):
        make_profile(density_per_nm3=np.ones(49))


def test_profile_rejects_nan() -> None:
    density = np.full(50, 10.0)
    density[3] = np.nan
    with pytest.raises(ValidationError, match="NaN"):
        make_profile(density_per_nm3=density)


def test_profile_rejects_decreasing_coordinate() -> None:
    with pytest.raises(ValidationError, match="возрастает"):
        make_profile(coordinate_nm=np.linspace(4.0, -4.0, 50))


def test_profile_rejects_negative_density() -> None:
    with pytest.raises(ValidationError, match="отрицательна"):
        make_profile(density_per_nm3=np.full(50, -1.0))


def test_profile_arrays_are_read_only() -> None:
    profile = make_profile()
    with pytest.raises(ValueError, match="read-only"):
        profile.density_per_nm3[0] = 1.0


def test_profile_round_trips_through_json() -> None:
    profile = make_profile(frame=Frame.LIQUID)
    restored = DensityProfile.model_validate_json(profile.model_dump_json())
    assert restored.frame is Frame.LIQUID
    np.testing.assert_array_equal(restored.coordinate_nm, profile.coordinate_nm)


def test_slit_reduces_by_width_and_bulk_density() -> None:
    reduced = SLIT.reduce(make_profile(), Normalization.BULK, 20.0)
    assert reduced.coordinate[0] == pytest.approx(-0.5)
    assert reduced.density[0] == pytest.approx(0.5)


def test_slit_reduces_by_mean_density() -> None:
    reduced = SLIT.reduce(make_profile(), Normalization.MEAN, None)
    assert reduced.density_scale_per_nm3 == pytest.approx(1000.0 / 2048.0)


def test_bulk_normalization_needs_bulk_density() -> None:
    with pytest.raises(InvalidProfileError, match="density_bulk"):
        SLIT.reduce(make_profile(), Normalization.BULK, None)


def test_mean_normalization_needs_particle_count() -> None:
    with pytest.raises(InvalidProfileError, match="particle_count"):
        SLIT.reduce(make_profile(particle_count=None), Normalization.MEAN, None)


def test_profile_rejects_all_zero_density() -> None:
    with pytest.raises(ValidationError, match="профиль пуст"):
        make_profile(density_per_nm3=np.zeros(50))


def test_saved_files_and_results_record_the_library_version(tmp_path: Path) -> None:
    """Итоговый аудит: ни в результате, ни в файлах не было версии PANDA — для статьи это воспроизводимость."""
    import json

    import panda_md
    from panda_md import SlitPore
    from panda_md.synthetic.profiles import synthetic_profile

    model = SlitPore(width_nm=8.0, length_nm=16.0).interface("roll")
    profile = synthetic_profile(model, {"theta": 120.0, "delta": 0.03, "phi": 0.4}, density_bulk_per_nm3=30.0)
    assert model.fit(profile, density_bulk_per_nm3=30.0).library_version == panda_md.__version__
    for name in ("profile.npz", "profile.csv"):
        profile.save(tmp_path / name)
        assert DensityProfile.load(tmp_path / name) == profile
    with np.load(tmp_path / "profile.npz") as archive:
        assert json.loads(str(archive["metadata"]))["panda_version"] == panda_md.__version__
    assert f'panda_version: "{panda_md.__version__}"' in (tmp_path / "profile.csv").read_text()


def test_csv_metadata_without_space_and_column_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "profile.csv"
    path.write_text('#frame: "liquid"\n#coordinate_nm,density_per_nm3\n-1,1\n0,2\n1,1\n')
    assert DensityProfile.load(path).frame.value == "liquid"
    path.write_text("# coordinate_nm,density_per_nm3\n-1,1,0.1\n0,2,0.1\n1,1,0.1\n")
    with pytest.raises(InvalidProfileError, match="столбц"):
        DensityProfile.load(path)
