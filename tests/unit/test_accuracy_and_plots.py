"""Исследование точности на синтетике."""

import numpy as np
import pytest
from pydantic import ValidationError

from panda_md import SlitPore
from panda_md.analysis.accuracy import (
    AccuracyCase,
    AccuracyStudy,
    accuracy_study,
    angle_errors,
    effective_noise_std,
)

DENSITY_BULK = 30.0
PORE = SlitPore(width_nm=8.0, length_nm=16.0)
ROLL_TRUTH = {"theta": 120.0, "delta": 0.03, "phi": 0.4}


def test_without_noise_the_angle_is_recovered() -> None:
    errors = angle_errors(
        PORE.interface("roll"),
        AccuracyCase(truth=ROLL_TRUTH, noise_std=0.0),
        density_bulk_per_nm3=DENSITY_BULK,
        seeds=[0],
    )
    assert abs(errors[0]) < 0.5


def test_study_summarises_errors_per_condition() -> None:
    cases = {
        "0.5": (0.5, AccuracyCase(truth=ROLL_TRUTH, noise_std=0.005)),
        "2": (2.0, AccuracyCase(truth=ROLL_TRUTH, noise_std=0.02)),
    }
    study = accuracy_study(PORE.interface("roll"), "шум, %", cases, density_bulk_per_nm3=DENSITY_BULK, seeds=range(3))
    assert study.errors_deg.shape == (2, 3)
    assert study.mean_absolute()[1] > study.mean_absolute()[0]  # больше шума — больше ошибка
    np.testing.assert_allclose(study.maximum_absolute(), np.max(np.abs(study.errors_deg), axis=1))
    table = study.table()
    assert table.splitlines()[0].startswith("| шум, %") and "| 2 |" in table
    with pytest.raises(ValidationError, match="подписи"):
        AccuracyStudy(
            shape_name="roll", varied="x", labels=("a",), values=np.array([1.0, 2.0]), errors_deg=np.zeros((2, 1))
        )


def test_noise_is_rescaled_with_the_bin_count() -> None:
    """Та же траектория, разбитая мельче, — шум на бин растёт как √(бинов / 200)."""
    assert effective_noise_std(AccuracyCase(truth=ROLL_TRUTH, noise_std=0.01)) == pytest.approx(0.01)
    assert effective_noise_std(AccuracyCase(truth=ROLL_TRUTH, noise_std=0.01, bin_count=400)) == pytest.approx(
        0.01 * np.sqrt(2.0)
    )
    with pytest.raises(ValidationError):
        AccuracyCase(truth=ROLL_TRUTH, noise_std=-0.1)
