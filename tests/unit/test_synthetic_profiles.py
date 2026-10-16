"""Синтетический профиль: как МД — среднее по бину и пуассоновский шум."""

import numpy as np
import pytest
from scipy.integrate import trapezoid

from panda_md import SlitPore
from panda_md.core.enums import NoiseModel, ProfileSampling
from panda_md.synthetic.profiles import synthetic_profile

DENSITY_BULK = 30.0
PORE = SlitPore(width_nm=8.0, length_nm=16.0)
DROPLET = {"theta": 130.0, "delta": 0.03, "phi": 0.05}


def volume_fraction(density: np.ndarray, coordinate_nm: np.ndarray) -> float:
    return float(trapezoid(density / DENSITY_BULK, coordinate_nm / PORE.width_nm))


def test_bin_averaged_profile_keeps_the_volume_fraction_on_a_coarse_grid() -> None:
    """Гистограмма МД — среднее по бину: ∫ρ = φ при любом числе бинов; значения в центрах бинов этого не дают."""
    model = PORE.interface("droplet")
    centers = synthetic_profile(model, DROPLET, density_bulk_per_nm3=DENSITY_BULK, bin_count=50)
    averaged = synthetic_profile(
        model, DROPLET, density_bulk_per_nm3=DENSITY_BULK, bin_count=50, sampling=ProfileSampling.BIN_AVERAGE
    )
    assert volume_fraction(averaged.density_per_nm3, averaged.coordinate_nm) == pytest.approx(0.05, rel=1e-3)
    assert volume_fraction(centers.density_per_nm3, centers.coordinate_nm) == pytest.approx(0.04935, abs=1e-4)


def test_poisson_noise_vanishes_where_there_is_no_liquid() -> None:
    model = PORE.interface("droplet")
    clean = synthetic_profile(model, DROPLET, density_bulk_per_nm3=DENSITY_BULK)
    noisy = [
        synthetic_profile(
            model, DROPLET, density_bulk_per_nm3=DENSITY_BULK, noise_std=0.02, noise_model=NoiseModel.POISSON, seed=seed
        )
        for seed in range(20)
    ]
    empty = clean.density_per_nm3 == 0.0
    assert all(np.all(profile.density_per_nm3[empty] == 0.0) for profile in noisy)
    fractions = [volume_fraction(profile.density_per_nm3, profile.coordinate_nm) for profile in noisy]
    assert float(np.mean(fractions)) == pytest.approx(0.05, rel=0.01)  # без обрезки — без смещения φ
    constant = [
        synthetic_profile(model, DROPLET, density_bulk_per_nm3=DENSITY_BULK, noise_std=0.02, seed=seed)
        for seed in range(20)
    ]
    biased = float(np.mean([volume_fraction(profile.density_per_nm3, profile.coordinate_nm) for profile in constant]))
    assert biased > 0.051  # постоянная σ с обрезкой в пустых бинах завышает φ
