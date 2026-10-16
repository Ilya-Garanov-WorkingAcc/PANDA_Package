"""Реальная МД: профили, посчитанные новым путём данных по трём системам (`generate_md_profiles.py`).

Ожидания — независимые: ручной расчёт стенок и результаты задачи 1 (docs/task1_current_panda.md).
- graphite: вода — несмачивающая перемычка (roll), θ ≈ 172° ± 7° по блокам; по 2D-карте вода отделена от графита
  плёнкой декана, поэтому θ близок к 180°;
- benzene: бензол на подложке SOLD смачивает её (плотность у стенок выше, чем в середине) — θ < 90°, формул
  смачивающего режима нет (O1), фит обязан упереться в границу режима и поднять флаг;
- calcite: стартовый кадр, подложка толстая и из разных атомов — проверяется только поиск стенок и то, что вся
  жидкость в поре.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import trapezoid

from panda_md import Frame, ProfileSet, SlitPore

DATA = Path(__file__).parent / "data"
WATER_DENSITY_BULK = 33.13  # ρ_bulk кислородов воды в графитовой системе: медиана 2D-карты в сердцевине фазы (задача 1)


def load_pore(name: str) -> SlitPore:
    return SlitPore.model_validate_json((DATA / f"{name}_pore.json").read_text())


def test_walls_are_found_as_computed_by_hand() -> None:
    """H — наибольший промежуток между атомами подложки через PBC (крайние атомы и длины ящиков — из .gro)."""
    graphite = load_pore("graphite")
    assert graphite.width_nm == pytest.approx(11.0 + 0.195 - 2.926, abs=1e-3)  # образ дна графита − его верх
    assert graphite.base_aspect_ratio == pytest.approx(5.12817 / 17.27104, abs=1e-5)
    benzene = load_pore("benzene")
    assert benzene.width_nm == pytest.approx(13.4607 - 3.147, abs=1e-3)  # задача 1: H = Lz − верх SOLD = 10.31
    calcite = load_pore("calcite")
    assert calcite.width_nm == pytest.approx(12.684 + 0.009 - 3.554, abs=1e-3)


@pytest.mark.parametrize(
    ("file_name", "particles"),
    [
        ("graphite_water_profiles.npz", 11807),
        ("benzene_profiles.npz", 16584),
        ("calcite_decane_profiles.npz", 13400),
        ("calcite_water_profiles.npz", 14458),
    ],
)
def test_whole_liquid_is_inside_the_pore(file_name: str, particles: int) -> None:
    profiles = ProfileSet.load(DATA / file_name)
    assert profiles.frame is Frame.PORE
    np.testing.assert_allclose(profiles.block_particle_counts, particles)


def test_graphite_water_is_a_non_wetting_roll_close_to_180_degrees() -> None:
    pore = load_pore("graphite")
    profiles = ProfileSet.load(DATA / "graphite_water_profiles.npz")
    assert profiles.block_frame_counts == (500,) * 8
    mean = profiles.mean()
    phi = trapezoid(mean.density_per_nm3, mean.coordinate_nm) / (WATER_DENSITY_BULK * pore.width_nm)
    assert phi == pytest.approx(0.487, abs=0.005)  # задача 1: φ = 0.487
    roll = pore.interface("roll")
    result = roll.fit(mean, density_bulk_per_nm3=WATER_DENSITY_BULK)
    assert 165.0 < result.values["theta"] <= 180.0
    assert 0.02 < result.values["delta"] < 0.1
    assert result.loss < 0.03
    block_angles = [
        roll.fit(profiles.block(index), density_bulk_per_nm3=WATER_DENSITY_BULK).values["theta"] for index in range(8)
    ]
    assert 165.0 < float(np.mean(block_angles)) < 178.0
    assert float(np.std(block_angles)) < 10.0


@pytest.mark.parametrize("density_bulk", [10.0, 15.0, 20.0, 25.0])
def test_benzene_wets_its_substrate_and_fit_flags_the_regime_boundary(density_bulk: float) -> None:
    """ρ_bulk бензола в этой системе неизвестна, поэтому вывод проверяется на всём разумном диапазоне."""
    pore = load_pore("benzene")
    profiles = ProfileSet.load(DATA / "benzene_profiles.npz")
    mean = profiles.mean()
    near_walls = mean.density_per_nm3[np.abs(mean.coordinate_nm) > 0.4 * pore.width_nm]
    middle = mean.density_per_nm3[np.abs(mean.coordinate_nm) < 0.1 * pore.width_nm]
    assert near_walls[near_walls > 0].mean() > 1.3 * middle.mean()  # вогнутая перемычка: шире у стенок
    result = pore.interface("roll").fit(mean, density_bulk_per_nm3=density_bulk)
    assert result.at_regime_boundary
    assert result.values["theta"] == pytest.approx(90.5, abs=0.1)
