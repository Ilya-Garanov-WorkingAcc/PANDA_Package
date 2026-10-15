"""Синтетическая валидация: форма → профиль с шумом → фит → восстановленные параметры.

Пороги и найденные закономерности описаны в tests/README.md («Синтетика», «O2», «O3»).
"""

import numpy as np
import pytest

from panda_md import Frame, SlitPore, SquarePore
from panda_md.core.model import InterfaceModel
from panda_md.synthetic.profiles import synthetic_profile

DENSITY_BULK = 30.0
NOISE_STD = 0.005  # шум 0.5 % ρ_bulk на бин, 200 бинов
SEEDS = range(5)
MAX_ANGLE_ERROR_DEG = 1.5  # в каждом прогоне
MEAN_ANGLE_ERROR_DEG = 1.0  # в среднем по прогонам (критерий ТЗ |Δθ| < 1°)

SLIT = SlitPore(width_nm=8.0, length_nm=16.0)
SQUARE = SquarePore(side_nm=6.0, length_nm=12.0, layer_width_nm=6.0)
CASES: list[tuple[InterfaceModel, dict[str, float]]] = [
    (SLIT.interface("roll"), {"theta": 120.0, "delta": 0.03, "phi": 0.4}),
    (SLIT.interface("droplet"), {"theta": 130.0, "delta": 0.03, "phi": 0.05}),
    (SQUARE.interface("droplet_w1c0"), {"theta": 140.0, "delta": 0.02, "phi": 0.05}),
    (SQUARE.interface("filament_w1c0"), {"theta": 130.0, "delta": 0.02, "phi": 0.1}),
    (SQUARE.interface("filament_w1c0"), {"theta": 60.0, "delta": 0.02, "phi": 0.05}),
]


def case_id(case: tuple[InterfaceModel, dict[str, float]]) -> str:
    model, truth = case
    return f"{model.pore.name}/{model.shape.name}/θ={truth['theta']:.0f}"


def angle_errors(model: InterfaceModel, truth: dict[str, float], density_bulk_per_nm3: float) -> list[float]:
    errors = []
    for seed in SEEDS:
        profile = synthetic_profile(model, truth, density_bulk_per_nm3=DENSITY_BULK, noise_std=NOISE_STD, seed=seed)
        result = model.fit(profile, density_bulk_per_nm3=density_bulk_per_nm3)
        errors.append(abs(result.values["theta"] - truth["theta"]))
    return errors


def assert_angle_recovered(errors: list[float]) -> None:
    assert max(errors) < MAX_ANGLE_ERROR_DEG, errors
    assert float(np.mean(errors)) < MEAN_ANGLE_ERROR_DEG, errors


@pytest.mark.slow
@pytest.mark.parametrize("case", CASES, ids=case_id)
def test_angle_and_wall_layer_recovered_with_bulk_density(case: tuple[InterfaceModel, dict[str, float]]) -> None:
    model, truth = case
    assert_angle_recovered(angle_errors(model, truth, DENSITY_BULK))


@pytest.mark.slow
@pytest.mark.parametrize(
    ("optimize", "fixed"),
    [(["phi"], {"theta": 120.0, "delta": 0.03}), (["delta", "phi"], {"theta": 120.0}), (["theta", "delta", "phi"], {})],
    ids=["phi", "delta-phi", "theta-delta-phi"],
)
def test_task_modes_recover_volume_fraction(optimize: list[str], fixed: dict[str, float]) -> None:
    model, truth = CASES[0]
    for seed in SEEDS:
        profile = synthetic_profile(model, truth, density_bulk_per_nm3=DENSITY_BULK, noise_std=NOISE_STD, seed=seed)
        result = model.fit(profile, optimize=optimize, fixed=fixed, density_bulk_per_nm3=DENSITY_BULK)
        assert result.values["phi"] == pytest.approx(truth["phi"], abs=0.01)
        assert result.values["theta"] == pytest.approx(truth["theta"], abs=MAX_ANGLE_ERROR_DEG)


@pytest.mark.slow
@pytest.mark.parametrize("case", CASES[1:4], ids=case_id)
def test_mean_normalization_recovers_compact_shapes(case: tuple[InterfaceModel, dict[str, float]]) -> None:
    """O2: без ρ_bulk φ и θ определяются вместе — для капель и нитей (у валика нет, см. README)."""
    model, truth = case
    for seed in SEEDS:
        profile = synthetic_profile(model, truth, density_bulk_per_nm3=DENSITY_BULK, noise_std=NOISE_STD, seed=seed)
        result = model.fit(profile, optimize=["theta", "delta", "phi"])
        assert result.values["phi"] == pytest.approx(truth["phi"], abs=0.01)
        assert result.values["theta"] == pytest.approx(truth["theta"], abs=3.0)
        assert result.density_bulk_derived_per_nm3 == pytest.approx(DENSITY_BULK, rel=0.15)


def test_mean_normalization_with_known_phi_recovers_roll() -> None:
    """O2: валику в нормировке mean нужна известная φ (из построения системы)."""
    model, truth = CASES[0]
    profile = synthetic_profile(model, truth, density_bulk_per_nm3=DENSITY_BULK)
    result = model.fit(profile, fixed={"phi": truth["phi"]})
    assert result.values["theta"] == pytest.approx(truth["theta"], abs=0.5)
    assert result.density_bulk_derived_per_nm3 == pytest.approx(DENSITY_BULK, rel=1e-3)


WALL_OFFSET = 0.05  # ошибка положения стенок: 5 % размера поры


@pytest.mark.parametrize("case", CASES[1:4], ids=case_id)
def test_wall_offset_goes_into_wall_layer_for_one_wall_shapes(case: tuple[InterfaceModel, dict[str, float]]) -> None:
    """O3: у форм на одной стенке сдвиг стенок поглощает δ, θ не меняется — центрирование по поре."""
    model, truth = case
    profile = synthetic_profile(model, truth, density_bulk_per_nm3=DENSITY_BULK, offset=WALL_OFFSET)
    result = model.fit(profile, density_bulk_per_nm3=DENSITY_BULK)
    assert result.values["theta"] == pytest.approx(truth["theta"], abs=0.1)


def test_wall_offset_needs_free_shift_for_roll() -> None:
    """O3: валик касается обеих стенок — сдвиг поглощает только параметр shift."""
    model, truth = CASES[0]
    profile = synthetic_profile(model, truth, density_bulk_per_nm3=DENSITY_BULK, offset=WALL_OFFSET)
    with_shift = model.fit(profile, optimize=["theta", "delta", "shift"], density_bulk_per_nm3=DENSITY_BULK)
    without_shift = model.fit(profile, density_bulk_per_nm3=DENSITY_BULK)
    assert with_shift.values["theta"] == pytest.approx(truth["theta"], abs=0.5)
    assert abs(without_shift.values["theta"] - truth["theta"]) > 2.0


@pytest.mark.parametrize("case", CASES[1:4], ids=case_id)
def test_liquid_frame_recovers_angle_with_known_wall_layer(case: tuple[InterfaceModel, dict[str, float]]) -> None:
    model, truth = case
    profile = synthetic_profile(model, truth, density_bulk_per_nm3=DENSITY_BULK, frame=Frame.LIQUID)
    result = model.fit(profile, optimize=["theta"], fixed={"delta": truth["delta"]}, density_bulk_per_nm3=DENSITY_BULK)
    assert result.values["theta"] == pytest.approx(truth["theta"], abs=0.1)
