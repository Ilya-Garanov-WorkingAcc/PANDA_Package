"""Сквозной путь: частицы из модели → траектория → центрирование и бины → профиль по блокам → фит.

Координата z частиц разыгрывается по профилю модели (одномерное распределение — всё, что видит профиль плотности),
x и y — равномерно. Подложка — три слоя атомов (1 нм), пора пересекает границу ящика и дрейфует между кадрами.
Проверяется путь данных, а не трёхмерная геометрия формы (облака точек по `contains` — фаза 6).
"""

import numpy as np
import pytest

from panda_md import ArrayTrajectory, InterfaceModel, ProfileCentering, SlitPore, density_profile

DENSITY_BULK = 33.0
WIDTH_NM = 4.0
LENGTH_NM = 8.0
ASPECT_RATIO = 0.5
BOX_NM = np.array([LENGTH_NM, ASPECT_RATIO * LENGTH_NM, 5.0])  # подложка — 1 нм
LOWER_WALL_NM = 3.2
FRAME_COUNT = 40
TIME_STEP_PS = 5.0
SAMPLING_POINTS = 20001
SEED = 11
MAX_ANGLE_ERROR_DEG = 1.5


def sample_heights(
    model: InterfaceModel, values: dict[str, float], count: int, generator: np.random.Generator
) -> np.ndarray:
    """z в системе поры (нм) с плотностью ∝ ρ_model(z): обратная функция распределения на мелкой сетке."""
    lower, upper = model.pore.geometry().coordinate_bounds()
    grid = np.linspace(lower, upper, SAMPLING_POINTS)
    density = model.profile(grid, **values)
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(grid))])
    heights = np.interp(generator.uniform(0.0, cumulative[-1], count), cumulative, grid)
    return heights * model.pore.characteristic_size_nm()


def particle_trajectory(model: InterfaceModel, values: dict[str, float]) -> ArrayTrajectory:
    generator = np.random.default_rng(SEED)
    pore = model.pore
    count = round(values["phi"] * DENSITY_BULK * pore.volume_nm3())
    grid = np.array([[x, y] for x in np.arange(0.0, BOX_NM[0], 0.4) for y in np.arange(0.0, BOX_NM[1], 0.4)])
    frames = []
    for frame_index in range(FRAME_COUNT):
        drift = 0.02 * frame_index  # стенки и жидкость вместе дрейфуют вдоль z
        lower = LOWER_WALL_NM + drift
        walls = np.vstack(
            [
                np.column_stack([grid, np.full(len(grid), lower)]),
                np.column_stack([grid, np.full(len(grid), lower + WIDTH_NM)]),
                np.column_stack([grid, np.full(len(grid), lower - 0.5)]),  # середина подложки толщиной 1 нм
            ]
        )
        lateral = generator.uniform(0.0, 1.0, size=(count, 2)) * BOX_NM[:2]
        heights = sample_heights(model, values, count, generator) + lower + WIDTH_NM / 2.0
        frames.append(np.mod(np.vstack([walls, np.column_stack([lateral, heights])]), BOX_NM))
    wall_count = 3 * len(grid)
    return ArrayTrajectory(
        positions_nm=np.array(frames),
        boxes_nm=np.tile(BOX_NM, (FRAME_COUNT, 1)),
        times_ps=TIME_STEP_PS * np.arange(FRAME_COUNT),
        groups={"walls": np.arange(wall_count), "liquid": wall_count + np.arange(count)},
    )


@pytest.mark.parametrize(
    ("shape_name", "truth", "centering"),
    [
        ("roll", {"theta": 120.0, "delta": 0.03, "phi": 0.4}, ProfileCentering.PORE),
        ("droplet", {"theta": 130.0, "delta": 0.03, "phi": 0.05}, ProfileCentering.PORE),
        ("droplet", {"theta": 130.0, "delta": 0.03, "phi": 0.05}, ProfileCentering.LIQUID),
    ],
    ids=["roll-pore", "droplet-pore", "droplet-liquid"],
)
def test_angle_recovered_from_particles(shape_name: str, truth: dict[str, float], centering: ProfileCentering) -> None:
    pore = SlitPore(width_nm=WIDTH_NM, length_nm=LENGTH_NM, base_aspect_ratio=ASPECT_RATIO, wall_selection="walls")
    model = pore.interface(shape_name)
    trajectory = particle_trajectory(model, truth)
    found = SlitPore.from_trajectory(trajectory, walls="walls")
    assert found.width_nm == pytest.approx(WIDTH_NM)
    profiles = density_profile(trajectory, pore, "liquid", centering=centering, bins=100, block="50 ps")
    assert profiles.block_count == 4
    optimize = ["theta", "delta"] if centering is ProfileCentering.PORE else ["theta"]
    fixed = {} if centering is ProfileCentering.PORE else {"delta": truth["delta"]}
    result = model.fit(profiles.mean(), density_bulk_per_nm3=DENSITY_BULK, optimize=optimize, fixed=fixed)
    assert result.values["theta"] == pytest.approx(truth["theta"], abs=MAX_ANGLE_ERROR_DEG)
    assert result.values["phi"] == pytest.approx(truth["phi"], rel=0.02)
