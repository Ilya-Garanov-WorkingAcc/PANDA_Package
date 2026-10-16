"""Эталон профиля плотности по траектории из старой PANDA (`utils.py`, `density_profile.py`, коммит 93c6dbb).

Старый путь: центр жидкости по круговому среднему (`get_center_pbc`), сдвиг в середину ящика и заворот
(`apply_pbc`), бины по всему ящику (`get_numerical_density_profile`). Новый аналог — центрирование по жидкости
(система LIQUID). Вход — случайное облако точек с фиксированным зерном: плотная полоса, пересекающая границу
ящика, и разреженный пар.

Запуск (нужно окружение со старой PANDA, см. ROADMAP «Фаза 0»):
    .venv-legacy/bin/python tests/regression/generate_density_reference.py
Результат: tests/regression/data/density_reference.npz — коммитится, тесты читают только его.
"""

import importlib
from pathlib import Path

import numpy as np

utils = importlib.import_module("panda.utils")
old_density = importlib.import_module("panda.density_profile")

OUTPUT = Path(__file__).parent / "data" / "density_reference.npz"
SEED = 20261016
FRAME_COUNT = 4
BOX_NM = np.array([4.0, 3.0, 10.0])
LIQUID_PARTICLES = 1500
VAPOUR_PARTICLES = 100
SLAB_HALF_WIDTH_NM = 1.5
BIN_COUNT = 80
TIME_STEP_PS = 2.0


def make_positions(generator: np.random.Generator) -> np.ndarray:
    """Кадры: полоса жидкости вокруг z ≈ 9.5 нм (пересекает верхнюю границу ящика) и пар по всему ящику."""
    frames = []
    for frame_index in range(FRAME_COUNT):
        slab_center = 9.5 + 0.1 * frame_index  # жидкость дрейфует — центр в каждом кадре свой
        liquid = generator.uniform(0.0, 1.0, size=(LIQUID_PARTICLES, 3)) * BOX_NM
        liquid[:, 2] = np.mod(
            slab_center + generator.uniform(-1.0, 1.0, LIQUID_PARTICLES) * SLAB_HALF_WIDTH_NM, BOX_NM[2]
        )
        vapour = generator.uniform(0.0, 1.0, size=(VAPOUR_PARTICLES, 3)) * BOX_NM
        frames.append(np.vstack([liquid, vapour]))
    return np.array(frames)


def main() -> None:
    positions = make_positions(np.random.default_rng(SEED))
    boxes = np.tile(BOX_NM, (FRAME_COUNT, 1))
    box = boxes[:, np.newaxis, :]
    centered = utils.apply_pbc(positions - utils.get_center_pbc(positions, box) + box / 2, box)
    profiles = [
        old_density.get_numerical_density_profile(centered[index], box[index], BIN_COUNT, center=True)[1]
        for index in range(FRAME_COUNT)
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT,
        positions_nm=positions,
        boxes_nm=boxes,
        times_ps=TIME_STEP_PS * np.arange(FRAME_COUNT),
        bin_count=BIN_COUNT,
        density_per_nm3=np.mean(profiles, axis=0),
        source="github.com/FlufffyMelon/PANDA@93c6dbb panda/utils.py, panda/density_profile.py",
    )
    print(f"записано в {OUTPUT}")


if __name__ == "__main__":
    main()
