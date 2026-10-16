"""Профили по реальным траекториям МД — вход для tests/md (сами траектории в репозиторий не входят).

Три системы (найденная пора сохраняется рядом с профилями, чтобы тесты сверили её с ручным расчётом):
- graphite: щель графита с деканом и водой TIP4P (PANDA_Reborn graphite_decane_tip4p_1.0), 10 нс, кадр — 2 пс;
  профиль воды (кислороды) после 2 нс, блоки по 1 нс;
- benzene: бензол в щели подложки SOLD (roughness_2/angles/base), 10 нс, кадр — 2 пс; после 4 нс, блоки по 1.2 нс;
- calcite: щель кальцита с деканом и водой (система примера старой PANDA), один стартовый кадр.

Запуск (нужен panda-md[md] и доступ к данным; корни — переменные PANDA_REBORN_DIR и PANDA_ROUGHNESS_DIR):
    uv run python tests/md/generate_md_profiles.py
Результат: tests/md/data/*_profiles.npz (ProfileSet), *_pore.json и *_frame.npz (один кадр: ящик и высоты атомов
подложки и жидкости — по нему тесты заново прогоняют поиск стенок и профиль) — коммитятся, тесты читают только их.
"""

import os
import time
from pathlib import Path

import numpy as np

from panda_md import SlitPore, density_profile, load_trajectory
from panda_md.core.trajectory import Trajectory

REBORN = Path(os.environ.get("PANDA_REBORN_DIR", "/mnt/c/Phystech/Computing_Physics_Center/Projects/PANDA_Reborn"))
ROUGHNESS = Path(os.environ.get("PANDA_ROUGHNESS_DIR", "/mnt/c/Phystech/Computing_Physics_Center/Projects/roughness_2"))
OUTPUT = Path(__file__).parent / "data"
BIN_COUNT = 200


def save_frame(name: str, trajectory: Trajectory, walls: str, liquid: str, time: str) -> None:
    """Высоты атомов подложки и жидкости и ящик одного кадра (float32, сжато: точность xtc — 0.001 нм)."""
    frame = trajectory.frame_at(time, [trajectory.select(walls), trajectory.select(liquid)])
    np.savez_compressed(
        OUTPUT / f"{name}_frame.npz",
        box_nm=frame.box_nm,
        wall_z_nm=frame.positions_nm[0][:, 2].astype(np.float32),
        liquid_z_nm=frame.positions_nm[1][:, 2].astype(np.float32),
    )


def save_pore(name: str, pore: SlitPore) -> None:
    (OUTPUT / f"{name}_pore.json").write_text(pore.model_dump_json(indent=2) + "\n")


def graphite() -> None:
    system = REBORN / "graphite_decane_tip4p_1.0"
    trajectory = load_trajectory(system / "graphite_decane_tip4p.tpr", system / "graphite_decane_tip4p.xtc")
    pore = SlitPore.from_trajectory(trajectory, walls="resname GRAPHITE")
    save_pore("graphite", pore)
    save_frame("graphite", trajectory, "resname GRAPHITE", "resname WATER and name OW", "2 ns")
    water = density_profile(trajectory, pore, "resname WATER and name OW", bins=BIN_COUNT, start="2 ns", block="1 ns")
    water.save(OUTPUT / "graphite_water_profiles.npz")


def benzene() -> None:
    system = ROUGHNESS / "angles" / "base"
    trajectory = load_trajectory(system / "nvt_prod.gro", system / "nvt_prod.xtc")
    pore = SlitPore.from_trajectory(trajectory, walls="resname SOLD")
    save_pore("benzene", pore)
    save_frame("benzene", trajectory, "resname SOLD", "resname BENZ", "4 ns")
    benzene_profiles = density_profile(trajectory, pore, "resname BENZ", bins=BIN_COUNT, start="4 ns", block="1.2 ns")
    benzene_profiles.save(OUTPUT / "benzene_profiles.npz")


def calcite() -> None:
    system = ROUGHNESS / "angles" / "Dev_PANDA" / "Dev_PANDA" / "systems" / "calcite_decane_tip4p_sc_2.5_offset"
    trajectory = load_trajectory(system / "cal_dec_tip4p.gro", system / "cal_dec_tip4p.xtc")
    pore = SlitPore.from_trajectory(trajectory, walls="resname CAL")
    save_pore("calcite", pore)
    save_frame("calcite", trajectory, "resname CAL", "resname DECAN", "0 ps")
    density_profile(trajectory, pore, "resname DECAN", bins=BIN_COUNT).save(OUTPUT / "calcite_decane_profiles.npz")
    density_profile(trajectory, pore, "resname WATER and name OW", bins=BIN_COUNT).save(
        OUTPUT / "calcite_water_profiles.npz"
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for system in (graphite, benzene, calcite):
        started = time.perf_counter()
        system()
        print(f"{system.__name__}: {time.perf_counter() - started:.0f} с")


if __name__ == "__main__":
    main()
