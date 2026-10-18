"""Profiles from real MD trajectories — input for tests/md (the trajectories themselves are not in the repository).

Three systems (the found pore is saved next to the profiles so that the tests compare it with a manual calculation):
- graphite: graphite slit with decane and TIP4P water (PANDA_Reborn graphite_decane_tip4p_1.0), 10 ns, frame 2 ps;
  water profile (oxygens) after 2 ns, 1 ns blocks;
- benzene: benzene in a slit of the SOLD substrate (roughness_2/angles/base), 10 ns, frame 2 ps; after 4 ns,
  1.2 ns blocks;
- calcite: calcite slit with decane and water (the example system of old PANDA), one starting frame.

Run (needs panda-md[md] and access to the data; the roots are the PANDA_REBORN_DIR and PANDA_ROUGHNESS_DIR variables):
    uv run python tests/md/generate_md_profiles.py
Output: tests/md/data/*_profiles.npz (ProfileSet), *_pore.json and *_frame.npz (one frame: the box and the heights
of the substrate and liquid atoms — the tests rerun the wall search and the profile on it) — committed, the tests
read only them.
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
    """Substrate and liquid atom heights and the box of one frame (float32, compressed: xtc precision 0.001 nm)."""
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
        print(f"{system.__name__}: {time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main()
