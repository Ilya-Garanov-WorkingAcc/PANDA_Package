"""Morphology: x–z density maps (averaged over y) of DECAN and WATER, every 10th frame of 2–10 ns."""
import numpy as np
import mdtraj as md
D = "/mnt/c/Phystech/Computing_Physics_Center/Projects/PANDA_Reborn/graphite_decane_tip4p_1.0"
OUT = "/home/igaranov/projects/PANDA_Dev/smoke/out"
top = md.load_topology(f"{D}/graphite_decane_tip4p.gro")
sel = {r: top.select(f"resname {r}") for r in ("DECAN", "WATER")}
sel["WATER"] = top.select("resname WATER and name OW")          # water: oxygens
Lx, Ly, Lz = 17.27104, 5.12817, 11.0
nx, nz = 70, 44
maps = {r: np.zeros((nx, nz)) for r in sel}
nfr = 0
for chunk in md.iterload(f"{D}/graphite_decane_tip4p.xtc", top=top, chunk=100, stride=10):
    for r, idx in sel.items():
        xyz = chunk.xyz[:, idx, :]
        x, z = (xyz[..., 0] % Lx).ravel(), (xyz[..., 2] % Lz).ravel()
        maps[r] += np.histogram2d(x, z, bins=(nx, nz), range=((0, Lx), (0, Lz)))[0]
    nfr += chunk.n_frames
vol = (Lx / nx) * Ly * (Lz / nz)
for r in maps:
    maps[r] /= nfr * vol
    np.save(f"{OUT}/graphite_map_{r}.npy", maps[r])
print("frames used:", nfr)
for r in maps:
    m = maps[r] / maps[r].max()
    print(f"\n{r}: x -> (0..{Lx}), z up (0..{Lz}); ' .:-=+*#' by fraction of the maximum")
    chars = " .:-=+*#"
    for j in range(nz - 1, -1, -2):
        print(f"{(j + 0.5) * Lz / nz:5.2f} |" + "".join(chars[min(int(v * 8), 7)] for v in m[:, j]))
