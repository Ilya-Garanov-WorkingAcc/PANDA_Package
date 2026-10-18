"""Task 1, item 2: a full run of the previous PANDA chain on a real trajectory.
Data: PANDA_Reborn/graphite_decane_tip4p_1.0 — graphite (frozen) + n-decane (1095) + TIP4P (11807),
NVT 370 K, 10 ns, 5001 frames every 2 ps, box 17.271 x 5.128 x 11.000 nm.
Step 1: panda.get_density_profile (as in examples/contact_angle), no workarounds or library changes.
"""
import time
import numpy as np
import panda
D = "/mnt/c/Phystech/Computing_Physics_Center/Projects/PANDA_Reborn/graphite_decane_tip4p_1.0"
OUT = "/home/igaranov/projects/PANDA_Dev/smoke/out"
t0 = time.time()
axises, denses = panda.get_density_profile(
    f"{D}/graphite_decane_tip4p.xtc", f"{D}/graphite_decane_tip4p.gro",
    residue="DECAN", sl=200,
    chunk_length=200,   # ps -> 100 frames per chunk (memory limit)
    begin_time=0, time=10000, timestep=2, units="ps",
)
print(f"profiles: {denses.shape}, {time.time() - t0:.0f} s")
np.save(f"{OUT}/graphite_decan_axises.npy", axises)
np.save(f"{OUT}/graphite_decan_denses.npy", denses)
