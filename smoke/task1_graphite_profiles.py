"""Задача 1, п.2: полный рабочий прогон исходной цепочки PANDA на реальной траектории.
Данные: PANDA_Reborn/graphite_decane_tip4p_1.0 — графит (заморожен) + н-декан (1095) + TIP4P (11807),
NVT 370 K, 10 нс, 5001 кадр через 2 пс, ящик 17.271 x 5.128 x 11.000 нм.
Шаг 1: panda.get_density_profile (как в examples/contact_angle) — без обходов и правок библиотеки.
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
    chunk_length=200,   # пс -> 100 кадров на чанк (ограничение по памяти)
    begin_time=0, time=10000, timestep=2, units="ps",
)
print(f"profiles: {denses.shape}, {time.time() - t0:.0f} s")
np.save(f"{OUT}/graphite_decan_axises.npy", axises)
np.save(f"{OUT}/graphite_decan_denses.npy", denses)
