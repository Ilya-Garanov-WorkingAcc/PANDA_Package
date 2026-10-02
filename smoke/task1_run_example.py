"""Задача 1, п.2: прогон цепочки примера PANDA на реальных данных.
(а) Система из примера (calcite–decane–TIP4P). Доступен только 1 кадр, поэтому
    get_density_profile (он считает кадры по time/timestep) вызвать нельзя;
    повторяем его тело теми же функциями библиотеки и параметрами ноутбука-примера.
(б) Бензол на подложке, 5001 кадр: профили из npys/ (посчитаны ноутбуком пользователя
    в фиксированной системе координат ящика) -> block_average_density_profile ->
    profile_approx_from_array(roll, alpha). Геометрия — как в theta_contact_panda.ipynb.
"""
import warnings
import numpy as np
import mdtraj as md
import panda
from panda.density_profile import get_numerical_density_profile
from panda.utils import get_center_pbc, apply_pbc
warnings.filterwarnings("ignore")
R = "/mnt/c/Phystech/Computing_Physics_Center/Projects/roughness_2/angles"
print("=== (а) calcite–decane, пример из репозитория, 1 кадр")
S = f"{R}/Dev_PANDA/Dev_PANDA/systems/calcite_decane_tip4p_sc_2.5_offset"
t = md.load(f"{S}/cal_dec_tip4p.xtc", top=f"{S}/cal_dec_tip4p.gro")
pos = t.xyz[:, t.top.select("resname DECAN"), :]
box = t.unitcell_lengths[:, np.newaxis, :]
pos = apply_pbc(pos - get_center_pbc(pos, box) + box / 2, box)
axis, dens = get_numerical_density_profile(pos[0], box[0], 200, center=True)
H, l, phi, rho_bulk = 6.0, 11.6235 / 6.0, 0.5, 3.0896 * 10  # параметры ноутбука-примера
_, _, res = panda.profile_approx_from_array(dens, axis, rho_bulk, l, phi, H,
                                            interface_type="roll", display=False, samples=5)
print(f"theta = {np.rad2deg(res['theta']):.2f} deg, delta = {res['delta']:.4f}, mae = {res['mae']:.4f}")
print("\n=== (б) BENZ на подложке SOLD, 5001 кадр, профили из npys/")
axises = np.load(f"{R}/npys/axises_benz.npy")
denses = np.load(f"{R}/npys/denses_benz.npy")
Lx, Lz, z_wall = 47.8981, 13.4607, 3.147                 # нм, из nvt_prod.gro: ящик и верх SOLD
H = Lz - z_wall
l = Lx / H
eq = int(0.4 * len(denses))                          # как в ноутбуке: 40% на уравновешивание
ax_b, de_b = panda.block_average_density_profile(axises[eq:], denses[eq:], block_size=600)
print(f"H = {H:.3f} nm, l = {l:.3f}, blocks = {len(de_b)}")
for i in range(len(de_b)):
    z_box = ax_b[i] + Lz / 2                          # оси npys: центры бинов - Lz/2
    # PANDA ждёт ось на весь ящик (включая подложку), иначе окно фита пустое (см. отчёт).
    z = (z_box - (z_wall + H / 2) + Lz / 2) % Lz - Lz / 2
    order = np.argsort(z)
    z, d = z[order], de_b[i][order]
    rho_b = np.median(d[d > 0.8 * d.max()])           # оценка ρ_bulk по плато (эвристика)
    inside = np.abs(z) < H / 2
    phi_i = np.trapezoid(d[inside] / rho_b, z[inside] / H)
    _, _, r = panda.profile_approx_from_array(d, z, rho_b, l, phi_i, H,
                                              interface_type="roll", display=False, samples=5)
    print(f"block {i}: rho_bulk = {rho_b:.3f} nm^-3, phi = {phi_i:.3f}, "
          f"theta = {np.rad2deg(r['theta']):.2f} deg, delta = {r['delta']:.4f}, mae = {r['mae']:.4f}")
