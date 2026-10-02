"""Задача 1, п.2: полноценный рабочий тест PANDA на graphite–decane–TIP4P (PANDA_Reborn).

Морфология (smoke/task1_graphite_map.py): декан — перемычка (roll) между стенками графита,
шире у стенок => декан смачивает графит; вода — выпуклая перемычка (несмачивающая фаза).

(1) DECAN — библиотека как есть: профили из task1_graphite_profiles.py (panda.get_density_profile).
(2) WATER — те же функции библиотеки, заменён только остаток (get_density_profile допускает лишь DECAN).
Опорное значение θ — геометрический фит окружности к изолинии ρ = ρ_bulk/2 на 2D-карте воды.
"""
import warnings
import numpy as np
import mdtraj as md
import panda
from panda.density_profile import get_numerical_density_profile
from panda.utils import get_center_pbc, apply_pbc

warnings.filterwarnings("ignore")
D = "/mnt/c/Phystech/Computing_Physics_Center/Projects/PANDA_Reborn/graphite_decane_tip4p_1.0"
OUT = "/home/igaranov/projects/PANDA_Dev/smoke/out"
Lx, Ly, Lz = 17.27104, 5.12817, 11.0
Z_TOP, Z_BOT = 2.926, 0.195            # верх графита и низ графита (его образ сверху — Lz + Z_BOT)
H = Lz + Z_BOT - Z_TOP                 # щель по центрам атомов углерода
l = Lx / H
EQ_FRAMES, BLOCK = 1000, 500           # 2 нс на уравновешивание, блоки по 1 нс
print(f"H = {H:.3f} nm, l = Lx/H = {l:.3f}")


def water_profiles():
    """Тело panda.get_density_profile, но для кислородов воды."""
    top = md.load_topology(f"{D}/graphite_decane_tip4p.gro")
    idx = top.select("resname WATER and name OW")
    ax, de = [], []
    for chunk in md.iterload(f"{D}/graphite_decane_tip4p.xtc", top=top, chunk=100, skip=1):
        pos = chunk.xyz[:, idx, :]
        box = chunk.unitcell_lengths[:, np.newaxis, :]
        pos = apply_pbc(pos - get_center_pbc(pos, box) + box / 2, box)
        for i in range(chunk.n_frames):
            a, d = get_numerical_density_profile(pos[i], box[i], 200, center=True)
            ax.append(a), de.append(d)
    return np.array(ax), np.array(de)


def bulk_density(grid, x_lo, x_hi):
    """ρ_bulk: медиана 2D-карты в сердцевине домена фазы (по x) и в середине щели (по z)."""
    xs = (np.arange(grid.shape[0]) + 0.5) * Lx / grid.shape[0]
    zs = (np.arange(grid.shape[1]) + 0.5) * Lz / grid.shape[1]
    mx = (xs > x_lo) & (xs < x_hi)
    mz = np.abs(zs - (Z_TOP + H / 2)) < 1.5
    return float(np.median(grid[np.ix_(mx, mz)]))


def geometric_theta(grid, rho_bulk, grid_other):
    """Фит окружностей к изолинии ρ = ρ_bulk/2 воды (левая и правая граница домена), угол на стенке."""
    nx, nz = grid.shape
    xs = (np.arange(nx) + 0.5) * Lx / nx
    zs = (np.arange(nz) + 0.5) * Lz / nz
    # домен воды может переходить через x = 0 (PBC): переносим центр домена декана на край ячейки
    occ = grid_other.sum(axis=1)
    ang = np.angle(np.sum(occ * np.exp(2j * np.pi * xs / Lx))) % (2 * np.pi)
    shift = int(round(ang / (2 * np.pi) * nx)) % nx
    g = np.roll(grid, -shift, axis=0) / rho_bulk
    xs_s = (np.arange(nx) + 0.5) * Lx / nx
    zmask = (zs > Z_TOP + 1.0) & (zs < Lz + Z_BOT - 1.0)   # без слоистости у стенок
    left, right = [], []
    for j in np.where(zmask)[0]:
        col = g[:, j]
        wet = np.where(col > 0.5)[0]
        if len(wet) < 2:
            continue
        i0, i1 = wet[0], wet[-1]
        # линейная интерполяция точки пересечения 0.5
        xl = np.interp(0.5, [col[i0 - 1], col[i0]], [xs_s[i0 - 1], xs_s[i0]])
        xr = np.interp(0.5, [col[i1 + 1], col[i1]], [xs_s[i1 + 1], xs_s[i1]])
        left.append((xl, zs[j])), right.append((xr, zs[j]))

    def circle(pts):
        pts = np.asarray(pts)
        A = np.column_stack([2 * pts[:, 0], 2 * pts[:, 1], np.ones(len(pts))])
        b = (pts ** 2).sum(1)
        xc, zc, c = np.linalg.lstsq(A, b, rcond=None)[0]
        return xc, zc, np.sqrt(c + xc ** 2 + zc ** 2)

    out = []
    for pts in (left, right):
        xc, zc, r = circle(pts)
        # вода выпуклая, центр окружности внутри воды над нижней стенкой: cos θ = (z_w − z_c)/r
        cos_t = np.clip((Z_TOP - zc) / r, -1, 1)
        out.append((np.degrees(np.arccos(cos_t)), xc, zc, r))
    return out


def fit_blocks(name, axises, denses, rho_bulk):
    ax_b, de_b = panda.block_average_density_profile(axises[EQ_FRAMES:], denses[EQ_FRAMES:], BLOCK)
    thetas = []
    print(f"\n{name}: rho_bulk = {rho_bulk:.3f} nm^-3, blocks = {len(de_b)} x {BLOCK} frames")
    for i in range(len(de_b)):
        z, d = ax_b[i], de_b[i]
        inside = np.abs(z) < H / 2
        phi = np.trapezoid(d[inside] / rho_bulk, z[inside] / H)
        _, _, r = panda.profile_approx_from_array(d, z, rho_bulk, l, phi, H, interface_type="roll",
                                                  display=False, samples=5)
        th = np.rad2deg(r["theta"])
        thetas.append(th)
        at_bound = "  <- нижняя граница фита" if th < 90.6 else ""
        print(f"  block {i}: phi = {phi:.3f}, theta = {th:6.2f} deg, delta = {r['delta']:.4f}, "
              f"mae = {r['mae']:.4f}{at_bound}")
    thetas = np.array(thetas)
    print(f"  => theta = {thetas.mean():.2f} +/- {thetas.std(ddof=1):.2f} deg (mean +/- std по блокам)")
    return thetas


map_w = np.load(f"{OUT}/graphite_map_WATER.npy")
map_d = np.load(f"{OUT}/graphite_map_DECAN.npy")
rho_w = bulk_density(map_w, 11.0, 16.0)     # сердцевина домена воды
rho_d = bulk_density(map_d, 2.0, 7.0)       # сердцевина домена декана

ax_d = np.load(f"{OUT}/graphite_decan_axises.npy")
de_d = np.load(f"{OUT}/graphite_decan_denses.npy")
th_d = fit_blocks("(1) DECAN, библиотека как есть", ax_d, de_d, rho_d)

ax_w, de_w = water_profiles()
np.save(f"{OUT}/graphite_water_axises.npy", ax_w)
np.save(f"{OUT}/graphite_water_denses.npy", de_w)
th_w = fit_blocks("(2) WATER (OW), функции библиотеки", ax_w, de_w, rho_w)

for side, (th, xc, zc, r) in zip(("левая", "правая"), geometric_theta(map_w, rho_w, map_d)):
    print(f"\nГеом. угол через воду, {side} граница: {th:.1f} deg (окружность xc={xc:.2f}, zc={zc:.2f}, r={r:.2f} нм)")
print(f"PANDA через воду: {th_w.mean():.2f} deg  =>  через декан 180 - theta = {180 - th_w.mean():.2f} deg")
