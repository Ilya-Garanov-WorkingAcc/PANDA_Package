"""Task 1, item 2: a complete working test of the previous PANDA on graphite–decane–TIP4P (PANDA_Reborn).

Morphology (smoke/task1_graphite_map.py): decane is a bridge (roll) between the graphite walls,
wider at the walls => decane wets graphite; water is a convex bridge (non-wetting phase).

(1) DECAN: the library as is, profiles from task1_graphite_profiles.py (panda.get_density_profile).
(2) WATER: the same library functions, only the residue is replaced (get_density_profile accepts DECAN only).
Reference θ: geometric circle fit to the ρ = ρ_bulk/2 isoline of the 2D water map.
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
Z_TOP, Z_BOT = 2.926, 0.195            # top and bottom of graphite (its image above is Lz + Z_BOT)
H = Lz + Z_BOT - Z_TOP                 # slit between the carbon atom centres
l = Lx / H
EQ_FRAMES, BLOCK = 1000, 500           # 2 ns for equilibration, 1 ns blocks
print(f"H = {H:.3f} nm, l = Lx/H = {l:.3f}")


def water_profiles():
    """The body of panda.get_density_profile, for water oxygens."""
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
    """ρ_bulk: median of the 2D map in the core of the phase domain (along x) and the middle of the slit (z)."""
    xs = (np.arange(grid.shape[0]) + 0.5) * Lx / grid.shape[0]
    zs = (np.arange(grid.shape[1]) + 0.5) * Lz / grid.shape[1]
    mx = (xs > x_lo) & (xs < x_hi)
    mz = np.abs(zs - (Z_TOP + H / 2)) < 1.5
    return float(np.median(grid[np.ix_(mx, mz)]))


def geometric_theta(grid, rho_bulk, grid_other):
    """Circle fits to the ρ = ρ_bulk/2 isoline of water (left and right domain edges), angle at the wall."""
    nx, nz = grid.shape
    xs = (np.arange(nx) + 0.5) * Lx / nx
    zs = (np.arange(nz) + 0.5) * Lz / nz
    # the water domain may cross x = 0 (PBC): move the decane domain centre to the cell edge
    occ = grid_other.sum(axis=1)
    ang = np.angle(np.sum(occ * np.exp(2j * np.pi * xs / Lx))) % (2 * np.pi)
    shift = int(round(ang / (2 * np.pi) * nx)) % nx
    g = np.roll(grid, -shift, axis=0) / rho_bulk
    xs_s = (np.arange(nx) + 0.5) * Lx / nx
    zmask = (zs > Z_TOP + 1.0) & (zs < Lz + Z_BOT - 1.0)   # without the layering at the walls
    left, right = [], []
    for j in np.where(zmask)[0]:
        col = g[:, j]
        wet = np.where(col > 0.5)[0]
        if len(wet) < 2:
            continue
        i0, i1 = wet[0], wet[-1]
        # linear interpolation of the 0.5 crossing
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
        # water is convex, the circle centre lies in the water above the lower wall: cos θ = (z_w − z_c)/r
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
        at_bound = "  <- lower fit bound" if th < 90.6 else ""
        print(f"  block {i}: phi = {phi:.3f}, theta = {th:6.2f} deg, delta = {r['delta']:.4f}, "
              f"mae = {r['mae']:.4f}{at_bound}")
    thetas = np.array(thetas)
    print(f"  => theta = {thetas.mean():.2f} +/- {thetas.std(ddof=1):.2f} deg (mean +/- std over blocks)")
    return thetas


map_w = np.load(f"{OUT}/graphite_map_WATER.npy")
map_d = np.load(f"{OUT}/graphite_map_DECAN.npy")
rho_w = bulk_density(map_w, 11.0, 16.0)     # core of the water domain
rho_d = bulk_density(map_d, 2.0, 7.0)       # core of the decane domain

ax_d = np.load(f"{OUT}/graphite_decan_axises.npy")
de_d = np.load(f"{OUT}/graphite_decan_denses.npy")
th_d = fit_blocks("(1) DECAN, the library as is", ax_d, de_d, rho_d)

ax_w, de_w = water_profiles()
np.save(f"{OUT}/graphite_water_axises.npy", ax_w)
np.save(f"{OUT}/graphite_water_denses.npy", de_w)
th_w = fit_blocks("(2) WATER (OW), library functions", ax_w, de_w, rho_w)

for side, (th, xc, zc, r) in zip(("left", "right"), geometric_theta(map_w, rho_w, map_d)):
    print(f"\nGeometric angle through water, {side} edge: {th:.1f} deg (circle xc={xc:.2f}, zc={zc:.2f}, r={r:.2f} nm)")
print(f"PANDA through water: {th_w.mean():.2f} deg  =>  through decane 180 - theta = {180 - th_w.mean():.2f} deg")
