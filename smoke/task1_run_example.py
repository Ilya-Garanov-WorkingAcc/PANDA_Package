"""Task 1, item 2: the example chain of the previous PANDA on real data.
(a) The example system (calcite–decane–TIP4P). Only 1 frame is available, so
    get_density_profile (it counts frames from time/timestep) cannot be called;
    its body is repeated with the same library functions and the parameters of the example notebook.
(b) Benzene on a substrate, 5001 frames: profiles from npys/ (computed by the user's notebook
    in the fixed box frame) -> block_average_density_profile ->
    profile_approx_from_array(roll, alpha). Geometry as in theta_contact_panda.ipynb.
"""
import warnings
import numpy as np
import mdtraj as md
import panda
from panda.density_profile import get_numerical_density_profile
from panda.utils import get_center_pbc, apply_pbc
warnings.filterwarnings("ignore")
R = "/mnt/c/Phystech/Computing_Physics_Center/Projects/roughness_2/angles"
print("=== (a) calcite–decane, the repository example, 1 frame")
S = f"{R}/Dev_PANDA/Dev_PANDA/systems/calcite_decane_tip4p_sc_2.5_offset"
t = md.load(f"{S}/cal_dec_tip4p.xtc", top=f"{S}/cal_dec_tip4p.gro")
pos = t.xyz[:, t.top.select("resname DECAN"), :]
box = t.unitcell_lengths[:, np.newaxis, :]
pos = apply_pbc(pos - get_center_pbc(pos, box) + box / 2, box)
axis, dens = get_numerical_density_profile(pos[0], box[0], 200, center=True)
H, l, phi, rho_bulk = 6.0, 11.6235 / 6.0, 0.5, 3.0896 * 10  # parameters of the example notebook
_, _, res = panda.profile_approx_from_array(dens, axis, rho_bulk, l, phi, H,
                                            interface_type="roll", display=False, samples=5)
print(f"theta = {np.rad2deg(res['theta']):.2f} deg, delta = {res['delta']:.4f}, mae = {res['mae']:.4f}")
print("\n=== (b) BENZ on the SOLD substrate, 5001 frames, profiles from npys/")
axises = np.load(f"{R}/npys/axises_benz.npy")
denses = np.load(f"{R}/npys/denses_benz.npy")
Lx, Lz, z_wall = 47.8981, 13.4607, 3.147                 # nm, from nvt_prod.gro: box and top of SOLD
H = Lz - z_wall
l = Lx / H
eq = int(0.4 * len(denses))                          # as in the notebook: 40% for equilibration
ax_b, de_b = panda.block_average_density_profile(axises[eq:], denses[eq:], block_size=600)
print(f"H = {H:.3f} nm, l = {l:.3f}, blocks = {len(de_b)}")
for i in range(len(de_b)):
    z_box = ax_b[i] + Lz / 2                          # npys axes: bin centres - Lz/2
    # PANDA expects the axis over the whole box (substrate included), otherwise the fit window is empty.
    z = (z_box - (z_wall + H / 2) + Lz / 2) % Lz - Lz / 2
    order = np.argsort(z)
    z, d = z[order], de_b[i][order]
    rho_b = np.median(d[d > 0.8 * d.max()])           # ρ_bulk estimated from the plateau (heuristic)
    inside = np.abs(z) < H / 2
    phi_i = np.trapezoid(d[inside] / rho_b, z[inside] / H)
    _, _, r = panda.profile_approx_from_array(d, z, rho_b, l, phi_i, H,
                                              interface_type="roll", display=False, samples=5)
    print(f"block {i}: rho_bulk = {rho_b:.3f} nm^-3, phi = {phi_i:.3f}, "
          f"theta = {np.rad2deg(r['theta']):.2f} deg, delta = {r['delta']:.4f}, mae = {r['mae']:.4f}")
