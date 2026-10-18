"""Phase 0: smoke check of the installed previous PANDA on a synthetic profile (roll, alpha)."""
import numpy as np
import panda
from panda import interface
H, l, phi, rho_bulk = 6.0, 11.6235 / 6.0, 0.5, 30.896
th_true, d_true = np.deg2rad(120.0), 0.05
print("phi range:", interface.phi_min_roll_alpha(l, th_true, d_true),
      interface.phi_max_roll_alpha(l, th_true, d_true))
z_norm = np.linspace(-0.5, 0.5, 200)
rho_norm = interface.rho_roll_alpha(z_norm, l, phi, th_true, d_true)
for sigma in (0.0, 0.02):
    rng = np.random.default_rng(0)
    dens = rho_bulk * (rho_norm + rng.normal(0, sigma, z_norm.size))
    _, _, res = panda.profile_approx_from_array(
        dens, z_norm * H, rho_bulk, l, phi, H,
        interface_type="roll", display=False, samples=3)
    dth = abs(np.rad2deg(res["theta"]) - 120.0)
    print(f"sigma={sigma}: theta={np.rad2deg(res['theta']):.3f} deg "
          f"delta={res['delta']:.4f} |dtheta|={dth:.3f} rmse={res['rmse']:.4g}")
