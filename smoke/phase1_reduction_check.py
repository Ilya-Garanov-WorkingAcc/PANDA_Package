"""Phase 1: are the modules theta/delta/pi special cases of alpha?
theta == alpha(delta=0), delta == alpha(theta=pi), pi == alpha(theta=pi, delta=0)."""
import warnings
import numpy as np
import importlib
from panda.interface import alpha, theta, delta

pim = importlib.import_module("panda.interface.pi")  # interface.pi is shadowed by numpy.pi
warnings.filterwarnings("ignore")
SHAPES = ["droplet", "doughnut", "worm", "roll", "perforation", "layer"]
z = np.linspace(-0.5, 0.5, 401)
l = 1.94
TH = [np.deg2rad(a) for a in (100, 120, 150, 179.9)]
DE = [0.0, 0.05, 0.1]
def f(mod, name):
    return getattr(mod, name, None)
def maxdiff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    both = np.isfinite(a) & np.isfinite(b)
    nan_mismatch = int(np.sum(np.isfinite(a) != np.isfinite(b)))
    return (float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0), nan_mismatch
def admissible_phis(pmin, pmax):
    lo, hi = max(pmin, 0.0), min(pmax, 1.0)
    return [] if not hi > lo else list(np.linspace(lo, hi, 5)[1:-1])
for s in SHAPES:
    A = f(alpha, f"rho_{s}_alpha")
    Amin, Amax = f(alpha, f"phi_min_{s}_alpha"), f(alpha, f"phi_max_{s}_alpha")
    rows = []
    T, Tmin, Tmax = (f(theta, f"{k}_{s}_theta") for k in ("rho", "phi_min", "phi_max"))
    worst, nans, lim = 0.0, 0, 0.0
    for th in TH:
        lim = max(lim, abs(Tmin(l, th) - Amin(l, th, 0)), abs(Tmax(l, th) - Amax(l, th, 0)))
        for ph in admissible_phis(Tmin(l, th), Tmax(l, th)):
            d, n = maxdiff(T(z, l, ph, th), A(z, l, ph, th, 0.0)); worst, nans = max(worst, d), nans + n
    rows.append(("theta=alpha(d=0)", worst, nans, lim))
    D, Dmin, Dmax = (f(delta, f"{k}_{s}_delta") for k in ("rho", "phi_min", "phi_max"))
    worst, nans, lim = 0.0, 0, 0.0
    for de in DE:
        lim = max(lim, abs(Dmin(l, de) - Amin(l, np.pi, de)), abs(Dmax(l, de) - Amax(l, np.pi, de)))
        for ph in admissible_phis(Dmin(l, de), Dmax(l, de)):
            d, n = maxdiff(D(z, l, ph, de), A(z, l, ph, np.pi, de)); worst, nans = max(worst, d), nans + n
    rows.append(("delta=alpha(th=pi)", worst, nans, lim))
    P, Pmin, Pmax = (f(pim, f"{k}_{s}_pi") for k in ("rho", "phi_min", "phi_max"))
    lim = max(abs(Pmin(l) - Amin(l, np.pi, 0)), abs(Pmax(l) - Amax(l, np.pi, 0)))
    worst, nans = 0.0, 0
    for ph in admissible_phis(Pmin(l), Pmax(l)):
        d, n = maxdiff(P(z, l, ph), A(z, l, ph, np.pi, 0.0)); worst, nans = max(worst, d), nans + n
    rows.append(("pi=alpha(th=pi,d=0)", worst, nans, lim))
    for name, w, n, lm in rows:
        print(f"{s:12s} {name:22s} max|drho|={w:.2e}  nan-mismatch={n:4d}  max|dphi_lim|={lm:.2e}")
