"""The RUWE that Gaia DR3 would report for a star on an orbit with a dark companion, for 300 orientations of the orbit."""
import os

os.environ.setdefault("WOBBLER_COMPILE", "0")  # run eagerly: no compile time on a first call

import numpy as np

from wobbler import Orbit, simulate_fit

from _scan import synthetic_scan

scan = synthetic_scan()
orbit = Orbit(period_days=185.59, ecc=0.45, t_peri_days=-1.1, omega=0.22)  # t_peri_days: days from JD 2457389.0
fit = simulate_fit(scan, orbit, a1_au=1.0, parallax_mas=np.array([2.0]), G=13.8, n_orient=300, seed=1, nu_eff=1.481, device="cpu")
ruwe = fit.ruwe.numpy().ravel()
print(f"RUWE of {ruwe.size} orientations: median {np.median(ruwe):.2f}, 16 to 84 percent {np.percentile(ruwe, 16):.2f} to {np.percentile(ruwe, 84):.2f}")
assert np.isfinite(ruwe).all() and ruwe.min() > 0
