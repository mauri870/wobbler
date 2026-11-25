"""The same orbit with a dark companion and with a companion that carries part of the light: the photocentre, which is what Gaia measures, moves less, and the RUWE is lower."""
import os

os.environ.setdefault("WOBBLER_COMPILE", "0")

import numpy as np

from wobbler import Orbit, simulate_fit

from _scan import synthetic_scan

scan = synthetic_scan()
orbit = Orbit(period_days=185.59, ecc=0.45, t_peri_days=-1.1, omega=0.22)
common = dict(a1_au=1.0, parallax_mas=np.array([2.0]), G=13.8, n_orient=300, seed=1, nu_eff=1.481, device="cpu")
dark = simulate_fit(scan, orbit, **common)
# flux_ratio: the companion's flux over the primary's in G (a number, or a function of the companion's mass in solar masses); primary_mass: in solar masses, which turns a1 sin i into the companion's mass at each inclination
luminous = simulate_fit(scan, orbit, flux_ratio=0.5, primary_mass=1.0, **common)
for name, fit in (("dark", dark), ("flux ratio 0.5", luminous)):
    print(f"{name:15s} median RUWE {np.median(fit.ruwe.numpy()):.2f}, median photocentre semi-major axis a0 {np.median(fit.a0_mas.numpy()):.3f} mas")
assert np.median(luminous.ruwe.numpy()) < np.median(dark.ruwe.numpy())
