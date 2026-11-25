"""The result of a simulation as a table of NumPy arrays."""
import os

os.environ.setdefault("WOBBLER_COMPILE", "0")

import numpy as np

from wobbler import Orbit, simulate_fit
from wobbler.results import to_table

from _scan import synthetic_scan

parallax = np.array([1.0, 2.0])
fit = simulate_fit(synthetic_scan(), Orbit(period_days=300.0, ecc=0.2, t_peri_days=0.0, omega=1.0), a1_au=0.8, parallax_mas=parallax, G=12.0, n_orient=50, seed=2, nu_eff=1.5, device="cpu")
table = to_table(fit, parallax)
print({name: column.shape for name, column in table.items()})
print("share of the 100 cases with RUWE over 1.4:", float((table["ruwe"] > 1.4).mean()))
assert len(table["ruwe"]) == 100
