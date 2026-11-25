"""Records gaiamock's deterministic orbit and noise-model numbers for constant inputs into gaiamock_golden.json (needs gaiamock 1.0.0).
usage: python tests/golden/record_gaiamock.py

The orbit lines are those of gaiamock.predict_astrometry_binary_in_terms_of_a0 (gaiamock/gaiamock.py, the block from `A_pred = ...` to `delta_eta = ...`, copied below without the random draws); the epoch error is gaiamock.al_uncertainty_per_ccd_interp over sqrt(8) (the `epoch_err_per_transit` line of the same function); the eccentric anomaly is gaiamock.solve_kepler_eqn_on_array."""
import json
from pathlib import Path

import gaiamock
import numpy as np

c_funcs = gaiamock.read_in_C_functions()
T_YR = [-1.2, -0.7, -0.31, 0.0, 0.18, 0.5, 0.93, 1.4, 1.77, 2.2]
PSI = [0.1, 0.9, 1.7, 2.5, 3.3, 4.1, 4.9, 5.7, 0.4, 2.0]
# period (d), t_peri (d from JD 2457389), ecc, node ("omega" of gaiamock), inclination, w ("little omega"), a0 (mas)
ORBITS = [
    (185.59, -1.1, 0.451, 1.707, 2.210, 0.223, 2.67),
    (880.0, 140.0, 0.12, 0.35, 0.60, 4.90, 1.8),
    (60.5, -20.0, 0.75, 5.20, 1.20, 2.40, 0.9),
    (400.0, 0.0, 0.0, 2.90, 1.57, 1.00, 3.3),
    (1217.19, 300.0, 0.08, 4.40, 2.60, 6.00, 2.2),
    (310.0, -85.0, 0.33, 0.0, 0.0, 0.0, 1.5),
]
G = [12.0, 15.5, 9.3]

cases = []
for period, tp, ecc, node, inc, w, a0 in ORBITS:
    t = np.array(T_YR)
    psi = np.array(PSI)
    EE = gaiamock.solve_kepler_eqn_on_array(M=2 * np.pi / period * (t * 365.25 - tp), ecc=ecc, c_funcs=c_funcs)
    omega = node
    A_pred = a0 * (np.cos(w) * np.cos(omega) - np.sin(w) * np.sin(omega) * np.cos(inc))
    B_pred = a0 * (np.cos(w) * np.sin(omega) + np.sin(w) * np.cos(omega) * np.cos(inc))
    F_pred = -a0 * (np.sin(w) * np.cos(omega) + np.cos(w) * np.sin(omega) * np.cos(inc))
    G_pred = -a0 * (np.sin(w) * np.sin(omega) - np.cos(w) * np.cos(omega) * np.cos(inc))
    X = np.cos(EE) - ecc
    Y = np.sqrt(1 - ecc**2) * np.sin(EE)
    x, y = B_pred * X + G_pred * Y, A_pred * X + F_pred * Y
    delta_eta = y * np.cos(psi) + x * np.sin(psi)
    cases.append(dict(period=period, t_peri=tp, ecc=ecc, node=node, inclination=inc, omega=w, a0=a0, thiele_innes=[A_pred, B_pred, F_pred, G_pred], sin_E=np.sin(EE).tolist(), cos_E=np.cos(EE).tolist(), delta_eta=delta_eta.tolist()))
out = dict(gaiamock_version="1.0.0", t_yr=T_YR, psi=PSI, cases=cases, G=G, epoch_error=[float(gaiamock.al_uncertainty_per_ccd_interp(g) / np.sqrt(8)) for g in G])
Path(__file__).with_name("gaiamock_golden.json").write_text(json.dumps(out, indent=1))
