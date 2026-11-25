"""Agreement of wobbler with gaiamock on identical inputs.

usage: python validation/agreement.py   (needs gaiamock 1.0.0; writes validation/agreement.json)

The abscissae are made by wobbler.observe from random orbits and noise. gaiamock's own functions then fit the same abscissae one case at a time, and the batched fit of wobbler is compared with them case by case. gaiamock: https://github.com/kareemelbadry/gaiamock (El-Badry et al. 2024, MNRAS 531, 2179).
"""
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("WOBBLER_COMPILE", "0")

import gaiamock
import numpy as np
import torch

import wobbler as wb

HERE = Path(__file__).parent
CASES, EPOCHS, SEED = 2000, 190, 2
GOLDEN = json.loads((HERE.parent / "tests" / "golden" / "gaiamock_golden.json").read_text())


def fit_agreement():
    rng = np.random.default_rng(SEED)
    jd = np.sort(rng.uniform(*wb.WINDOWS["DR3"], EPOCHS))
    scan = wb.Scan(jd=jd, psi=rng.uniform(0, np.pi, EPOCHS), plx_factor=rng.uniform(-0.9, 0.9, EPOCHS))
    n = CASES
    parameters = dict(parallax=rng.uniform(0.5, 4, n), pmra=rng.normal(0, 5, n), pmdec=rng.normal(0, 5, n), a0_mas=rng.uniform(0, 4, n), inclination=np.arccos(rng.uniform(0, 1, n)), node=rng.uniform(0, 2 * np.pi, n),
                      period_days=rng.uniform(100, 900, n), ecc=rng.uniform(0, 0.8, n), t_peri_days=rng.uniform(-400, 400, n), omega=rng.uniform(0, 2 * np.pi, n))
    z_err, z_extra, extra, keep = rng.normal(size=(n, EPOCHS)), rng.normal(size=(n, EPOCHS)), rng.uniform(0, 0.04, n), rng.uniform(size=(n, EPOCHS)) > 0.1
    error = gaiamock.al_uncertainty_per_ccd_interp(13.0) / np.sqrt(8)
    observed = wb.observe(scan, error=error, z_err=z_err, extra=extra, z_extra=z_extra, device="cpu", **parameters)
    t = scan.t_yr()
    batched = {k: v.numpy() if hasattr(v, "numpy") else v for k, v in wb.fit_five_parameter(t, scan.psi, scan.plx_factor, observed, error, keep=keep, device="cpu").items()}
    observed, t, psi, plx = observed.numpy(), t.numpy(), scan.psi.numpy(), scan.plx_factor.numpy()
    started = time.perf_counter()
    reference = [gaiamock.fit_5par_solution_only(t_ast_yr=t[keep[i]], psi=psi[keep[i]], plx_factor=plx[keep[i]], ast_obs=observed[i, keep[i]], ast_err=error * np.ones(keep[i].sum())) for i in range(n)]
    gaiamock_seconds = time.perf_counter() - started
    reference_ok = np.array([r[0] != 0 or r[4] != 0 for r in reference])
    ok = batched["ok"]
    both = ok & reference_ok
    parameters_ref = np.array([r[:5] for r in reference])
    scale = np.maximum(np.abs(parameters_ref), 1e-12)
    return {
        "cases": n, "epochs": EPOCHS, "seed": SEED, "G": 13.0,
        "fits_attempted_by_gaiamock": int(reference_ok.sum()), "fits_attempted_by_wobbler": int(ok.sum()), "cases_where_the_two_disagree_on_whether_to_fit": int((ok != reference_ok).sum()),
        "max_abs_difference_parameters": float(np.max(np.abs(batched["parameters"][both] - parameters_ref[both]))),
        "max_relative_difference_parameters": float(np.max(np.abs(batched["parameters"][both] - parameters_ref[both]) / scale[both])),
        "max_relative_difference_ruwe_of_gaiamock": float(np.max(np.abs(batched["ruwe"][both] / np.array([r[10] for r in reference])[both] - 1))),
        "max_relative_difference_parallax_error": float(np.max(np.abs(batched["parameter_errors"][both, 4] / np.array([r[9] for r in reference])[both] - 1))),
        "gaiamock_seconds_one_case_at_a_time": gaiamock_seconds,
    }


def kepler_agreement():
    c_funcs = gaiamock.read_in_C_functions()
    rng = np.random.default_rng(1)
    worst = {}
    for ecc in (0.0, 0.21, 0.6, 0.9):
        mean = rng.uniform(-10, 10, 50)
        ours, theirs = wb.kepler(mean, ecc, device="cpu").numpy(), gaiamock.solve_kepler_eqn_on_array(mean, ecc, c_funcs)
        worst[str(ecc)] = float(max(np.max(np.abs(np.sin(ours) - np.sin(theirs))), np.max(np.abs(np.cos(ours) - np.cos(theirs)))))
    return {"max_difference_of_sin_and_cos_of_the_eccentric_anomaly_by_eccentricity": worst, "points_per_eccentricity": 50}


def orbit_agreement():
    t_yr, psi = np.array(GOLDEN["t_yr"]), np.array(GOLDEN["psi"])
    thiele, along = [], []
    for case in GOLDEN["cases"]:
        constants = wb.thiele_innes(case["a0"], case["node"], case["omega"], case["inclination"])
        thiele.append(float(np.max(np.abs(np.asarray(constants, dtype=float) - np.array(case["thiele_innes"])))))
        x, y = wb.photocentre(t_yr, case["period"], case["ecc"], case["t_peri"], case["a0"], case["inclination"], case["node"], case["omega"])
        along.append(float(np.max(np.abs(np.asarray(y) * np.cos(psi) + np.asarray(x) * np.sin(psi) - np.array(case["delta_eta"])))))
    error = [float(abs(float(wb.epoch_error(g)) / e - 1)) for g, e in zip(GOLDEN["G"], GOLDEN["epoch_error"])]
    return {"orbits": len(GOLDEN["cases"]), "epochs": len(t_yr), "max_abs_difference_thiele_innes_constants_mas": max(thiele), "max_abs_difference_along_scan_offset_mas": max(along),
            "max_relative_difference_epoch_error": max(error), "gaiamock_version_of_the_record": GOLDEN["gaiamock_version"]}


if __name__ == "__main__":
    out = {"fit": fit_agreement(), "kepler": kepler_agreement(), "orbit_against_recorded_gaiamock": orbit_agreement(),
           "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "gaiamock": "1.0.0"}}
    (HERE / "agreement.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
