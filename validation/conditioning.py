"""The conditioning of wobbler's five-parameter solve on real Gaia scans, against an SVD least-squares fit: the largest difference in units of the formal errors and the condition numbers of each scan.

usage: GAIAMOCK_DATA_DIR=... python validation/conditioning.py   (writes validation/conditioning.json; needs gaiamock and its scanning-law data for Scan.from_position)
"""
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import wobbler as wb

HERE = Path(__file__).resolve().parent
N_POSITIONS, N_SETS, SEED, ERROR = 12288, 20, 1733, 0.3
rng = np.random.default_rng(SEED)
ra = rng.uniform(0, 360, N_POSITIONS)
dec = np.degrees(np.arcsin(rng.uniform(-1, 1, N_POSITIONS)))
commit = subprocess.run(["git", "-C", str(Path(wb.__file__).resolve().parents[2]), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
result = {"wobbler_commit": commit, "positions": N_POSITIONS, "sets_per_scan": N_SETS, "seed": SEED, "epoch_error_mas": ERROR, "windows": {}}
start = time.time()
for window in ("DR3", "DR2"):
    worst, conds, normals, skipped, worst_where = 0.0, [], [], 0, None
    for k in range(N_POSITIONS):
        scan = wb.Scan.from_position(float(ra[k]), float(dec[k]), window=window, device="cpu")
        t, psi, plx = scan.t_yr().numpy(), scan.psi.numpy(), scan.plx_factor.numpy()
        A = np.stack([np.sin(psi), t * np.sin(psi), np.cos(psi), t * np.cos(psi), plx], axis=1)
        truth = np.column_stack([rng.normal(0, 1, N_SETS), rng.normal(0, 10, N_SETS), rng.normal(0, 1, N_SETS), rng.normal(0, 10, N_SETS), np.full(N_SETS, 5.0)])
        observed = truth @ A.T + rng.normal(0, ERROR, (N_SETS, len(t)))
        fit = wb.fit_five_parameter(t, psi, plx, observed, ERROR, device="cpu")
        ok = fit["ok"].numpy()
        if not ok.all():
            skipped += 1
            continue
        mine = fit["parameters"].numpy()
        sigma = fit["parameter_errors"].numpy()
        reference = np.linalg.lstsq(A / ERROR, (observed / ERROR).T, rcond=None)[0].T
        difference = float(np.max(np.abs(mine - reference) / sigma))
        conds.append(float(np.linalg.cond(A)))
        normals.append(float(np.linalg.cond(A.T @ A)))
        if difference > worst:
            worst, worst_where = difference, {"ra": float(ra[k]), "dec": float(dec[k]), "epochs": len(t), "cond_A": conds[-1], "cond_normal": normals[-1]}
    conds, normals = np.array(conds), np.array(normals)
    result["windows"][window] = {"fitted_scans": int(len(conds)), "not_fitted_scans": skipped, "largest_difference_in_formal_errors": worst, "worst_scan": worst_where,
                                 "cond_A_median": float(np.median(conds)), "cond_A_max": float(conds.max()), "cond_normal_median": float(np.median(normals)), "cond_normal_max": float(normals.max())}
    print(window, json.dumps(result["windows"][window]), flush=True)
worst = max(w["largest_difference_in_formal_errors"] for w in result["windows"].values())
result["largest_difference_in_formal_errors"] = worst
result["seconds"] = round(time.time() - start, 1)
(HERE / "conditioning.json").write_text(json.dumps(result, indent=1))
print("largest difference in formal errors:", worst, "seconds:", result["seconds"])
