"""Gaia BH3: why the RUWE of the noise draws has two groups, and the RUWE under the weighting of DR3's fit.

usage: GAIAMOCK_DATA_DIR=... python validation/bh3_groups.py   (writes validation/bh3_groups.json; seconds on one CPU process)

The draws are those of validation/bh3.py (the combined orbit, 4,000 noise draws, seed 47): the model rejects a random 10 percent of the epochs of each draw, and the RUWE distribution of the default form has a small group below the main one. This script rebuilds the same draws, finds which rejected epochs separate the groups and measures their shares. It then fits each draw with the weighting of Gaia's astrometric solution (validation/bh3_proper_motion.py, candidate 3) and applies the default form to the unit weight error of that fit, with the same draws.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import bh3  # noqa: E402
import bh3_proper_motion as pm  # noqa: E402
import wobbler as wb  # noqa: E402
from wobbler import gaiafit  # noqa: E402


def percentiles(x):
    return [float(v) for v in np.percentile(x, [16, 50, 84])]


def main():
    c = {k: float(v) for k, v in pm.cat.items() if k not in ("read_on", "source_id")}
    orbit = bh3.ORBITS["combined"]
    scan, count, n = pm.scan, bh3.DRAWS, len(pm.JD)
    t = scan.t_yr(bh3.REFERENCE_JD)
    # the draws of bh3.predict, in the same order of the generator
    rng = np.random.default_rng(bh3.SEED)
    z_err, z_extra = rng.normal(size=(count, n)), rng.normal(size=(count, n))
    keep = rng.uniform(size=(count, n)) > gaiafit.REJECTED_SHARE
    extra = rng.uniform(0, gaiafit.BRIGHT_EXTRA_NOISE, count)
    o = orbit
    observed = wb.observe(scan, o["parallax"][0], o["pmra"][0], o["pmdec"][0], o["a0"], np.radians(o["inclination"]), np.radians(o["node"]), o["period"], o["ecc"], o["t_peri_jd"] - bh3.REFERENCE_JD, np.radians(o["omega"]),
                          pm.SIGMA, z_err, extra, z_extra, device="cpu")
    fit = wb.fit_five_parameter(t, scan.psi, scan.plx_factor, observed, pm.SIGMA, keep=keep, errors=False, device="cpu")
    ok = fit["ok"].numpy()
    raw = fit["uwe"].numpy()
    u0 = float(wb.u0(c["phot_g_mean_mag"], c["nu_eff_used_in_astrometry"], device="cpu"))
    ruwe = wb.calibrated_uwe(c["phot_g_mean_mag"], raw) / u0
    reference = bh3.predict(scan, c, orbit, "orbit")["ruwe"]
    assert np.allclose(np.sort(ruwe[ok]), np.sort(reference)), "the rebuilt draws are not those of validation/bh3.py"
    out = {"draws": int(count), "median_ruwe": float(np.median(ruwe[ok])), "share_of_draws_below_the_catalogued_ruwe": float(np.mean(ruwe[ok] < c["ruwe"]))}
    # the groups: the gap of the histogram of the RUWE
    counts, edges = np.histogram(ruwe[ok], bins=np.linspace(2.5, 5.0, 51))
    low = ruwe < 3.8
    out["group_boundary_ruwe"] = 3.8
    out["share_in_the_lower_group"] = float(np.mean(low[ok]))
    out["median_lower_group"], out["median_upper_group"] = float(np.median(ruwe[low & ok])), float(np.median(ruwe[~low & ok]))
    # which epochs: the share of draws that reject each epoch, in each group
    reject_low, reject_high = (1 - keep[low & ok]).mean(axis=0), (1 - keep[~low & ok]).mean(axis=0)
    years = 2000.0 + (pm.JD - 2451544.5) / 365.25
    order = np.argsort(-(reject_low - reject_high))
    out["epochs_most_often_rejected_in_the_lower_group"] = [{"epoch_index": int(i), "year": float(years[i]), "rejected_in_the_lower_group": float(reject_low[i]), "rejected_in_the_upper_group": float(reject_high[i])} for i in order[:4]]
    top = [int(i) for i in order[:2]]
    both_rejected = (keep[:, top] == 0).all(axis=1)
    one_rejected = (keep[:, top] == 0).sum(axis=1) == 1
    none_rejected = (keep[:, top] == 1).all(axis=1)
    out["the_two_epochs"] = top
    out["draws_with_both_rejected"], out["draws_with_one_rejected"], out["draws_with_none_rejected"] = float(both_rejected.mean()), float(one_rejected.mean()), float(none_rejected.mean())
    for label, mask in (("both rejected", both_rejected), ("one rejected", one_rejected), ("neither rejected", none_rejected)):
        out[f"ruwe_{label.replace(' ', '_')}_16_50_84"] = percentiles(ruwe[mask & ok])
    # the residuals of the epochs in the noise-free fit: which epochs sit far from the single-star fit
    clean = wb.observe(scan, o["parallax"][0], o["pmra"][0], o["pmdec"][0], o["a0"], np.radians(o["inclination"]), np.radians(o["node"]), o["period"], o["ecc"], o["t_peri_jd"] - bh3.REFERENCE_JD, np.radians(o["omega"]),
                       pm.SIGMA, np.zeros((1, n)), np.zeros(1), np.zeros((1, n)), device="cpu").numpy()
    mu = wb.fit_five_parameter(t, scan.psi, scan.plx_factor, clean, pm.SIGMA, keep=np.ones((1, n), bool), errors=False, device="cpu")["parameters"].numpy()
    residual = (clean - mu @ pm.design_of(t.numpy(), pm.PSI, pm.PLX).T)[0]
    out["noise_free_residuals_of_the_two_epochs_mas"] = [float(residual[i]) for i in top]
    out["largest_noise_free_residuals_mas"] = [{"epoch_index": int(i), "year": float(years[i]), "residual_mas": float(residual[i])} for i in np.argsort(-np.abs(residual))[:4]]
    # the RUWE under the weighting of DR3's fit, same draws
    for variant in ("fixed", "solved"):
        mu_w, eps, w, _ = pm.fit_agis(observed.numpy(), keep.astype(float), variant)
        residual_w = observed.numpy() - mu_w @ pm.design_of(t.numpy(), pm.PSI, pm.PLX).T
        residual_u = observed.numpy() - fit["parameters"].numpy() @ pm.design_of(t.numpy(), pm.PSI, pm.PLX).T
        ratio = np.sqrt(np.sum(np.where(keep, residual_w, 0.0) ** 2, axis=1) / np.sum(np.where(keep, residual_u, 0.0) ** 2, axis=1))
        weighted = wb.calibrated_uwe(c["phot_g_mean_mag"], raw * ratio) / u0
        out[f"ruwe_under_the_weighting_eps_{variant}"] = {"ruwe_16_50_84": percentiles(weighted[ok]), "ratio_of_the_root_sum_of_squared_residuals_16_50_84": percentiles(ratio[ok]), "ln_median_over_catalogue": float(np.log(np.median(weighted[ok]) / c["ruwe"])),
                                                         "share_of_draws_in_the_lower_group_before": float(np.mean(low[ok]))}
        out[f"ruwe_under_the_weighting_eps_{variant}"]["share_below_the_catalogued_ruwe"] = float(np.mean(weighted[ok] < c["ruwe"]))
    (HERE / "bh3_groups.json").write_text(json.dumps(out, indent=1) + "\n")
    print("median", round(out["median_ruwe"], 3), "share of draws below 3.41", round(out["share_of_draws_below_the_catalogued_ruwe"], 3), "lower group share", round(out["share_in_the_lower_group"], 3), "medians", round(out["median_lower_group"], 2), round(out["median_upper_group"], 2))
    print("epochs", out["epochs_most_often_rejected_in_the_lower_group"][:3])
    print("both/one/none rejected", round(out["draws_with_both_rejected"], 3), round(out["draws_with_one_rejected"], 3), round(out["draws_with_none_rejected"], 3), {k: [round(x, 2) for x in out[f"ruwe_{k}_16_50_84"]] for k in ("both_rejected", "one_rejected", "neither_rejected")})
    print("residuals of the two epochs", [round(x, 2) for x in out["noise_free_residuals_of_the_two_epochs_mas"]], "largest", [(d["epoch_index"], round(d["residual_mas"], 2)) for d in out["largest_noise_free_residuals_mas"]])
    for variant in ("fixed", "solved"):
        r = out[f"ruwe_under_the_weighting_eps_{variant}"]
        print("weighting", variant, [round(x, 2) for x in r["ruwe_16_50_84"]], "ln", round(r["ln_median_over_catalogue"], 3), "ratio", [round(x, 3) for x in r["ratio_of_the_root_sum_of_squared_residuals_16_50_84"]], "share below 3.41", round(r["share_below_the_catalogued_ruwe"], 3))


if __name__ == "__main__":
    main()
