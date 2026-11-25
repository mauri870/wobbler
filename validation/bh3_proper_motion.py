"""Gaia BH3: the proper-motion offsets of the DR3 fit against the published orbit, and four candidate causes of the difference in declination.

usage: GAIAMOCK_DATA_DIR=... python validation/bh3_proper_motion.py   (writes validation/bh3_proper_motion.json; needs gaiamock and its scanning-law data; seconds on one CPU process)

The prediction is that of validation/bh3.py: the combined orbit of Table 2 of Gaia Collaboration, Panuzzo et al. 2024 (A&A 686, L2) and the DR3 scan of the source. The offset of the DR3 fit is its fitted proper motion minus the proper motion of the orbit's barycentre. DR3 reports +6.08 mas/yr in right ascension and -0.055 in declination above the barycentric values; the prediction is +6.24 and -0.56.
Rule, written before the candidates were run: a candidate accounts for the declination offset when it moves the predicted offset by at least 0.25 mas/yr (half the difference) towards the catalogued one, or brings the prediction within 2 sigma of it, sigma being the quadrature sum of the catalogue's error and the uncertainty of the orbit's elements.
1. The uncertainty of the elements: each element of Table 2 moved by plus and minus its error in the noise-free fit; the quadrature sum is sigma_el (uncorrelated; the paper does not publish the correlations).
2. The epochs that the scan lacks: DR3 matched 47 transits and the scan has 42; five epochs, each a copy of one drawn from the 42, in 100 choices.
3. The weighting of DR3's fit: Lindegren et al. 2012 (A&A 538, A78, Eq. 27, 28, 62, 64 and 66): each epoch has the weight w / (sigma^2 + eps^2), eps^2 the solution of the weighted sum of squared residuals equal to the number of epochs with a weight above 0.2 minus five, and w = 1 for |z| up to 2, 1 - 1.773735 t^2 + 1.141615 t^3 (t = |z| - 2) up to 3 and exp(-|z|/3) beyond, z the residual over sqrt(sigma^2 + eps^2); four iterations. eps solved, or fixed at DR3's astrometric excess noise.
4. The frame of the bright stars: Cantat-Gaudin and Brandt 2021 (A&A 649, A124, Eq. 5 and Table 1): the DR3 proper motion minus [-sin d cos a, -sin d sin a, cos d; sin a, -cos a, 0] x (omega_X, omega_Y, omega_Z) of the bin of G, here G 11.00 to 11.50: 19.4, 59.9 and 0.2 micro-arcsec/yr.
"""
import json
import math
import os
import platform
import sys
from pathlib import Path

os.environ.setdefault("WOBBLER_COMPILE", "0")

import numpy as np
import torch

import wobbler as wb
from wobbler import gaiafit

sys.path.insert(0, str(Path(__file__).parent))
import bh3  # noqa: E402

HERE = Path(__file__).parent
REF = 2457389.0
cat = json.loads((HERE / "bh3_catalogue.json").read_text())
G, RA, DEC = float(cat["phot_g_mean_mag"]), float(cat["ra"]), float(cat["dec"])
OBS_RA, OBS_DEC, ERR_RA, ERR_DEC = float(cat["pmra"]), float(cat["pmdec"]), float(cat["pmra_error"]), float(cat["pmdec_error"])
AEN = float(cat["astrometric_excess_noise"])
ELEMENTS = dict(period=(4253.1, 98.5), ecc=(0.7291, 0.0048), a0=(27.39, 0.49), inclination=(110.580, 0.095), node=(136.236, 0.128), omega=(77.34, 0.76), t_peri_jd=(2458177.39, 0.88), parallax=(1.6933, 0.0164), pmra=(-28.317, 0.067), pmdec=(-155.221, 0.111))
BASE = {k: v[0] for k, v in ELEMENTS.items()}
scan = wb.Scan.from_position(RA, DEC, window="DR3", device="cpu")
JD, PSI, PLX = scan.jd.numpy(), scan.psi.numpy(), scan.plx_factor.numpy()
SIGMA = float(wb.epoch_error(G, device="cpu"))


def design_of(t, psi, plx):
    return np.stack([np.sin(psi), t * np.sin(psi), np.cos(psi), t * np.cos(psi), plx], axis=1)


def abscissae(e, jd, psi, plx, count, noisy, seed=47):
    s = wb.Scan(jd=jd, psi=psi, plx_factor=plx, device="cpu")
    n = len(jd)
    rng = np.random.default_rng(seed)
    zero = np.zeros((count, n))
    z_err, z_extra = (rng.normal(size=(count, n)), rng.normal(size=(count, n))) if noisy else (zero, zero)
    extra = rng.uniform(0, gaiafit.BRIGHT_EXTRA_NOISE, count) if noisy else np.zeros(count)
    keep = rng.uniform(size=(count, n)) > gaiafit.REJECTED_SHARE
    observed = wb.observe(s, e["parallax"], e["pmra"], e["pmdec"], e["a0"], np.radians(e["inclination"]), np.radians(e["node"]), e["period"], e["ecc"], np.asarray(e["t_peri_jd"]) - REF, np.radians(e["omega"]), SIGMA, z_err, extra, z_extra, device="cpu")
    return s, observed.numpy(), keep


def offsets(e, jd=JD, psi=PSI, plx=PLX):
    """The noise-free fit of the DR3 window: fitted proper motion minus the barycentric one."""
    s, observed, _ = abscissae(e, jd, psi, plx, int(np.size(e["a0"])), False)
    n, count = len(jd), observed.shape[0]
    fit = wb.fit_five_parameter(s.t_yr(REF), s.psi, s.plx_factor, observed, SIGMA, keep=np.ones((count, n), bool), errors=False, device="cpu")
    p = fit["parameters"].numpy()
    return p[:, 1] - np.asarray(e["pmra"]), p[:, 3] - np.asarray(e["pmdec"])


def robust_weight(z):
    a = np.abs(z)
    t = a - 2.0
    return np.where(a <= 2.0, 1.0, np.where(a < 3.0, 1.0 - 1.773735 * t**2 + 1.141615 * t**3, np.exp(-a / 3.0)))


def fit_agis(observed, keep, variant, iterations=4):
    s = wb.Scan(jd=JD, psi=PSI, plx_factor=PLX, device="cpu")
    t, design, n, count = s.t_yr(REF), design_of(s.t_yr(REF).numpy(), PSI, PLX), len(JD), observed.shape[0]
    w, eps2 = np.ones((count, n)) * keep, np.zeros(count)
    for _ in range(iterations + 1):
        error = np.sqrt((SIGMA**2 + eps2)[:, None] / np.maximum(w, 1e-12))
        mu = wb.fit_five_parameter(t, s.psi, s.plx_factor, observed, error, keep=w > 1e-9, errors=False, device="cpu")["parameters"].numpy()
        residual = observed - mu @ design.T
        nu = np.maximum((w > 0.2).sum(axis=1) - 5, 1)
        eps2 = np.maximum(0.0, (residual**2 * w).sum(axis=1) / nu - SIGMA**2) if variant == "solved" else np.full(count, AEN**2)
        w = robust_weight(residual / np.sqrt(SIGMA**2 + eps2)[:, None]) * keep
    return mu, np.sqrt(eps2), w, residual


def percentiles(x):
    return [float(v) for v in np.percentile(x, [16, 50, 84])]


def main():
    ra0, dec0 = (float(x[0]) for x in offsets(BASE))
    out = {"catalogue_offsets": [OBS_RA - BASE["pmra"], OBS_DEC - BASE["pmdec"]], "predicted_noise_free": [ra0, dec0], "catalogue_errors": [ERR_RA, ERR_DEC], "sensitivity": {}}
    out["convention_check_noise_free_offsets_ra_dec"] = {"published": [ra0, dec0]}
    for label, change in (("argument of periastron + 180 deg", {"omega": BASE["omega"] + 180}), ("node + 180 deg", {"node": BASE["node"] + 180}), ("inclination 180 deg - i", {"inclination": 180 - BASE["inclination"]})):
        out["convention_check_noise_free_offsets_ra_dec"][label] = [float(x[0]) for x in offsets({**BASE, **change})]
    sq = np.zeros(2)
    for name, (value, error) in ELEMENTS.items():
        shifts = []
        for sign in (-1, 1):
            e = dict(BASE)
            e[name] = value + sign * error
            shifts.append([float(x[0]) for x in offsets(e)])
        change = [(shifts[1][k] - shifts[0][k]) / 2 for k in (0, 1)]
        out["sensitivity"][name] = {"error": error, "change_in_the_offsets_per_error_ra_dec": change}
        sq += np.square(change)
    sigma_el = np.sqrt(sq)
    sigma_total = np.hypot([ERR_RA, ERR_DEC], sigma_el)
    out["sigma_elements"], out["sigma_total"] = sigma_el.tolist(), sigma_total.tolist()
    out["candidate_1_elements"] = {axis: {"difference": out["catalogue_offsets"][k] - [ra0, dec0][k], "in_sigma": (out["catalogue_offsets"][k] - [ra0, dec0][k]) / sigma_total[k], "within_2_sigma": bool(abs(out["catalogue_offsets"][k] - [ra0, dec0][k]) <= 2 * sigma_total[k])} for k, axis in ((0, "ra"), (1, "dec"))}
    rng = np.random.default_rng(1695)
    draw = {k: rng.normal(v[0], v[1], 2000) for k, v in ELEMENTS.items()}
    r, d = offsets(draw)
    out["element_draws"] = {"ra_16_50_84": percentiles(r), "dec_16_50_84": percentiles(d), "share_dec_at_or_above_the_catalogue": float(np.mean(d >= out["catalogue_offsets"][1]))}
    moves = []
    for s in range(100):
        pick = np.random.default_rng(1695 + s).integers(0, len(JD), 5)
        order = np.argsort(np.concatenate([JD, JD[pick]]), kind="stable")
        moves.append([float(x[0]) for x in offsets(BASE, *(np.concatenate([a, a[pick]])[order] for a in (JD, PSI, PLX)))])
    moves = np.array(moves)
    out["candidate_2_epochs"] = {"dec_offset_16_50_84": percentiles(moves[:, 1]), "median_move_dec": float(np.median(moves[:, 1]) - dec0), "ra_offset_16_50_84": percentiles(moves[:, 0])}
    # the same epochs as the RUWE of the default form: the published run, no epoch rejected, and five more epochs (copies, 100 choices of 1,000 draws)
    cat_row = {"phot_g_mean_mag": G, "nu_eff_used_in_astrometry": float(cat["nu_eff_used_in_astrometry"]), "ruwe": float(cat["ruwe"])}
    orbit_combined = {"period": 4253.1, "ecc": 0.7291, "a0": 27.39, "inclination": 110.580, "node": 136.236, "omega": 77.34, "t_peri_jd": 2458177.39, "parallax": (1.6933, 0.0164), "pmra": (-28.317, 0.067), "pmdec": (-155.221, 0.111)}

    def ruwe_median(s):
        return float(np.median(bh3.predict(s, cat_row, orbit_combined, "orbit")["ruwe"]))

    published = ruwe_median(scan)
    saved = gaiafit.REJECTED_SHARE
    gaiafit.REJECTED_SHARE = 0.0
    none_rejected = ruwe_median(scan)
    copies = []
    old_draws = bh3.DRAWS
    bh3.DRAWS = 1000
    for s in range(100):
        pick = np.random.default_rng(1694 + s).integers(0, len(JD), 5)
        order = np.argsort(np.concatenate([JD, JD[pick]]), kind="stable")
        copies.append(ruwe_median(wb.Scan(*(np.concatenate([a, a[pick]])[order] for a in (JD, PSI, PLX)), device="cpu")))
    gaiafit.REJECTED_SHARE = saved
    bh3.DRAWS = old_draws
    out["ruwe_with_more_epochs"] = {"published_run_median": published, "none_rejected_median": none_rejected, "47_epochs_none_rejected_16_50_84_over_100_choices": percentiles(copies), "47_epochs_none_rejected_min_max": [float(min(copies)), float(max(copies))]}
    rows = {}
    for label, noisy, rejected in (("noise-free, none rejected", False, False), ("noise, 10 percent rejected", True, True), ("noise, none rejected", True, False)):
        count = 200 if noisy else 1
        s, observed, keep = abscissae(BASE, JD, PSI, PLX, count, noisy)
        keep = keep if rejected else np.ones_like(keep)
        base = wb.fit_five_parameter(s.t_yr(REF), s.psi, s.plx_factor, observed, SIGMA, keep=keep, errors=False, device="cpu")["parameters"].numpy()
        design = design_of(s.t_yr(REF).numpy(), PSI, PLX)
        row = {"unweighted_ra_dec": [float(np.median(base[:, 1] - BASE["pmra"])), float(np.median(base[:, 3] - BASE["pmdec"]))]}
        sum_unweighted = np.sqrt(np.sum(np.where(keep, observed - base @ design.T, 0.0) ** 2, axis=1))
        for variant in ("solved", "fixed"):
            mu, eps, w, _ = fit_agis(observed, keep.astype(float), variant)
            ratio = np.sqrt(np.sum(np.where(keep, observed - mu @ design.T, 0.0) ** 2, axis=1)) / sum_unweighted
            row[f"agis_eps_{variant}"] = {"ratio_of_the_root_sum_of_squared_residuals_to_the_unweighted_fit": float(np.median(ratio)), "ra_16_50_84": percentiles(mu[:, 1] - BASE["pmra"]), "dec_16_50_84": percentiles(mu[:, 3] - BASE["pmdec"]), "eps_median_mas": float(np.median(eps)), "share_of_epochs_with_weight_under_0.2": float(np.mean(w[keep] < 0.2))}
        rows[label] = row
    out["candidate_3_agis_weighting"] = {"excess_noise_of_dr3_mas": AEN, **rows}
    omega = np.array([19.4, 59.9, 0.2]) / 1000.0
    a, d = math.radians(RA), math.radians(DEC)
    correction = [-math.sin(d) * math.cos(a) * omega[0] - math.sin(d) * math.sin(a) * omega[1] + math.cos(d) * omega[2], math.sin(a) * omega[0] - math.cos(a) * omega[1]]
    corrected = [out["catalogue_offsets"][0] - correction[0], out["catalogue_offsets"][1] - correction[1]]
    out["candidate_4_frame"] = {"omega_mas_per_yr": omega.tolist(), "correction_ra_dec_mas_per_yr": correction, "catalogue_offsets_corrected": corrected, "difference_from_the_prediction_ra_dec": [corrected[0] - ra0, corrected[1] - dec0],
                                "in_sigma": [(corrected[k] - [ra0, dec0][k]) / sigma_total[k] for k in (0, 1)]}
    out["versions"] = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__}
    (HERE / "bh3_proper_motion.json").write_text(json.dumps(out, indent=1) + "\n")
    print("predicted (noise-free)", round(ra0, 3), round(dec0, 3), "catalogue", [round(x, 3) for x in out["catalogue_offsets"]], "sigma_el", [round(x, 3) for x in sigma_el])
    print("candidate 1", {k: (round(v["difference"], 3), round(v["in_sigma"], 2)) for k, v in out["candidate_1_elements"].items()}, "element draws dec", [round(x, 3) for x in out["element_draws"]["dec_16_50_84"]], "share", out["element_draws"]["share_dec_at_or_above_the_catalogue"])
    print("candidate 2", out["candidate_2_epochs"]["dec_offset_16_50_84"], "move", round(out["candidate_2_epochs"]["median_move_dec"], 3))
    for label, row in rows.items():
        print("candidate 3", label, "unweighted", [round(x, 3) for x in row["unweighted_ra_dec"]], "residual sum ratio solved", round(row["agis_eps_solved"]["ratio_of_the_root_sum_of_squared_residuals_to_the_unweighted_fit"], 3), "fixed", round(row["agis_eps_fixed"]["ratio_of_the_root_sum_of_squared_residuals_to_the_unweighted_fit"], 3), "solved dec", [round(x, 3) for x in row["agis_eps_solved"]["dec_16_50_84"]], "fixed dec", [round(x, 3) for x in row["agis_eps_fixed"]["dec_16_50_84"]])
    print("candidate 4", [round(x, 4) for x in correction], [round(x, 3) for x in corrected], [round(x, 2) for x in out["candidate_4_frame"]["in_sigma"]])


if __name__ == "__main__":
    main()
