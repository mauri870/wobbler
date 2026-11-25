"""Gaia BH3: what the published orbit predicts for the five-parameter fit of Gaia DR3, against what DR3 reports.

usage: GAIAMOCK_DATA_DIR=... python validation/bh3.py [--catalogue]
(--catalogue asks the Gaia archive for the DR3 row of the source and writes validation/bh3_catalogue.json; without it the file is read. Writes validation/bh3.json. Needs gaiamock and its scanning-law data for Scan.from_position.)

Inputs. The orbit is Table 2 of Gaia Collaboration, Panuzzo et al. 2024, A&A 686, L2 (arXiv:2404.10486, version 2, page 4 of its PDF; ADS 2024A&A...686L...2G), the "Combined solution" column, whose elements are Campbell elements computed from the Thiele-Innes elements of Gaia's pre-release DR4 pipeline with the equations of Appendix A of Halbwachs et al. 2023:
period 4253.1 d, eccentricity 0.7291, photocentre semi-major axis a0 27.39 mas, inclination 110.580 deg, node 136.236 deg, argument of periastron 77.34 deg, time of periastron JD 2458177.39 (TCB), parallax 1.6933 mas, proper motion -28.317 and -155.221 mas/yr.
The "Astrometric solution" column (P 4194.7 d, e 0.7262, a0 27.07 mas, i 110.659 deg, node 136.200 deg, omega 77.77 deg, Tp JD 2458177.28, parallax 1.6747 mas, proper motion -28.372 and -155.150 mas/yr) is run as a sensitivity. The scan is the DR3 scanning law of gaiamock at the position of the source, and the catalogued values are one query of gaiadr3.gaia_source of the Gaia archive (https://gea.esac.esa.int/tap-server/tap).
Every element is given, so no orientation is drawn; the noise is.

Rule, written before the first run of this script (2026-11-23; the commit that holds this text precedes the commit of the results):
  Prediction: 4,000 noise draws of the DR3 five-parameter fit under the combined orbit with the DR3 scan; the median and the 16th and 84th percentiles of the RUWE and of the fitted parallax and proper motion minus the orbit's barycentric values.
  R1 (RUWE): the catalogued RUWE is at least 1.4, and |ln(median predicted RUWE / catalogued RUWE)| is at most 0.14.
     The 1.4 is the usual value above which Gaia's RUWE is read as a poor single-star fit; 0.14 is the scatter in the logarithm of the RUWE that gaiamock predicts for DR3 orbits (El-Badry 2025, OJAp 8, 62, Section 2.2: 16.6 percent; Mueller-Horn et al. 2026, A&A 709, A62, adopted 0.14).
  R2 (offsets): for the parallax or for one component of the proper motion, the catalogued value minus the orbit's barycentric value exceeds 3 times the catalogue's error, and lies within 2 sigma of the median predicted offset, with sigma the quadrature sum of the catalogue's error, the orbit's error of that barycentric value (Table 2) and the half-width of the 16th to 84th percentile of the predicted offsets.
  BH3 is worth showcasing beside Gaia BH1 if R1 or R2 holds. It is not if DR3's values carry no signal of the orbit (the RUWE under 1.4 and every offset within 3 sigma of zero), or if the prediction fails both R1 and R2, and then the reason is stated.
Limits to state with the result: the orbit's period (11.6 years) is longer than DR3's 34 months, so the fit sees an arc; the star is bright (G about 11.2), where the factor on the unit weight error (wobbler.uwe_factor) is about 1.9 and its check against orbit stars is about 12 percent high at G 11 to 12; the orbit comes from a preliminary pipeline whose parallax bias is not quantified (Panuzzo et al. 2024, Section 2.2); the DR3 parallax of the archive has no zero-point correction, so the comparison is with the uncorrected value.
Sensitivity (not part of the rule): the astrometric-only orbit, and the unit weight error without the factor (uwe_factor = 1).
Amendment (2026-11-23, after the first run of this script and a calibration of the unit weight error on 60,940 DR3 orbit stars): the default of the package is now the orbit form sqrt(f^2 + B^2 (w^2 - 1)) of each draw's raw unit weight error w (wobbler.calibrated_uwe), which holds the nine bins of G on the held-out half of those stars where the product f x w does not. The rule R1 and R2 above are read on the default; the product (the form of the first run: RUWE 5.07 with the combined orbit) and the raw simulation (2.71) stay as sensitivities.
"""
import gzip
import json
import os
import platform
import sys
import urllib.parse
import urllib.request
from pathlib import Path

os.environ.setdefault("WOBBLER_COMPILE", "0")

import numpy as np
import torch

import wobbler as wb

HERE = Path(__file__).parent
SOURCE_ID = 4318465066420528000  # Panuzzo et al. 2024, Section 2.1
TAP = "https://gea.esac.esa.int/tap-server/tap/sync"
DRAWS, SEED = 4000, 47
REFERENCE_JD = 2457389.0  # J2016.0, the reference epoch of DR3 and of wobbler's times
ORBITS = {  # Table 2 of Panuzzo et al. 2024: value and error of each element
    "combined": dict(period=4253.1, ecc=0.7291, a0=27.39, inclination=110.580, node=136.236, omega=77.34, t_peri_jd=2458177.39, parallax=(1.6933, 0.0164), pmra=(-28.317, 0.067), pmdec=(-155.221, 0.111)),
    "astrometric": dict(period=4194.7, ecc=0.7262, a0=27.07, inclination=110.659, node=136.200, omega=77.77, t_peri_jd=2458177.28, parallax=(1.6747, 0.0094), pmra=(-28.372, 0.077), pmdec=(-155.150, 0.129)),
}


def catalogue_from_archive():
    adql = ("SELECT ra,dec,phot_g_mean_mag,parallax,parallax_error,pmra,pmra_error,pmdec,pmdec_error,ruwe,nu_eff_used_in_astrometry,astrometric_params_solved,astrometric_excess_noise,astrometric_n_good_obs_al,visibility_periods_used "
            f"FROM gaiadr3.gaia_source WHERE source_id={SOURCE_ID}")
    data = urllib.parse.urlencode({"QUERY": adql, "REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv"}).encode()
    lines = urllib.request.urlopen(TAP, data, timeout=120).read().decode().splitlines()
    out = {"source_id": SOURCE_ID, "read_on": "2026-11-23", **dict(zip(lines[0].split(","), lines[1].split(",")))}
    (HERE / "bh3_catalogue.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def predict(scan, c, orbit, mode):
    """DRAWS noise draws of the DR3 fit under `orbit`: RUWE and the fitted parameters minus the barycentric parallax and proper motion of the orbit. mode: `orbit` (the default form of the package), `product` (f x the raw unit weight error) or `raw`."""
    rng = np.random.default_rng(SEED)
    epochs = len(scan.jd)
    t = scan.t_yr(REFERENCE_JD)
    magnitude = c["phot_g_mean_mag"]
    error = float(wb.epoch_error(magnitude, device="cpu"))
    z_err, z_extra = rng.normal(size=(DRAWS, epochs)), rng.normal(size=(DRAWS, epochs))
    keep = rng.uniform(size=(DRAWS, epochs)) > wb.gaiafit.REJECTED_SHARE
    extra = rng.uniform(0, wb.gaiafit.BRIGHT_EXTRA_NOISE, DRAWS) if magnitude < wb.gaiafit.BRIGHT_LIMIT else np.zeros(DRAWS)
    parallax, pmra, pmdec = orbit["parallax"][0], orbit["pmra"][0], orbit["pmdec"][0]
    observed = wb.observe(scan, parallax, pmra, pmdec, orbit["a0"], np.radians(orbit["inclination"]), np.radians(orbit["node"]), orbit["period"], orbit["ecc"], orbit["t_peri_jd"] - REFERENCE_JD, np.radians(orbit["omega"]),
                          error, z_err, extra, z_extra, device="cpu")
    fit = wb.fit_five_parameter(t, scan.psi, scan.plx_factor, observed, error, keep=keep, errors=False, device="cpu")
    ok = fit["ok"].numpy()
    u0 = float(wb.u0(magnitude, c["nu_eff_used_in_astrometry"], device="cpu"))
    raw = fit["uwe"].numpy()[ok]
    uwe = {"raw": raw, "product": raw * float(wb.uwe_factor(magnitude)), "orbit": wb.calibrated_uwe(magnitude, raw)}[mode]
    ruwe = uwe / u0
    parameters = fit["parameters"].numpy()[ok]  # ra offset, pmra, dec offset, pmdec, parallax
    return {"raw_uwe": raw, "ruwe": ruwe, "parallax": parameters[:, 4] - parallax, "pmra": parameters[:, 1] - pmra, "pmdec": parameters[:, 3] - pmdec, "fits": int(ok.sum())}


def percentiles(values):
    return [float(x) for x in np.percentile(values, [16, 50, 84])]


def main():
    raw = catalogue_from_archive() if "--catalogue" in sys.argv else json.loads((HERE / "bh3_catalogue.json").read_text())
    c = {k: float(v) for k, v in raw.items() if k not in ("read_on", "source_id") and v not in ("", None)}
    scan = wb.Scan.from_position(c["ra"], c["dec"], window="DR3", device="cpu")
    factor = float(wb.uwe_factor(c["phot_g_mean_mag"]))
    results = {}
    for name, orbit in ORBITS.items():
        for label, mode in (("with the orbit form", "orbit"), ("with the product", "product"), ("uwe factor 1", "raw")):
            p = predict(scan, c, orbit, mode)
            row = {"fits": p["fits"], "uwe_factor": factor, "orbit_factor": float(wb.orbit_factor(c["phot_g_mean_mag"])), "ruwe_16_50_84": percentiles(p["ruwe"])}
            row["ln_median_predicted_over_catalogue_ruwe"] = float(np.log(row["ruwe_16_50_84"][1] / c["ruwe"]))
            for key, catalogue, error_key in (("parallax", c["parallax"], "parallax_error"), ("pmra", c["pmra"], "pmra_error"), ("pmdec", c["pmdec"], "pmdec_error")):
                barycentre, barycentre_error = orbit[key]
                observed = catalogue - barycentre
                low, mid, high = percentiles(p[key])
                sigma = float(np.sqrt(c[error_key] ** 2 + barycentre_error**2 + ((high - low) / 2) ** 2))
                row[key] = {"catalogue_minus_barycentre": observed, "catalogue_error": c[error_key], "barycentre_error": barycentre_error, "predicted_offset_16_50_84": [low, mid, high], "sigma_of_the_comparison": sigma,
                            "offset_in_catalogue_errors": observed / c[error_key], "distance_from_prediction_in_sigma": (observed - mid) / sigma,
                            "R2_condition_met": bool(abs(observed) > 3 * c[error_key] and abs(observed - mid) <= 2 * sigma)}
            row["R1_condition_met"] = bool(c["ruwe"] >= 1.4 and abs(row["ln_median_predicted_over_catalogue_ruwe"]) <= 0.14)
            results[f"{name} orbit, {label}"] = row
    # where BH3 lies among DR3 orbit stars of its magnitude: the ratio of the catalogued unit weight error to the prediction of the orbit form, against the 12,000 stars recorded in tests/golden/orbit_stars_dr3.csv.gz
    with gzip.open(HERE.parent / "tests" / "golden" / "orbit_stars_dr3.csv.gz", "rt") as handle:
        handle.readline()
        stars = np.array([[float(x) for x in line.split(",")[1:]] for line in handle])
    star_g, star_w, star_archive = stars.T
    w_bh3 = float(np.median(predict(scan, c, ORBITS["combined"], "raw")["raw_uwe"]))
    u0_bh3 = float(wb.u0(c["phot_g_mean_mag"], c["nu_eff_used_in_astrometry"], device="cpu"))
    r_bh3 = c["ruwe"] * u0_bh3 / float(wb.calibrated_uwe(c["phot_g_mean_mag"], w_bh3))
    ratio = star_archive / wb.calibrated_uwe(star_g, star_w)
    band = (star_g >= 11.0) & (star_g < 11.25)
    near = band & (np.abs(star_w / w_bh3 - 1) <= 0.20)
    population = {"raw_uwe_of_bh3": w_bh3, "ratio_of_bh3": r_bh3, "stars_in_the_band_of_g_11_to_11.25": int(band.sum()), "ratio_of_the_band_16_50_84": percentiles(ratio[band]), "percentile_of_bh3_in_the_band": float(100 * np.mean(ratio[band] < r_bh3)),
                  "stars_with_w_within_20_percent": int(near.sum()), "percentile_of_bh3_among_them": float(100 * np.mean(ratio[near] < r_bh3))}
    print("BH3 among the recorded orbit stars of its magnitude:", {k: round(v, 3) if isinstance(v, float) else v for k, v in population.items() if k != "ratio_of_the_band_16_50_84"}, "band ratio", [round(x, 3) for x in population["ratio_of_the_band_16_50_84"]])
    out = {"catalogue": raw, "orbit_form_among_orbit_stars": population, "draws": DRAWS, "seed": SEED, "observations_in_the_dr3_window": int(len(scan.jd)), "orbits": ORBITS, "results": results, "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__}}
    (HERE / "bh3.json").write_text(json.dumps(out, indent=1) + "\n")
    print("catalogue RUWE", c["ruwe"], "parallax", c["parallax"], "pm", c["pmra"], c["pmdec"], "epochs", len(scan.jd))
    for name, row in results.items():
        print(name, "RUWE", [round(x, 2) for x in row["ruwe_16_50_84"]], "ln", round(row["ln_median_predicted_over_catalogue_ruwe"], 3), "R1", row["R1_condition_met"])
        for key in ("parallax", "pmra", "pmdec"):
            r = row[key]
            print("   ", key, "catalogue-bary", round(r["catalogue_minus_barycentre"], 3), "predicted", [round(x, 3) for x in r["predicted_offset_16_50_84"]], "sigma", round(r["sigma_of_the_comparison"], 3), "dist", round(r["distance_from_prediction_in_sigma"], 2), "R2", r["R2_condition_met"])


if __name__ == "__main__":
    main()
