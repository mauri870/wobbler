"""Gaia BH1: what the published orbit predicts for the RUWE of Gaia DR3, against the catalogue.

usage: GAIAMOCK_DATA_DIR=... python validation/bh1.py   (writes validation/bh1.json; needs gaiamock and its scanning-law data)

Inputs, all public: the orbit of El-Badry et al. 2023 (MNRAS 518, 1057, Table 1, astrometry + RVs): period 185.59 d, eccentricity 0.451, time of periastron -1.1 d from JD 2457389.0, argument of periastron 12.8 deg, node 97.8 deg, inclination 126.6 deg, photocentre semi-major axis 2.67 mas, parallax 2.09 mas; and the Gaia DR3 row of Gaia DR3 4373465352415301632 (G 13.771998, nu_eff_used_in_astrometry 1.4811925, RUWE 7.6376343; the Gaia archive, read 2026-11-20, as validation/application.py). The scan is the DR3 scanning law of gaiamock at the position. Every element is given, so no orientation is drawn; the noise is (4,000 draws, seed 11).
Rule (that of validation/bh3.py, R1): the published orbit is reproduced when the logarithm of the ratio of the median predicted RUWE to the catalogued one is at most 0.14 in absolute value (the scatter of gaiamock's RUWE prediction for DR3 orbits). The forms: the default of the package (the orbit form), the product f x w and the raw simulation. At G 13.77 the orbit factor of the package is the single-star factor, so the first two agree.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import bh3  # noqa: E402
import wobbler as wb  # noqa: E402

bh3.DRAWS, bh3.SEED = 4000, 11
CATALOGUE = dict(ra=262.17120727595824, dec=-0.5810919626714084, phot_g_mean_mag=13.771998, nu_eff_used_in_astrometry=1.4811925, ruwe=7.6376343)
ORBIT = dict(period=185.59, ecc=0.451, a0=2.67, inclination=126.6, node=97.8, omega=12.8, t_peri_jd=bh3.REFERENCE_JD - 1.1, parallax=(2.09, 0.0), pmra=(0.0, 0.0), pmdec=(0.0, 0.0))


def main():
    scan = wb.Scan.from_position(CATALOGUE["ra"], CATALOGUE["dec"], window="DR3", device="cpu")
    results = {}
    for label, mode in (("the default of the package (orbit form)", "orbit"), ("the product f x w", "product"), ("raw simulation", "raw")):
        ruwe = bh3.predict(scan, CATALOGUE, ORBIT, mode)["ruwe"]
        row = {"ruwe_16_50_84": bh3.percentiles(ruwe), "ln_median_over_catalogue": float(np.log(np.median(ruwe) / CATALOGUE["ruwe"]))}
        row["within_0.14"] = bool(abs(row["ln_median_over_catalogue"]) <= 0.14)
        results[label] = row
        print(label, [round(x, 2) for x in row["ruwe_16_50_84"]], "ln", round(row["ln_median_over_catalogue"], 3), "within", row["within_0.14"])
    out = {"catalogue_ruwe": CATALOGUE["ruwe"], "observations_in_the_dr3_window": int(len(scan.jd)), "uwe_factor": float(wb.uwe_factor(CATALOGUE["phot_g_mean_mag"])), "orbit_factor": float(wb.orbit_factor(CATALOGUE["phot_g_mean_mag"])), "results": results, "draws": bh3.DRAWS, "seed": bh3.SEED}
    (HERE / "bh1.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
