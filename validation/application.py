"""Competing companion hypotheses for Gaia BH1 against its catalogued RUWE.

usage: GAIAMOCK_DATA_DIR=... python validation/application.py   (writes validation/application.json; needs gaiamock and its scanning-law data for Scan.from_position)

Inputs, all public: Gaia DR3 4373465352415301632 (gaia_source: position, G, parallax, RUWE, nu_eff_used_in_astrometry; the Gaia archive, read 2026-11-20) and the spectroscopic orbit of El-Badry et al. 2023 (MNRAS 518, 1057, Table 1): period, eccentricity, time of periastron, argument of periastron, velocity semi-amplitude K and the mass of the visible star. The scan is the scanning law of gaiamock at the position.
Each hypothesis fixes the orbit that the radial velocities give and leaves the inclination (isotropic) and the node (uniform) free. The companion's mass follows from the mass function at each inclination. A hypothesis differs in the light of the companion, which moves the photocentre that Gaia measures.
"""
import json
import os
import platform
from pathlib import Path

os.environ.setdefault("WOBBLER_COMPILE", "0")

import numpy as np
import torch

import wobbler as wb
from wobbler import light

HERE = Path(__file__).parent
# Gaia DR3 gaia_source of 4373465352415301632 (Gaia archive, 2026-11-20)
GAIA = dict(ra=262.17120727595824, dec=-0.5810919626714084, G=13.771998, parallax=2.069988584035135, ruwe=7.6376343, nu_eff=1.4811925)
# El-Badry et al. 2023, Table 1 (astrometry + RVs): period, eccentricity, time of periastron (days from JD 2457389.0), argument of periastron; K of the RV-only fit; mass of the visible star
ORBIT = dict(period=185.59, ecc=0.451, t_peri=-1.1, omega=np.radians(12.8), k_star=66.7, m_star=0.93)
AU_KM = 1.495978707e8
DRAWS, SEED, TOLERANCES = 20000, 11, (0.05, 0.10, 0.20)
FLUX_RATIOS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0)
EXPONENTS = (3.0, 4.0, 5.0)  # the flux ratio of a main-sequence companion, (M2 / M1) ** k in the G band


def hypotheses():
    yield "single star", dict(a1_scale=0.0)
    yield "dark companion", dict()
    for k in EXPONENTS:
        yield f"luminous companion, flux ratio (M2/M1)^{k:g}", dict(flux_ratio=lambda m, k=k: (m / ORBIT["m_star"]) ** k, primary_mass=ORBIT["m_star"])


def main():
    scan = wb.Scan.from_position(GAIA["ra"], GAIA["dec"], window="DR3", device="cpu")
    a1_au = ORBIT["k_star"] * ORBIT["period"] * 86400 * np.sqrt(1 - ORBIT["ecc"] ** 2) / (2 * np.pi) / AU_KM
    f = float(light.mass_function(period_days=ORBIT["period"], k_kms=ORBIT["k_star"], ecc=ORBIT["ecc"]))
    orbit = wb.Orbit(period_days=ORBIT["period"], ecc=ORBIT["ecc"], t_peri_days=ORBIT["t_peri"], omega=ORBIT["omega"])
    window = {"DR3": wb.WINDOWS["DR3"]}
    results = {}
    for name, options in hypotheses():
        scale = options.pop("a1_scale", 1.0)
        fit = wb.simulate_fit(scan, orbit, a1_au=a1_au * scale, parallax_mas=np.array([GAIA["parallax"]]), G=GAIA["G"], n_orient=DRAWS, windows=window, seed=SEED, device="cpu", nu_eff=GAIA["nu_eff"], **options)
        w = fit.windows["DR3"]
        ok = w["ok"][0].numpy()
        ruwe = fit.ruwe[0].numpy()[ok]
        row = {"fits": int(ok.sum()), "ruwe_16_50_84": [float(x) for x in np.percentile(ruwe, [16, 50, 84])], "share_at_or_above_catalogue_ruwe": float(np.mean(ruwe >= GAIA["ruwe"]))}
        for tolerance in TOLERANCES:
            row[f"share_within_{int(tolerance * 100)}_percent_of_catalogue_ruwe"] = float(np.mean(np.abs(ruwe / GAIA["ruwe"] - 1) <= tolerance))
        if scale:
            mass = wb.mass.companion_mass(f, ORBIT["m_star"], np.arccos(np.random.default_rng(SEED).uniform(0, 1, 100000)), device="cpu").numpy()
            row["companion_mass_16_50_84_solar"] = [float(x) for x in np.percentile(mass, [16, 50, 84])]
        results[name] = row
    scan_rows = []  # a companion that gives a fixed share of the light: RUWE against the flux ratio
    for ratio in FLUX_RATIOS:
        options = dict(flux_ratio=ratio, primary_mass=ORBIT["m_star"]) if ratio else {}
        fit = wb.simulate_fit(scan, orbit, a1_au=a1_au, parallax_mas=np.array([GAIA["parallax"]]), G=GAIA["G"], n_orient=DRAWS, windows=window, seed=SEED, device="cpu", nu_eff=GAIA["nu_eff"], **options)
        ok = fit.windows["DR3"]["ok"][0].numpy()
        ruwe = fit.ruwe[0].numpy()[ok]
        scan_rows.append({"flux_ratio": ratio, "ruwe_16_50_84": [float(x) for x in np.percentile(ruwe, [16, 50, 84])], "share_at_or_above_catalogue_ruwe": float(np.mean(ruwe >= GAIA["ruwe"]))})
    out = {"mass_function_solar": f, "a1_sin_i_au": float(a1_au), "minimum_companion_mass_solar": float(wb.mass.min_companion_mass(f, ORBIT["m_star"], device="cpu")), "draws": DRAWS, "seed": SEED, "catalogue": GAIA, "orbit": {k: float(v) for k, v in ORBIT.items()},
           "hypotheses": results, "flux_ratio_scan": scan_rows, "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__}}
    (HERE / "application.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
