"""Figure of the README: Gaia BH3 through wobbler.

Gaia DR3 4318465066420528000 is a giant star in an 11.6-year orbit around a dark companion of 33 solar masses (Gaia Collaboration, Panuzzo et al. 2024, A&A 686, L2). Its orbit comes from Gaia's pre-release epoch astrometry; DR3 fitted it as a single star over 34 months.
Inputs: the combined orbit of Table 2 of that paper and the DR3 row of the source (validation/bh3.py), the scan of gaiamock at the position (Scan.from_position; `pip install gaiamock==1.0.0`, then `gaiamock-download`), and the results of validation/bh3_proper_motion.py (run it first; it writes validation/bh3_proper_motion.json).

usage: GAIAMOCK_DATA_DIR=... python docs/make_bh3_figure.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "validation"))
import bh3  # noqa: E402
import wobbler as wb  # noqa: E402

sys.path.insert(0, str(HERE))
from make_bh1_figure import earth_offsets  # noqa: E402

DRAWS = bh3.DRAWS


def main():
    raw = json.loads((HERE.parent / "validation" / "bh3_catalogue.json").read_text())
    c = {k: float(v) for k, v in raw.items() if k not in ("read_on", "source_id")}
    orbit = bh3.ORBITS["combined"]
    scan = wb.Scan.from_position(c["ra"], c["dec"], window="DR3", device="cpu")
    jd, psi, plx_factor = (x.numpy() for x in (scan.jd, scan.psi, scan.plx_factor))
    t = scan.t_yr(bh3.REFERENCE_JD).numpy()
    parallax = orbit["parallax"][0]
    a0, inc, node, omega = orbit["a0"], np.radians(orbit["inclination"]), np.radians(orbit["node"]), np.radians(orbit["omega"])
    t_peri = orbit["t_peri_jd"] - bh3.REFERENCE_JD
    proper = json.loads((HERE.parent / "validation" / "bh3_proper_motion.json").read_text())
    ruwe = {mode: bh3.predict(scan, c, orbit, mode)["ruwe"] for mode in ("orbit", "product", "raw")}

    # panel a: the path on the sky over the DR3 window, proper motion removed
    days = np.linspace(jd.min(), jd.max(), 800)
    loop_e, loop_n = earth_offsets(days, c["ra"], c["dec"], parallax)
    orb_e, orb_n = (x.cpu().numpy() for x in wb.photocentre((days - bh3.REFERENCE_JD) / 365.25, orbit["period"], orbit["ecc"], t_peri, a0, inc, node, omega))
    obs_loop_e, obs_loop_n = earth_offsets(jd, c["ra"], c["dec"], parallax)
    obs_orb_e, obs_orb_n = (x.cpu().numpy() for x in wb.photocentre(t, orbit["period"], orbit["ecc"], t_peri, a0, inc, node, omega))

    # panel b: the residuals of the single-star fit of one simulated scan, with and without the companion
    rng = np.random.default_rng(2)
    error = float(wb.epoch_error(c["phot_g_mean_mag"], device="cpu"))
    z_err, z_extra = rng.normal(size=(1, len(t))), rng.normal(size=(1, len(t)))
    design = np.stack([np.sin(psi), t * np.sin(psi), np.cos(psi), t * np.cos(psi), plx_factor], axis=1)
    residuals = {}
    for name, amplitude in (("single", 0.0), ("companion", a0)):
        observed = wb.observe(scan, parallax, 0.0, 0.0, amplitude, inc, node, orbit["period"], orbit["ecc"], t_peri, omega, error, z_err, 0.0, z_extra, device="cpu").numpy()
        fit = {k: v.numpy() if hasattr(v, "numpy") else v for k, v in wb.fit_five_parameter(t, psi, plx_factor, observed, error, device="cpu").items()}
        residuals[name] = observed[0] - design @ fit["parameters"][0]

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 11})
    fig = plt.figure(figsize=(16, 6.2), constrained_layout=True)
    grid = fig.add_gridspec(1, 4, width_ratios=[1.1, 1.1, 1.1, 0.9])
    a, b, d = (fig.add_subplot(grid[0, i]) for i in (0, 1, 2))
    e1, e2 = grid[0, 3].subgridspec(2, 1, hspace=0.12).subplots(sharex=True)
    a.plot(loop_e, loop_n, color="0.55", lw=2, label="single star")
    a.plot(loop_e + orb_e, loop_n + orb_n, color="C0", lw=1.6, label="star with its companion")
    a.plot(obs_loop_e + obs_orb_e, obs_loop_n + obs_orb_n, "o", color="C0", mec="k", mew=0.6, ms=6, label=f"Gaia's {len(jd)} observations")
    a.invert_xaxis()
    a.set_aspect("equal")
    a.set_xlabel("east (mas)")
    a.set_ylabel("north (mas)")
    a.set_title("The star's path on the sky\n(DR3 window, proper motion removed)")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False)
    years = 2000.0 + (jd - 2451544.5) / 365.25
    b.axhline(0, color="k", lw=0.6)
    b.plot(years, residuals["single"], "o", color="0.45", ms=5, label="single star")
    b.plot(years, residuals["companion"], "o", color="C3", ms=5, label="star with its companion")
    b.set_xticks([2014, 2015, 2016, 2017])
    b.set_xlim(2014.4, 2017.6)
    b.set_xlabel("year")
    b.set_ylabel("measurement minus fit, along the scan (mas)")
    b.set_title("How far Gaia's measurements\nmiss the single-star fit")
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False)
    bins = np.linspace(1.5, 7.5, 61)
    d.hist(ruwe["orbit"], bins=bins, color="C0", alpha=0.5, label=f"default (median {np.median(ruwe['orbit']):.2f})")
    d.hist(ruwe["product"], bins=bins, histtype="step", color="C1", lw=1, label=f"product f x w (median {np.median(ruwe['product']):.2f})")
    d.hist(ruwe["raw"], bins=bins, histtype="step", color="0.35", lw=1, label=f"raw simulation (median {np.median(ruwe['raw']):.2f})")
    d.axvline(c["ruwe"], color="k", lw=1.8, label=f"Gaia DR3: {c['ruwe']:.2f}")
    d.set_xlim(1.5, 7.5)
    d.set_xlabel("RUWE of the published orbit, noise draws")
    d.set_ylabel(f"number of {DRAWS:,} draws")
    d.set_title("The RUWE that DR3 should fit")
    d.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False)
    errors = proper["catalogue_errors"]
    barycentre_errors = [0.067, 0.111]
    weighting = proper["candidate_3_agis_weighting"]["noise-free, none rejected"]
    for axis, k, name in ((e1, 0, "right ascension"), (e2, 1, "declination")):
        observed = proper["catalogue_offsets"][k]
        predicted = proper["predicted_noise_free"][k]
        axis.errorbar([1], [observed], yerr=[np.hypot(errors[k], barycentre_errors[k])], fmt="s", color="k", ms=8, capsize=5, label="Gaia DR3")
        axis.errorbar([0], [predicted], yerr=[proper["sigma_elements"][k]], fmt="o", color="C0", ms=8, capsize=5, label="published orbit")
        w_solved, w_fixed = weighting["agis_eps_solved"]["ra_16_50_84" if k == 0 else "dec_16_50_84"][1], weighting["agis_eps_fixed"]["ra_16_50_84" if k == 0 else "dec_16_50_84"][1]
        axis.plot([0, 0], [w_solved, w_fixed], "^", mfc="none", color="C0", ms=9, label="with DR3's weighting")
        axis.set_xlim(-0.7, 1.7)
        axis.set_ylabel(f"{name} (mas/yr)")
    e2.set_xticks([0, 1], ["orbit", "DR3"])
    e1.set_title("Proper motion offsets\n(fit minus barycentre)")
    e2.set_ylim(-1.15, 0.2)
    e1.set_ylim(5.95, 6.5)
    e2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False)
    out = HERE / "fig-bh3.png"
    fig.savefig(out, dpi=100)
    print(out)


if __name__ == "__main__":
    main()
