"""The unbinned mode against gaiamock's unbinned mode (binned=False), at three real positions with the real per-CCD scans and the real map, at the same draws.

usage: GAIAMOCK_DATA_DIR=... python validation/unbinned.py   (writes validation/unbinned.json)
Needs gaiamock, its scanning-law data, and the two files of its README section "A modified version to predict RUWE more reliably for small orbits and single stars": individual_ccds.zip and healpix_16_med_ruwe.npz, installed with `gaiamock-download individual_ccds.zip healpix_16_med_ruwe.npz`.

Positions: Gaia BH1, Gaia BH3 and the pixel of the map with the highest median RUWE. For each, 4 orbits drawn with seeds 100 to 103 and noise drawn by gaiamock with numpy seeds 0 to 3. gaiamock's predict_astrometry_binary_in_terms_of_a0 and fit_5par_solution_only (binned=False) give the reference. wobbler's simulate_fit(binned=False) is given the draws gaiamock made, in the order gaiamock makes them (the rejection of a tenth of the epochs, the jitter of the error, the noise), on the same epochs.
Rule: the two agree when the relative difference of the RUWE is at most 1e-11 and the parallax differs by at most 1e-11 mas in every case.
"""
import hashlib
import json
from pathlib import Path

import gaiamock
import healpy
import numpy as np

import wobbler as wb
from wobbler import unbinned

HERE = Path(__file__).parent
TOLERANCE = 1e-11
WINDOW = {"DR3": (2456891.5, 2457902.0)}
POSITIONS = {"Gaia BH1": (262.17120727595824, -0.5810919626714084, 13.771998), "Gaia BH3": (294.82786250815144, 14.930979608612361, 11.231088)}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    sky_path = unbinned.sky_map_path()
    sky = unbinned.read_sky_map(sky_path)
    hot = int(np.argmax(sky))
    theta, phi = healpy.pix2ang(16, hot, nest=False)
    positions = dict(POSITIONS)
    positions[f"pixel {hot}, the highest median RUWE of the map"] = (float(np.degrees(phi)), float(90 - np.degrees(theta)), 13.0)
    c_funcs = gaiamock.read_in_C_functions()
    out = {"gaiamock": gaiamock.__version__, "map_sha256": sha256(sky_path), "tolerance": TOLERANCE, "positions": {}}
    worst_ruwe, worst_parallax = 0.0, 0.0
    for name, (ra, dec, magnitude) in positions.items():
        pixel = int(unbinned.ang2pix_ring(16, ra, dec))
        assert pixel == healpy.ang2pix(16, np.radians(90 - dec), np.radians(ra))
        table = gaiamock.get_gost_one_position(ra, dec, data_release="dr3", binned=False)
        jd = np.asarray(gaiamock.fetch_table_element("ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]", table), float)
        order = np.argsort(jd, kind="stable")
        scan = wb.Scan.from_position(ra, dec, window="DR3", binned=False, device="cpu")
        assert len(scan) == len(jd) and np.array_equal(scan.jd.numpy(), jd[order]), "wobbler's scan is gaiamock's table sorted by time"
        cases = []
        for seed in range(4):
            rng = np.random.default_rng(100 + seed)
            period, ecc, t_peri, node, omega = rng.uniform(150, 900), rng.uniform(0, 0.7), rng.uniform(-300, 300), rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)
            inclination, a0, parallax, pmra, pmdec = np.arccos(rng.uniform(0.1, 1)), rng.uniform(0.3, 3), rng.uniform(0.5, 4), rng.normal(0, 5), rng.normal(0, 5)
            np.random.seed(seed)
            t, psi, plx, observed, error = gaiamock.predict_astrometry_binary_in_terms_of_a0(ra=ra, dec=dec, parallax=parallax, pmra=pmra, pmdec=pmdec, period=period, Tp=t_peri, ecc=ecc, omega=node, inc=inclination, w=omega, a0_mas=a0, phot_g_mean_mag=magnitude, data_release="dr3", c_funcs=c_funcs, binned=False)
            expected = gaiamock.fit_5par_solution_only(t_ast_yr=t, psi=psi, plx_factor=plx, ast_obs=observed, ast_err=error, binned=False)
            draws = np.random.RandomState(seed)
            keep = draws.uniform(0, 1, len(jd)) > 0.1
            jitter = draws.randn()
            z = np.zeros(len(jd))
            z[keep] = draws.randn(keep.sum())
            assert keep.sum() == len(t)
            noise = {"keep": keep[order][None, :], "z_err": z[order][None, :], "jitter": np.array([jitter])}
            fit = wb.simulate_fit(scan, wb.Orbit(period_days=period, ecc=ecc, t_peri_days=t_peri, omega=omega), a1_au=a0 * np.sin(inclination) / parallax, parallax_mas=np.array([parallax]), G=magnitude, n_orient=1, inclination=inclination, node=node, pm=(pmra, pmdec),
                                  noise=noise, windows=WINDOW, binned=False, position=(ra, dec), sky_map=sky_path, device="cpu")
            ours = fit.windows["DR3"]
            relative, difference = float(ours["uwe"]) / expected[10] - 1, abs(float(ours["parallax_mas"]) - expected[4])
            worst_ruwe, worst_parallax = max(worst_ruwe, abs(relative)), max(worst_parallax, difference)
            cases.append({"seed": seed, "epochs_kept": int(ours["n_epochs"]), "ruwe_gaiamock": float(expected[10]), "ruwe_wobbler": float(ours["uwe"]), "relative_difference": relative, "parallax_difference_mas": difference})
        out["positions"][name] = {"ra": ra, "dec": dec, "G": magnitude, "pixel": pixel, "median_ruwe_of_pixel": float(sky[pixel]), "ccd_epochs": len(scan), "cases": cases}
    out["largest_relative_difference_of_the_ruwe"], out["largest_parallax_difference_mas"] = worst_ruwe, worst_parallax
    out["agree"] = bool(worst_ruwe <= TOLERANCE and worst_parallax <= TOLERANCE)
    (HERE / "unbinned.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({key: out[key] for key in ("largest_relative_difference_of_the_ruwe", "largest_parallax_difference_mas", "agree")}))


if __name__ == "__main__":
    main()
