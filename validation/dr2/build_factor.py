"""The first build of DR2's single-star factor f_DR2(G): for each of 41 bands of G, up to 5,000 ordinary Gaia DR2 stars (the first rows the archive returns) and the median UWE that wobbler simulates for a single star in the DR2 window at the band's centre (300 draws on each of 60 seeded sky positions, median of the position medians). Writes validation/dr2/factor_dr2_first_build.json, the input of refit_factor.py, which makes the table.
usage: python validation/dr2/build_factor.py [CACHE_DIRECTORY]   (needs the network and the gaiamock extra; the downloaded bands are kept in CACHE_DIRECTORY, default /tmp/wobbler-dr2-factor). The rows of a band with more than 5,000 stars are the lowest-G ones: refit_factor.py and fetch_bands.py replace them."""
import csv
import io
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

import wobbler as wb
from wobbler.gaiafit import WINDOWS

HERE = Path(__file__).resolve().parent
CACHE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/wobbler-dr2-factor")
URL = "https://gea.esac.esa.int/tap-server/tap/sync"
EDGES = [4.5 + 0.5 * k for k in range(12)] + [10.5 + 0.1 * k for k in range(15)] + [12.0 + 0.5 * k for k in range(15)]  # 0.5 mag from G 4.5 to 10.5, 0.1 mag from 10.5 to 12, 0.5 mag from 12 to 19.5 (19.5 is dropped below)
EDGES = [round(e, 2) for e in EDGES if e <= 19.0]
PER_BAND, POSITIONS, DRAWS, SEED = 5000, 60, 300, 826
ADQL = """SELECT TOP {top} phot_g_mean_mag, astrometric_chi2_al, astrometric_n_good_obs_al FROM gaiadr2.gaia_source
WHERE astrometric_params_solved = 31 AND bp_rp BETWEEN 0.90 AND 1.55 AND astrometric_n_good_obs_al > 20 AND phot_g_mean_mag >= {low} AND phot_g_mean_mag < {high}{index}"""


def band_rows(low, high):
    path = CACHE / f"band_{low}_{high}.csv"
    if not path.exists():
        index = " AND random_index < 200000000" if low >= 8 else ""
        data = urllib.parse.urlencode({"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": ADQL.format(top=PER_BAND, low=low, high=high, index=index)}).encode()
        with urllib.request.urlopen(urllib.request.Request(URL, data=data), timeout=600) as response:
            path.write_text(response.read().decode())
    rows = list(csv.DictReader(io.StringIO(path.read_text())))
    g = np.array([float(r["phot_g_mean_mag"]) for r in rows])
    uwe = np.sqrt(np.array([float(r["astrometric_chi2_al"]) for r in rows]) / (np.array([float(r["astrometric_n_good_obs_al"]) for r in rows]) - 5.0))
    return g, uwe


def simulated_uwe(scans, magnitude):
    """The median over the sky positions of the median over DRAWS single-star draws of the raw simulated UWE in the DR2 window, and the spread (standard deviation) of the position medians."""
    orbit = wb.Orbit(period_days=100.0, ecc=0.0, t_peri_days=0.0, omega=0.0)
    medians = []
    for k, scan in enumerate(scans):
        fit = wb.simulate_fit(scan, orbit, a1_au=0.0, parallax_mas=np.array([1.0]), G=magnitude, n_orient=DRAWS, seed=SEED + k, windows={"DR2": WINDOWS["DR2"]}, uwe_factor=1.0, device="cpu")
        window = fit.windows["DR2"]
        good = window["ok"].numpy().reshape(-1)
        medians.append(float(np.median(window["uwe"].numpy().reshape(-1)[good])))
    return float(np.median(medians)), float(np.std(medians))


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    ra, dec = rng.uniform(0, 360, POSITIONS), np.degrees(np.arcsin(rng.uniform(-1, 1, POSITIONS)))
    scans = [wb.Scan.from_position(a, d, window="DR2") for a, d in zip(ra, dec)]
    bands = []
    for low, high in zip(EDGES[:-1], EDGES[1:]):
        began = time.time()
        g, uwe = band_rows(low, high)
        fit, held = np.arange(len(g)) % 2 == 0, np.arange(len(g)) % 2 == 1
        centre = 0.5 * (low + high)
        simulated, spread = simulated_uwe(scans, centre)
        bands.append({"G": [low, high], "f": float(np.median(uwe[fit]) / simulated), "n_archive": int(len(g)), "n_fit": int(fit.sum()), "n_heldout": int(held.sum()), "median_uwe_fit": float(np.median(uwe[fit])), "median_uwe_heldout": float(np.median(uwe[held])),
                      "simulated_single_star_uwe": simulated, "simulated_spread_over_positions": spread})
        print(f"G {low}-{high}: {len(g)} stars, archive {np.median(uwe[fit]):.3f}, simulated {simulated:.3f} (spread {spread:.3f}), f {bands[-1]['f']:.3f}, {time.time() - began:.0f} s", flush=True)
    (HERE / "factor_dr2_first_build.json").write_text(json.dumps({"bands": bands}, indent=1) + "\n")


if __name__ == "__main__":
    main()
