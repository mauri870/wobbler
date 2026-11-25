"""Rows of gaiadr2.gaia_source for the 41 bands of build_factor.py, from the ARI Heidelberg TAP mirror, in windows of 0.02 mag of G: each window gives its first matching stars (five-parameter, BP-RP 0.90 to 1.55, astrometric_n_good_obs_al over 20), 200 per window for a 0.5 mag band and 1,000 for a 0.1 mag band, so that the rows of a band cover its G. (The archive returns rows in G order, and a query that selects by random_index does not finish on the mirror.)
usage: python fetch_bands.py OUT_DIRECTORY [LOW ...]   (only the bands whose lower edge is listed; a finished band file is kept)"""
import csv
import io
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

SYNC = "https://gaia.ari.uni-heidelberg.de/tap/sync"  # TAP sync endpoint, https://gaia.ari.uni-heidelberg.de/tap
EDGES = [4.5 + 0.5 * k for k in range(12)] + [10.5 + 0.1 * k for k in range(15)] + [12.0 + 0.5 * k for k in range(15)]
EDGES = [round(e, 2) for e in EDGES if e <= 19.0]
WINDOW, PER_BAND = 0.02, 5000
ADQL = """SELECT TOP {top} phot_g_mean_mag, astrometric_chi2_al, astrometric_n_good_obs_al FROM gaiadr2.gaia_source
WHERE astrometric_params_solved = 31 AND bp_rp BETWEEN 0.90 AND 1.55 AND astrometric_n_good_obs_al > 20 AND phot_g_mean_mag >= {low} AND phot_g_mean_mag < {high}"""


def query(top, low, high):
    data = urllib.parse.urlencode({"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": ADQL.format(top=top, low=low, high=high)}).encode()
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(SYNC, data=data), timeout=120) as response:
                return list(csv.reader(io.StringIO(response.read().decode())))[1:]
        except Exception as error:
            print(f"retry {attempt} G {low}-{high}: {error}", flush=True)
            time.sleep(5)
    raise RuntimeError(f"G {low}-{high} failed")


def main(out, only):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for low, high in zip(EDGES[:-1], EDGES[1:]):
        path = out / f"band_{low}_{high}.csv"
        if only and low not in only:
            continue
        if path.exists():
            continue
        count = int(round((high - low) / WINDOW))
        top = min(2000, PER_BAND // count)
        rows = []
        for k in range(count):
            rows += query(top, round(low + WINDOW * k, 3), round(low + WINDOW * (k + 1) if k + 1 < count else high, 3))
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(["phot_g_mean_mag", "astrometric_chi2_al", "astrometric_n_good_obs_al"])
        writer.writerows(rows)
        path.write_text(buffer.getvalue())
        print(f"G {low}-{high}: {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], {float(x) for x in sys.argv[2:]})
