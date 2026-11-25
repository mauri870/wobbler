"""Accuracy scoreboard: wobbler's prediction against Gaia DR3 for every star with an orbital solution, star by star.

Three things are compared with the archive for each star: the RUWE; the part of the orbit that the five-parameter fit absorbs (the parallax, pmra and pmdec of gaia_source minus those of nss_two_body_orbit); and the formal errors.
The orbit factor B(G) is fitted on one seeded half of the stars of G below 13 and frozen; the scoreboard is computed on the other half and on every star of G 13 and above (B equals the single-star factor there). The statistics are the median M and the width W = (P84 - P16) / 2 of
x_ruwe    = ln(RUWE_archive / RUWE_predicted)
z_<p>     = (d_archive - d_predicted) / sigma_archive for p in parallax, pmra, pmdec, with d_archive the value of gaia_source minus that of nss_two_body_orbit and d_predicted the median over the noise draws of the fitted minus the true value
base_<p>  = d_archive / sigma_archive, the prediction of zero, to read z against
pull_<p>  = (d_archive - d_predicted) / ((P84 - P16) / 2 of the offsets over the noise draws): the offset in units of the model's own spread, 1 for a model that is right
error_<p> = ln(sigma_archive / sigma_predicted), sigma_predicted the median over the draws of the formal error of the fit (inflated by the raw chi-square, not scaled by B)
overall and in bins of G, period over the DR3 window, eccentricity, absolute ecliptic latitude and astrometric_n_good_obs_al, each with the standard deviation of the statistic over bootstrap resamples of the stars of the bin.

usage (the GPU: GAIAMOCK_DATA_DIR=... python validation/scoreboard.py simulate ROWS OUT --device cuda):
  simulate ROWS OUT [--limit N] [--draws 100] [--device cpu|cuda] [--g-max G] [--seed-offset N] [--window DR3|DR2] [--scans DIR]   simulate the stars of ROWS (a directory of chunk files, below; with --g-max only those brighter than G) that OUT does not hold yet, in part files of OUT
  score ROWS OUT [--json PATH] [--swapped] [--n-good-edges E0 E1 E2 E3 E4]                                     fit B on half A, freeze it, and write the scoreboard of half B (JSON) and the per-star values (OUT/stars.npz)
  rows SPECIAL.json SCOREBOARD.json OUT.json                       the rows of BH1, BH2 and BH3; SPECIAL.json holds {name: {gaia_source: [row], nss_two_body_orbit: [row]}} for BH1 and BH2
  compare OLD.npz NEW.npz                                          the improvement rule on two per-star files of the same stars
ROWS holds files *.json, each {"rows": [{...}]} with these columns of the archive (read from tap_schema.columns of the ESA Gaia archive): of gaiadr3.nss_two_body_orbit source_id, nss_solution_type as type, period, eccentricity as ecc,
t_periastron as t_peri, a/b/f/g_thiele_innes as ti_a/ti_b/ti_f/ti_g, parallax, pmra and pmdec as orb_parallax, orb_pmra, orb_pmdec; of gaiadr3.gaia_source for the same star ra, dec, phot_g_mean_mag as g, parallax, parallax_error, pmra, pmra_error,
pmdec, pmdec_error, nu_eff_used_in_astrometry as nu_eff, ruwe, astrometric_chi2_al as chi2, astrometric_n_good_obs_al as n_good, ecl_lat. The selection of the sample is on the standard Orbital and AstroSpectroSB1 solutions of
significance 10 or more with astrometric_params_solved 31 (the five-parameter solutions)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEED_HALVES = 1692  # the seed that splits the stars of G below 13 into the half that B is fitted on and the half that is scored
SEED_BOOTSTRAP = 1720
RESAMPLES = 200
MINIMUM_STARS = 100  # a bin with fewer stars is printed and is not used in a comparison
BRIGHT_LIMIT = 13.0  # the orbit factor is fitted below this G; above it B equals the single-star factor
WINDOW_DAYS = 2457902.0 - 2456891.5  # the DR3 window of wobbler.WINDOWS
G_EDGES = [9, 10, 11, 12, 13, 14, 15, 16.0001]
PERIOD_EDGES = [0, 0.5, 1, 3, np.inf]  # period over the DR3 window
ECC_EDGES = [0, 0.1, 0.3, 0.5, 0.7, 1.0001]
LATITUDE_EDGES = [0, 15, 30, 60, 90.0001]  # absolute ecliptic latitude
PARAMETERS = ("parallax", "pmra", "pmdec")
NUMERIC = ("period", "ecc", "t_peri", "ti_a", "ti_b", "ti_f", "ti_g", "orb_parallax", "orb_pmra", "orb_pmdec", "ra", "dec", "g", "parallax", "parallax_error", "pmra", "pmra_error", "pmdec", "pmdec_error", "nu_eff", "ruwe", "chi2", "n_good", "ecl_lat")
PER_STAR = ("w_raw", "w16", "w84", "u0_ratio", "d_parallax", "d_pmra", "d_pmdec", "s_parallax", "s_pmra", "s_pmdec", "o16_parallax", "o84_parallax", "o16_pmra", "o84_pmra", "o16_pmdec", "o84_pmdec", "n_ok")


def load_rows(directory):
    """The stars of the chunk files of `directory`, sorted by source_id, with the numeric columns as floats."""
    stars = {}
    for path in sorted(Path(directory).glob("*.json")):
        for row in json.loads(path.read_text())["rows"]:
            star = {key: (float(row[key]) if key in NUMERIC else row[key]) for key in row if key in NUMERIC or key == "type"}
            star["source_id"] = int(row["source_id"])  # an exact integer: ids are above 2^53, where a float64 merges neighbours
            stars[star["source_id"]] = star
    return [stars[key] for key in sorted(stars)]


def halves(g):
    """'A' or 'B' for each star (g in the order of the stars, sorted by source_id): the stars of G below 13 are permuted with default_rng(SEED_HALVES) and split at the middle, the first half A, which B(G) is fitted on, and the second B; every star of G 13 and above is B, since no fit uses it."""
    g = np.asarray(g, dtype=float)
    out = np.full(len(g), "B", dtype="<U1")
    index = np.flatnonzero(g < BRIGHT_LIMIT)
    order = np.random.default_rng(SEED_HALVES).permutation(len(index))
    out[index[order[: len(index) // 2]]] = "A"
    return out


def draws_of(wb, star, draws, device, seed_offset=0, window="DR3", scans=None):
    """The raw simulation of one star: its scan, its orbit from the Thiele-Innes constants, its own inclination and node, `draws` noise draws, the window `window` of wobbler.WINDOWS (DR3 or DR2). Returns the arrays over the draws that fit, or None."""
    scan = wb.Scan.from_position(star["ra"], star["dec"], device=device) if scans is None else scans[str(int(star["source_id"]))]  # scans: {target: Scan} of the exact positions (Scan.from_gost_batches), a star with no scan is an error and is counted as dropped
    elements = wb.campbell(star["ti_a"], star["ti_b"], star["ti_f"], star["ti_g"])
    incl, node, omega, a0 = (float(elements[key]) for key in ("inclination", "node", "omega", "a"))
    orbit = wb.Orbit(period_days=star["period"], ecc=star["ecc"], t_peri_days=star["t_peri"], omega=omega)
    fit = wb.simulate_fit(scan, orbit, a1_au=a0 * np.sin(incl) / star["orb_parallax"], parallax_mas=np.array([star["orb_parallax"]]), G=star["g"], n_orient=draws, inclination=incl, node=node,
                          windows={window: wb.WINDOWS[window]}, seed=(int(star["source_id"]) + seed_offset) % 2**31, device=device, nu_eff=star["nu_eff"], uwe_factor=1.0, formal_error_factor=1.0, parameters=True)
    fitted = {key: value[0].cpu().numpy() for key, value in fit.windows[window].items()}
    ok = fitted["ok"]
    if ok.sum() < draws // 2:
        return None
    return {key: value[ok] for key, value in fitted.items()}


def simulate_star(wb, star, draws, device, seed_offset=0, window="DR3", scans=None):
    """The PER_STAR values of one star (medians over the draws that fit), or None."""
    fitted = draws_of(wb, star, draws, device, seed_offset, window, scans)
    if fitted is None:
        return None
    uwe = fitted["uwe"]

    def median(key):
        return float(np.median(fitted[key]))

    offsets = {"parallax": fitted["parallax_mas"] - star["orb_parallax"], "pmra": fitted["pmra_mas_yr"], "pmdec": fitted["pmdec_mas_yr"]}
    spread = {f"o{q}_{p}": float(np.percentile(offsets[p], q)) for p in PARAMETERS for q in (16, 84)}

    return {"w_raw": float(np.median(uwe)), "w16": float(np.percentile(uwe, 16)), "w84": float(np.percentile(uwe, 84)), "u0_ratio": float(np.median(fitted["ruwe"] / uwe)),
            "d_parallax": median("parallax_mas") - star["orb_parallax"], "d_pmra": median("pmra_mas_yr"), "d_pmdec": median("pmdec_mas_yr"),
            "s_parallax": median("parallax_error_mas"), "s_pmra": median("pmra_error_mas_yr"), "s_pmdec": median("pmdec_error_mas_yr"), **spread, "n_ok": len(uwe)}


def simulate(rows, out, limit=0, draws=100, device="cpu", part_size=2000, seed_offset=0, g_max=None, window="DR3", scans=None):
    import wobbler as wb

    if scans is not None:
        scans = wb.Scan.from_gost_batches(scans, window=window, device="cpu")

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    stars = load_rows(rows)
    wanted = [s for s in stars if g_max is None or s["g"] < g_max]
    done = set()
    for path in out.glob("part-*.npz"):
        done |= set(np.load(path)["source_id"].tolist())
    todo = [s for s in wanted if s["source_id"] not in done]
    todo = todo[:limit] if limit else todo
    dropped = 0
    for start in range(0, len(todo), part_size):
        batch, results = todo[start:start + part_size], []
        for star in batch:
            try:
                results.append((star, simulate_star(wb, star, draws, device, seed_offset, window, scans)))
            except Exception as error:  # a failed star is counted, not hidden
                print("dropped", int(star["source_id"]), str(error)[:80], flush=True)
                results.append((star, None))
        kept = [(s, r) for s, r in results if r is not None]
        dropped += len(results) - len(kept)
        arrays = {"source_id": np.array([s["source_id"] for s, _ in kept], dtype=np.int64)} | {key: np.array([r[key] for _, r in kept]) for key in PER_STAR}
        np.savez(out / f"part-{len(list(out.glob('part-*.npz'))):04d}.npz", **arrays)
        print(f"{start + len(batch)} of {len(todo)}, dropped {dropped}", flush=True)
    return len(todo), dropped


def width(x):
    low, high = np.percentile(x, [16, 84])
    return float((high - low) / 2)


def summary(x, indices):
    """Median, width and their bootstrap standard deviations; `indices` is the (resamples, n) array of resampled positions."""
    sample = x[indices]
    medians, widths = np.median(sample, axis=1), (np.percentile(sample, 84, axis=1) - np.percentile(sample, 16, axis=1)) / 2
    return {"n": int(len(x)), "median": float(np.median(x)), "median_error": float(medians.std()), "width": width(x), "width_error": float(widths.std())}


def bins_of(stars, n_good_edges):
    """name -> list of (label, boolean mask) over the stars; `stars` holds g, period, ecc, ecl_lat, n_good."""
    out = {"G": [(f"{lo:g} to {min(hi, 16):g}", (stars["g"] >= lo) & (stars["g"] < hi)) for lo, hi in zip(G_EDGES[:-1], G_EDGES[1:])]}
    ratio = stars["period"] / WINDOW_DAYS
    out["period over window"] = [(f"{lo:g} to {hi:g}" if np.isfinite(hi) else f"{lo:g} and above", (ratio >= lo) & (ratio < hi)) for lo, hi in zip(PERIOD_EDGES[:-1], PERIOD_EDGES[1:])]
    out["eccentricity"] = [(f"{lo:g} to {min(hi, 1):g}", (stars["ecc"] >= lo) & (stars["ecc"] < hi)) for lo, hi in zip(ECC_EDGES[:-1], ECC_EDGES[1:])]
    out["absolute ecliptic latitude"] = [(f"{lo:g} to {min(hi, 90):g}", (np.abs(stars["ecl_lat"]) >= lo) & (np.abs(stars["ecl_lat"]) < hi)) for lo, hi in zip(LATITUDE_EDGES[:-1], LATITUDE_EDGES[1:])]
    out["good observations"] = [(f"{lo:g} to {hi:g}", (stars["n_good"] >= lo) & (stars["n_good"] < hi)) for lo, hi in zip(n_good_edges[:-1], n_good_edges[1:])]
    return out


def n_good_edges_of(n_good):
    """The quartiles of the archive's astrometric_n_good_obs_al over the sample, as the edges of four bins."""
    edges = np.percentile(n_good, [0, 25, 50, 75]).tolist()
    return [float(np.floor(e)) for e in edges] + [float(np.max(n_good)) + 1.0]


def quantities(stars, predicted):
    """The per-star statistics of the scoreboard: stars and predicted are dicts of equally long arrays."""
    out = {"ruwe": np.log(stars["ruwe"] / predicted["ruwe"])}
    for p in PARAMETERS:
        archive = stars[p] - stars["orb_" + p]
        out["offset_" + p] = (archive - predicted["d_" + p]) / stars[p + "_error"]
        out["zero_" + p] = archive / stars[p + "_error"]
        out["pull_" + p] = (archive - predicted["d_" + p]) / ((predicted["o84_" + p] - predicted["o16_" + p]) / 2)
        out["error_" + p] = np.log(stars[p + "_error"] / predicted["s_" + p])
    return out


def fit_orbit_table(stars, predicted, half, fit_half="A"):
    """B(G) of wobbler.uwe_factor.fit_orbit_factor from the stars of `fit_half` (A, or B for the swapped control) of G below 13: their G, raw simulated unit weight error and archive unit weight error."""
    from wobbler.uwe_factor import fit_orbit_factor

    m = (half == fit_half) & (stars["g"] < BRIGHT_LIMIT)
    return fit_orbit_factor(stars["g"][m], predicted["w_raw"][m], np.sqrt(stars["chi2"][m] / (stars["n_good"][m] - 5)))


def predicted_ruwe(stars, predicted, table):
    from wobbler.uwe_factor import calibrated_uwe

    return calibrated_uwe(stars["g"], predicted["w_raw"], bands=table) * predicted["u0_ratio"]


def join(rows, out):
    """The stars of `rows` that `out` simulated, as dicts of arrays: the archive columns and the per-star results."""
    stars = load_rows(rows)
    parts = [np.load(p) for p in sorted(Path(out).glob("part-*.npz"))]
    ids = np.concatenate([p["source_id"] for p in parts])
    predicted = {key: np.concatenate([p[key] for p in parts]) for key in PER_STAR}
    half_of = dict(zip((s["source_id"] for s in stars), halves([s["g"] for s in stars])))
    by_id = {s["source_id"]: s for s in stars}
    chosen = [by_id[i] for i in ids]
    table = {key: np.array([s[key] for s in chosen]) for key in NUMERIC} | {"source_id": ids.astype(np.int64)}
    table["type"] = np.array([s["type"] for s in chosen])
    return table, predicted, np.array([half_of[i] for i in ids])


def build(stars, predicted, half, swapped=False):
    """B fitted on half A, then the quantities of half B of G below 13 and of every star of G 13 and above: the table, the scored stars and their quantities. With `swapped` B is fitted on half B and half A is scored, for the stars of G below 13 (every star is simulated)."""
    table = fit_orbit_table(stars, predicted, half, "B" if swapped else "A")
    scored = (half == "A") & (stars["g"] < BRIGHT_LIMIT) if swapped else half == "B"
    chosen = {key: value[scored] for key, value in stars.items()}
    pred = {key: value[scored] for key, value in predicted.items()}
    pred["ruwe"] = predicted_ruwe(chosen, pred, table)
    values = quantities(chosen, pred)
    return table, chosen, values


def scoreboard(chosen, values, edges=None):
    rng = np.random.default_rng(SEED_BOOTSTRAP)
    edges = n_good_edges_of(chosen["n_good"]) if edges is None else [float(e) for e in edges]
    bins = bins_of(chosen, edges)
    out = {"n_good_edges": edges, "overall": {}, "bins": {name: {label: {} for label, _ in group} for name, group in bins.items()}}
    def resample(n):
        return rng.integers(0, n, size=(RESAMPLES, n))

    for quantity, x in values.items():
        out["overall"][quantity] = summary(x, resample(len(x)))
    for name, group in bins.items():
        for label, mask in group:
            n = int(mask.sum())
            index = resample(n) if n >= 2 else None
            for quantity, x in values.items():
                out["bins"][name][label][quantity] = summary(x[mask], index) if n >= MINIMUM_STARS else {"n": n}
    return out


def compare(old, new):
    """The improvement rule for each compared quantity (ruwe, the three offsets, the three errors) on two per-star files of the same stars: overall W falls by more than the standard error of the difference (paired bootstrap), and no bin
    with MINIMUM_STARS stars or more has W rise by more than its standard error; a bin where |M| rises by more than its standard error is a regression."""
    ids = np.intersect1d(old["source_id"], new["source_id"])
    a, b = (np.argsort(file["source_id"], kind="stable")[np.searchsorted(file["source_id"][np.argsort(file["source_id"], kind="stable")], ids)] for file in (old, new))  # the file of a whole run is two sorted blocks, not one
    rng = np.random.default_rng(SEED_BOOTSTRAP)
    index = rng.integers(0, len(ids), size=(RESAMPLES, len(ids)))
    stars = {key: old[key][a] for key in ("g", "period", "ecc", "ecl_lat", "n_good")}
    bins = bins_of(stars, [float(e) for e in old["n_good_edges"]])
    report = {}
    for quantity in (k[2:] for k in old.files if k.startswith("v_") and not k.startswith("v_zero_")):  # the zero prediction is a reference, not a version of the model
        x, y = old["v_" + quantity][a], new["v_" + quantity][b]
        row = {"overall": difference(x, y, index), "bins": {}}
        for name, group in bins.items():
            for label, mask in group:
                if mask.sum() >= MINIMUM_STARS:
                    row["bins"][f"{name}: {label}"] = difference(x[mask], y[mask], rng.integers(0, int(mask.sum()), size=(RESAMPLES, int(mask.sum()))))
        row["improved"] = row["overall"]["width_change"] < -row["overall"]["width_change_error"] and not any(
            bin_["width_change"] > bin_["width_change_error"] for bin_ in row["bins"].values())
        row["regressions"] = [label for label, bin_ in row["bins"].items() if abs(bin_["median_new"]) - abs(bin_["median_old"]) > bin_["median_abs_change_error"]]
        report[quantity] = row
    return report


def difference(x, y, index):
    """The change of W and of |M| from x to y over the same resamples of the stars."""
    wx = (np.percentile(x[index], 84, axis=1) - np.percentile(x[index], 16, axis=1)) / 2
    wy = (np.percentile(y[index], 84, axis=1) - np.percentile(y[index], 16, axis=1)) / 2
    mx, my = np.abs(np.median(x[index], axis=1)), np.abs(np.median(y[index], axis=1))
    return {"n": int(len(x)), "width_old": width(x), "width_new": width(y), "width_change": width(y) - width(x), "width_change_error": float((wy - wx).std()),
            "median_old": float(np.median(x)), "median_new": float(np.median(y)), "median_abs_change_error": float((my - mx).std())}


def percentiles(values):
    return [float(v) for v in np.percentile(values, [16, 50, 84])]


def star_row(wb, star, table, draws=1000, device="cpu"):
    """The row of one named star with an orbit solution: its archive values and the 16th, 50th and 84th percentile over `draws` noise draws of the prediction, with the frozen orbit factor `table`."""
    from wobbler.uwe_factor import calibrated_uwe

    window = draws_of(wb, star, draws, device)
    ruwe = calibrated_uwe(star["g"], window["uwe"], bands=table) * window["ruwe"] / window["uwe"]
    row = {"archive_ruwe": star["ruwe"], "ruwe_16_50_84": percentiles(ruwe), "ln_median_predicted_over_archive_ruwe": float(np.log(np.median(ruwe) / star["ruwe"])), "draws_that_fit": int(len(ruwe))}
    for p in PARAMETERS:
        value = window["parallax_mas"] - star["orb_parallax"] if p == "parallax" else window[p + "_mas_yr"]
        archive = star[p] - star["orb_" + p]
        row[p] = {"archive_offset": archive, "predicted_offset_16_50_84": percentiles(value), "archive_error": star[p + "_error"], "predicted_error": float(np.median(window[{"parallax": "parallax_error_mas"}.get(p, p + "_error_mas_yr")])),
                  "archive_over_predicted_offset": archive / float(np.median(value))}
    return row


def star_of(entry):
    """The star dict of a data/special.json entry: the archive's gaia_source row and its nss_two_body_orbit row, each a list of one row of strings."""
    gaia, orbit = entry["gaia_source"][0], entry["nss_two_body_orbit"][0]
    merged = {**gaia, **orbit}
    merged = {"g": gaia["g"], **{key: value for key, value in merged.items() if key != "g"}}
    star = {key: (float(value) if key in NUMERIC else value) for key, value in merged.items() if key in NUMERIC or key == "type"}
    star["source_id"] = int(merged["source_id"])
    return star


def named_rows(special_path, table, draws=1000, device="cpu"):
    """Rows for Gaia BH1 and BH2 (their DR3 orbital solutions), Gaia BH3 (the published combined orbit through validation/bh3.py; DR3 has no orbital solution of it)."""
    import wobbler as wb

    special = json.loads(Path(special_path).read_text())
    rows = {name: star_row(wb, star_of(special[name]), table, draws, device) for name in ("BH1", "BH2")}
    sys.path.insert(0, str(HERE))
    import bh3

    catalogue = {k: float(v) for k, v in json.loads((HERE / "bh3_catalogue.json").read_text()).items() if k not in ("read_on", "source_id") and v not in ("", None)}
    scan = wb.Scan.from_position(catalogue["ra"], catalogue["dec"], window="DR3", device="cpu")
    orbit = bh3.ORBITS["combined"]
    p = bh3.predict(scan, catalogue, orbit, "raw")
    from wobbler.uwe_factor import calibrated_uwe

    ruwe = calibrated_uwe(catalogue["phot_g_mean_mag"], p["raw_uwe"], bands=table) / float(wb.u0(catalogue["phot_g_mean_mag"], catalogue["nu_eff_used_in_astrometry"], device="cpu"))
    row = {"archive_ruwe": catalogue["ruwe"], "ruwe_16_50_84": percentiles(ruwe), "ln_median_predicted_over_archive_ruwe": float(np.log(np.median(ruwe) / catalogue["ruwe"])), "draws_that_fit": p["fits"],
           "orbit": "the combined solution of the published orbit (validation/bh3.py); DR3 has no nss_two_body_orbit row"}
    for key in PARAMETERS:
        archive = catalogue[key] - orbit[key][0]
        row[key] = {"archive_offset": archive, "predicted_offset_16_50_84": percentiles(p[key]), "archive_error": catalogue[key + "_error"], "archive_over_predicted_offset": archive / float(np.median(p[key]))}
    rows["BH3"] = row
    return rows


def score(rows, out, json_path=None, swapped=False, n_good_edges=None):
    stars, predicted, half = join(rows, out)
    table, chosen, values = build(stars, predicted, half, swapped)
    result = scoreboard(chosen, values, n_good_edges)
    result["orbit_factor_fitted_on_half_" + ("B" if swapped else "A")] = [float(b) for b in table]
    result["stars"] = {"fitting_half_below_G13": int(((half == ("B" if swapped else "A")) & (stars["g"] < BRIGHT_LIMIT)).sum()), "scored": int(len(chosen["g"])), "swapped": swapped}
    np.savez(Path(out) / ("stars_swapped.npz" if swapped else "stars.npz"), source_id=chosen["source_id"], n_good_edges=np.array(result["n_good_edges"]), **{key: chosen[key] for key in ("g", "period", "ecc", "ecl_lat", "n_good")},
             **{"v_" + key: value for key, value in values.items()})
    Path(json_path or Path(out) / ("scoreboard_swapped.json" if swapped else "scoreboard.json")).write_text(json.dumps(result, indent=1) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("simulate")
    one.add_argument("rows"), one.add_argument("out")
    one.add_argument("--limit", type=int, default=0), one.add_argument("--draws", type=int, default=100), one.add_argument("--device", default="cpu"), one.add_argument("--seed-offset", type=int, default=0), one.add_argument("--g-max", type=float, default=None), one.add_argument("--window", default="DR3", choices=("DR3", "DR2")), one.add_argument("--scans", default=None)
    two = sub.add_parser("score")
    two.add_argument("rows"), two.add_argument("out"), two.add_argument("--json"), two.add_argument("--swapped", action="store_true"), two.add_argument("--n-good-edges", type=float, nargs=5, default=None)
    rows = sub.add_parser("rows")
    rows.add_argument("special"), rows.add_argument("scoreboard"), rows.add_argument("json"), rows.add_argument("--draws", type=int, default=1000), rows.add_argument("--device", default="cpu")
    three = sub.add_parser("compare")
    three.add_argument("old"), three.add_argument("new")
    arguments = parser.parse_args()
    if arguments.command == "simulate":
        simulate(arguments.rows, arguments.out, arguments.limit, arguments.draws, arguments.device, seed_offset=arguments.seed_offset, g_max=arguments.g_max, window=arguments.window, scans=arguments.scans)
    elif arguments.command == "score":
        score(arguments.rows, arguments.out, arguments.json, arguments.swapped, arguments.n_good_edges)
    elif arguments.command == "rows":
        table = json.loads(Path(arguments.scoreboard).read_text())["orbit_factor_fitted_on_half_A"]
        Path(arguments.json).write_text(json.dumps(named_rows(arguments.special, table, arguments.draws, arguments.device), indent=1) + "\n")
    else:
        print(json.dumps(compare(np.load(arguments.old), np.load(arguments.new)), indent=1))


if __name__ == "__main__":
    sys.exit(main())
