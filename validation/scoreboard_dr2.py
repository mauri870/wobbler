"""Accuracy scoreboard of the DR2 window: wobbler's DR2 window against Gaia DR2's five-parameter solutions of the stars of validation/scoreboard.py.

The DR2 solution of each star is the candidate of gaiadr3.dr2_neighbourhood with the smallest angular_distance (the documentation of the table gives no rule and has no rank column; the choice is this script's), kept when its gaiadr2.gaia_source row has astrometric_params_solved 31.
The simulation is that of scoreboard.py in the window WINDOWS["DR2"] (`scoreboard.py simulate ... --window DR2`), with wobbler's DR3 error model, so the result shows what wobbler does today for DR2. B(G) and f(G) are those of the DR3 scoreboard, fitted on half A of G below 13 and frozen.
The quantities are those of scoreboard.py with the DR2 archive values: ruwe holds ln(UWE_DR2 / UWE_predicted) with UWE_DR2 = sqrt(astrometric_chi2_al / (astrometric_n_good_obs_al - 5)) of the DR2 row (DR2's RUWE needs DR2's own normalisation); offset_, zero_, pull_ and error_ are the DR2 parallax and proper motion minus the DR3 orbit solution's, over the
DR2 formal error and the draws' spread, and the ln of the DR2 formal error over the predicted one. The bins are those of scoreboard.py, on the star's DR3 columns. The DR2 RUWE of gaiadr2.ruwe is reported beside it as ln(RUWE_DR2 / (UWE_predicted / u0_DR3)), information only.

usage:
  score ROWS DR3_OUT DR2_OUT DR2_ROWS --n-good-edges E0 E1 E2 E3 E4 [--json PATH] [--factors dr3|dr2] [--orbit dr3|dr2|dr2-faint] [--swapped]
ROWS and DR3_OUT as for scoreboard.py (DR3_OUT the simulation of the DR3 window, which fixes B), DR2_OUT the simulation in the DR2 window, DR2_ROWS a directory of files *.json, each {"neighbours": [...], "dr2": [...]}: neighbours with dr3_source_id, dr2_source_id and angular_distance; dr2 with dr2_source_id, parallax,
parallax_error, pmra, pmra_error, pmdec, pmdec_error, chi2, n_good, params_solved and dr2_ruwe (the columns of gaiadr2.gaia_source and gaiadr2.ruwe)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scoreboard as sb  # noqa: E402

DR2_NUMERIC = ("angular_distance", "parallax", "parallax_error", "pmra", "pmra_error", "pmdec", "pmdec_error", "chi2", "n_good", "dr2_ruwe")
FIVE_PARAMETER = 31
FAINT_EDGES = tuple(13.0 + 0.25 * k for k in range(9))  # the bands of the faint orbit factor: G 13 to 15 in steps of 0.25 mag
FAINT_LIMIT = FAINT_EDGES[-1]
SUFFIX = {"dr3": "", "dr2": "_b2", "dr2-faint": "_b3"}


def as_float(value):
    return float(value) if value not in ("", None) else float("nan")


def candidates_by_star(directory):
    """dr3_source_id -> list of the candidates (the neighbour row merged with its DR2 row), from the batch files of `directory`."""
    found = {}
    for path in sorted(Path(directory).glob("g*.json")):
        batch = json.loads(path.read_text())
        rows = {int(r["dr2_source_id"]): r for r in batch["dr2"]}
        for n in batch["neighbours"]:
            row = rows[int(n["dr2_source_id"])]
            merged = {"dr2_source_id": str(int(n["dr2_source_id"])), "params_solved": int(float(row["params_solved"])) if row["params_solved"] not in ("", None) else -1}
            merged.update({key: as_float(n[key]) if key in n else as_float(row.get(key)) for key in DR2_NUMERIC})
            found.setdefault(int(n["dr3_source_id"]), []).append(merged)
    return found


def best_neighbours(directory, ids):
    """The nearest candidate of each of `ids` that has a five-parameter DR2 solution, and the counts. A star whose nearest candidate is not a five-parameter solution is dropped and counted, not given its next candidate."""
    found = candidates_by_star(directory)
    chosen, counts = {}, {"stars": len(ids), "with_a_candidate": 0, "with_exactly_one": 0, "with_several": 0, "nearest_is_the_same_id_of_several": 0, "no_candidate": 0, "not_five_parameter": 0, "kept": 0}
    for star in ids:
        star = int(star)
        candidates = found.get(star, [])
        if not candidates:
            counts["no_candidate"] += 1
            continue
        counts["with_a_candidate"] += 1
        nearest = min(candidates, key=lambda c: c["angular_distance"])
        if len(candidates) == 1:
            counts["with_exactly_one"] += 1
        else:
            counts["with_several"] += 1
            counts["nearest_is_the_same_id_of_several"] += int(int(nearest["dr2_source_id"]) == star)
        if nearest["params_solved"] != FIVE_PARAMETER:
            counts["not_five_parameter"] += 1
            continue
        chosen[star] = nearest
        counts["kept"] += 1
    return chosen, counts


def predicted_uwe(g, w_raw, table, factors="dr3", faint=None):
    """UWE_predicted = sqrt(f(G)^2 + B(G)^2 (w^2 - 1)) with the single-star factor of DR3 (factors="dr3", wobbler.uwe_factor.calibrated_uwe) or of DR2 (factors="dr2", uwe_factor_dr2) and the DR3 orbit table B frozen on half A; beyond the table's last edge B equals the single-star factor, as in DR3."""
    import importlib

    uf = importlib.import_module("wobbler.uwe_factor")  # the module: the package's attribute of that name is the function

    if factors == "dr3":
        return uf.calibrated_uwe(g, w_raw, bands=table)
    if factors != "dr2":
        raise ValueError(f"factors must be 'dr3' or 'dr2', not {factors!r}")
    g, w = np.broadcast_arrays(np.asarray(g, dtype=float), np.asarray(w_raw, dtype=float))
    f = uf.uwe_factor_dr2(g)
    b = np.where(g >= uf._orbit_bands()[0][-1], f, uf._lookup(g, table))
    if faint is not None:  # B_DR2 of G 13 to 15, one value per band of FAINT_EDGES; B equals f from G 15
        index = np.clip(np.searchsorted(FAINT_EDGES, g, side="right") - 1, 0, len(FAINT_EDGES) - 2)
        b = np.where((g >= FAINT_EDGES[0]) & (g < FAINT_LIMIT), np.asarray(faint, dtype=float)[index], b)
    return np.sqrt(np.maximum(f**2 + b**2 * (w**2 - 1.0), 0.0))


def faint_halves(g, half):
    """`half` with the stars of G 13 to 15 split in two: they are permuted with default_rng(SEED_HALVES) and the first half is A, the rest B (the scoreboard labels every star of G 13 and above B). Stars of G 15 and above stay B, the stars of G below 13 keep their label."""
    g = np.asarray(g, dtype=float)
    out = np.array(half, copy=True)
    index = np.flatnonzero((g >= FAINT_EDGES[0]) & (g < FAINT_LIMIT))
    order = np.random.default_rng(sb.SEED_HALVES).permutation(len(index))
    out[index] = "B"
    out[index[order[: len(index) // 2]]] = "A"
    return out


def fit_orbit_table_dr2(g, w_raw, uwe_archive, single=None, minimum_uwe=1.5, edges=None):
    """B_DR2(G): the orbit factor of each band of wobbler.uwe_factor.ORBIT_EDGES fitted on DR2's unit weight errors, the method of wobbler.uwe_factor.fit_orbit_factor with the single-star factor `single` (default uwe_factor_dr2, DR2's) in place of DR3's: the square root of the median, over the stars of the band with a simulated unit weight error of `minimum_uwe` or more, of max(0, (UWE_DR2^2 - f^2) / (w^2 - 1))."""
    import importlib

    uf = importlib.import_module("wobbler.uwe_factor")
    single = uf.uwe_factor_dr2 if single is None else single
    g, w, archive = (np.asarray(x, dtype=float) for x in (g, w_raw, uwe_archive))
    f = single(g)
    out = []
    edges = uf.ORBIT_EDGES if edges is None else edges
    for low, high in zip(edges[:-1], edges[1:]):
        m = (g >= low) & (g < high) & (w >= minimum_uwe)
        out.append(float(np.sqrt(np.median(np.maximum(0.0, (archive[m] ** 2 - f[m] ** 2) / (w[m] ** 2 - 1.0))))) if m.any() else float("nan"))
    return np.array(out)


def score(rows, dr3_out, dr2_out, dr2_rows, n_good_edges, json_path=None, factors="dr3", orbit="dr3", swapped=False):
    fit_half, score_half = ("B", "A") if swapped else ("A", "B")
    stars3, predicted3, half3 = sb.join(rows, dr3_out)
    table = sb.fit_orbit_table(stars3, predicted3, half3, fit_half)  # DR3's B, fitted on the fit half (the swapped control fits on half B)
    stars, predicted, half = sb.join(rows, dr2_out)
    faint = None
    if orbit == "dr2-faint":
        half = faint_halves(stars["g"], half)
    scored = np.flatnonzero(half == score_half)
    if orbit in ("dr2", "dr2-faint"):  # B_DR2 fitted on the DR2 solutions of the fit half of G below 13 (and of G 13 to 15 for dr2-faint), matched by the same rule (the nearest candidate, five-parameter)
        fit_index = np.flatnonzero((half == fit_half) & (stars["g"] < (FAINT_LIMIT if orbit == "dr2-faint" else sb.BRIGHT_LIMIT)))
        fit_rows, fit_counts = best_neighbours(dr2_rows, stars["source_id"][fit_index])
        fit_keep = np.array([k for k in fit_index if int(stars["source_id"][k]) in fit_rows], dtype=int)
        fit_uwe = np.sqrt(np.array([fit_rows[int(stars["source_id"][k])]["chi2"] / (fit_rows[int(stars["source_id"][k])]["n_good"] - 5) for k in fit_keep]))
        bright = stars["g"][fit_keep] < sb.BRIGHT_LIMIT
        table = fit_orbit_table_dr2(stars["g"][fit_keep][bright], predicted["w_raw"][fit_keep][bright], fit_uwe[bright])
        if orbit == "dr2-faint":
            faint = fit_orbit_table_dr2(stars["g"][fit_keep][~bright], predicted["w_raw"][fit_keep][~bright], fit_uwe[~bright], edges=FAINT_EDGES)
    chosen_rows, counts = best_neighbours(dr2_rows, stars["source_id"][scored])
    keep = np.array([k for k in scored if int(stars["source_id"][k]) in chosen_rows], dtype=int)
    candidates = [chosen_rows[int(stars["source_id"][k])] for k in keep]
    chosen = {key: stars[key][keep] for key in ("source_id", "g", "period", "ecc", "ecl_lat", "n_good", "orb_parallax", "orb_pmra", "orb_pmdec")}
    for key in ("parallax", "parallax_error", "pmra", "pmra_error", "pmdec", "pmdec_error"):
        chosen[key] = np.array([c[key] for c in candidates])
    uwe_archive = np.sqrt(np.array([c["chi2"] / (c["n_good"] - 5) for c in candidates]))
    chosen["ruwe"] = uwe_archive
    pred = {key: value[keep] for key, value in predicted.items()}
    uwe_predicted = predicted_uwe(chosen["g"], pred["w_raw"], table, factors, faint)
    pred["ruwe"] = uwe_predicted
    values = sb.quantities(chosen, pred)
    result = sb.scoreboard(chosen, values, n_good_edges)
    catalogue = np.log(np.array([c["dr2_ruwe"] for c in candidates]) / (uwe_predicted * pred["u0_ratio"]))
    finite = np.isfinite(catalogue)
    result["information_ruwe_of_gaiadr2_ruwe"] = {"stars": int(finite.sum()), "median": float(np.median(catalogue[finite])), "width": sb.width(catalogue[finite])}
    result["matching"] = counts
    result["scored_stars"] = int(len(keep))
    result["factors"] = factors
    result["orbit"] = orbit
    result["swapped"] = swapped
    if faint is not None:
        result["orbit_factor_faint_used"] = [float(b) for b in faint]
    if orbit in ("dr2", "dr2-faint"):
        result["fit_matching"] = fit_counts
        result["fit_stars"] = int(len(fit_keep))
    result["quantity_ruwe"] = "ln(UWE_DR2 / UWE_predicted), UWE_DR2 = sqrt(astrometric_chi2_al / (astrometric_n_good_obs_al - 5)) of the DR2 row; the other quantities as in scoreboard.py with the DR2 values"
    result["orbit_factor_used"] = [float(b) for b in table]
    np.savez(Path(dr2_out) / f"stars_dr2_{factors}{SUFFIX[orbit]}{'_swapped' if swapped else ''}.npz", source_id=chosen["source_id"], n_good_edges=np.array(result["n_good_edges"]), **{key: chosen[key] for key in ("g", "period", "ecc", "ecl_lat", "n_good")}, **{"v_" + key: value for key, value in values.items()})
    Path(json_path or Path(dr2_out) / f"scoreboard_dr2_{factors}{SUFFIX[orbit]}{'_swapped' if swapped else ''}.json").write_text(json.dumps(result, indent=1) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("score")
    one.add_argument("rows"), one.add_argument("dr3_out"), one.add_argument("dr2_out"), one.add_argument("dr2_rows")
    one.add_argument("--n-good-edges", type=float, nargs=5, required=True), one.add_argument("--json"), one.add_argument("--factors", default="dr3", choices=("dr3", "dr2")), one.add_argument("--orbit", default="dr3", choices=("dr3", "dr2", "dr2-faint")), one.add_argument("--swapped", action="store_true")
    arguments = parser.parse_args()
    score(arguments.rows, arguments.dr3_out, arguments.dr2_out, arguments.dr2_rows, arguments.n_good_edges, arguments.json, arguments.factors, arguments.orbit, arguments.swapped)


if __name__ == "__main__":
    sys.exit(main())
