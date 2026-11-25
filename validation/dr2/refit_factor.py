"""Refits the DR2 single-star factor f_DR2(G) in nodes that follow its changes. Input: the rows of the 41 bands of build_factor.py, as band_LOW_HIGH.csv files (fetch_bands.py), and factor_dr2_first_build.json (the bands and the UWE that wobbler simulates for a single star in each). The fit half of a band is its even rows, the held-out half its odd rows. A 0.5 mag band becomes its 0.1 mag nodes when some pair of its sub-bin medians (fit half) differs by more than 3 bootstrap standard deviations, or when a sub-bin median is more than 3 standard deviations from the line between the nodes just below and above the band; a node needs 50 rows. A node sits at the median G of its rows. The median UWE of the held-out rows over the node's factor is reported for every node of 200 or more held-out rows.
usage: python validation/dr2/refit_factor.py CACHE_DIRECTORY   (the band files; no network); rewrites src/wobbler/tables/uwe_factor_dr2.json and validation/dr2/factor_dr2_check.json, and writes factor_dr2_heldout.json. The bootstrap is seeded (SEED)."""
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TABLE = HERE.parents[1] / "src" / "wobbler" / "tables" / "uwe_factor_dr2.json"
SUB, SIGMAS, MINIMUM_ROWS, RESAMPLES, SEED = 0.1, 3.0, 50, 400, 826


def median_and_error(values, rng):
    if len(values) == 0:
        return float("nan"), float("nan")
    return float(np.median(values)), float(np.median(values[rng.integers(0, len(values), size=(RESAMPLES, len(values)))], axis=1).std())


def sub_bins(g, uwe, low, high, rng):
    """The 0.1 mag sub-bins of a band from its fit-half stars: G range, median G, rows, median UWE and its bootstrap error."""
    count = int(round((high - low) / SUB))
    index = np.clip(np.floor((g - low) / SUB + 1e-9).astype(int), 0, max(count - 1, 0))
    subs = []
    for k in range(count):
        chosen = uwe[index == k]
        median, error = median_and_error(chosen, rng)
        subs.append({"G": [round(low + SUB * k, 2), round(low + SUB * (k + 1), 2)], "G_node": float(np.median(g[index == k])) if len(chosen) else round(low + SUB * (k + 0.5), 3), "n_fit": int(len(chosen)), "median_uwe": median, "error": error})
    return subs


def pair_split(subs):
    """Some pair of sub-bin medians differs by more than SIGMAS standard errors, every sub-bin having MINIMUM_ROWS rows."""
    return len(subs) > 1 and all(s["n_fit"] >= MINIMUM_ROWS for s in subs) and any(
        abs(a["median_uwe"] - b["median_uwe"]) > SIGMAS * np.hypot(a["error"], b["error"]) for i, a in enumerate(subs) for b in subs[i + 1:])


def misfit_split(subs, lower, upper):
    """Some sub-bin median differs from the linear interpolation, at the sub-bin's median G, between the nodes (G_node, median_uwe) just below and just above the band by more than SIGMAS of the sub-bin's error (a step at a band edge). No neighbour on a side: no test."""
    if lower is None or upper is None or len(subs) < 2 or not all(s["n_fit"] >= MINIMUM_ROWS for s in subs):
        return False
    return any(abs(s["median_uwe"] - np.interp(s["G_node"], [lower["G_node"], upper["G_node"]], [lower["median_uwe"], upper["median_uwe"]])) > SIGMAS * s["error"] for s in subs)


def nodes_of_band(g, uwe, low, high, rng):
    """The nodes of one 0.5 mag band from its fit-half stars: the band itself, or its 0.1 mag sub-bins when some pair of sub-bin medians differs (pair_split)."""
    subs = sub_bins(g, uwe, low, high, rng)
    if pair_split(subs):
        return subs
    median, error = median_and_error(uwe, rng)
    return [{"G": [low, high], "G_node": float(np.median(g)) if len(g) else 0.5 * (low + high), "n_fit": int(len(uwe)), "median_uwe": median, "error": error}]


def sliver_sampled(g, low, high):
    """True if the 5th to 95th percentile of the rows' G spans under 80 percent of the band, that is, the rows cover only a part of it."""
    return bool(len(g) == 0 or (np.percentile(g, 95) - np.percentile(g, 5)) < 0.8 * (high - low))


def read_band(directory, low, high):
    rows = list(csv.DictReader(io.StringIO((Path(directory) / f"band_{low}_{high}.csv").read_text())))
    g = np.array([float(r["phot_g_mean_mag"]) for r in rows])
    uwe = np.sqrt(np.array([float(r["astrometric_chi2_al"]) for r in rows]) / (np.array([float(r["astrometric_n_good_obs_al"]) for r in rows]) - 5.0))
    return g, uwe


def main(directory):
    old = json.loads((HERE / "factor_dr2_first_build.json").read_text())["bands"]  # the 41 bands of build_factor.py and their simulated single-star UWE; the refit never reads its own output
    old_centres = np.array([0.5 * (b["G"][0] + b["G"][1]) for b in old])
    old_simulated = np.array([b["simulated_single_star_uwe"] for b in old])
    rng = np.random.default_rng(SEED)
    nodes, held, fits, sliver = [], {}, {}, {}
    for band in old:
        low, high = band["G"]
        g, uwe = read_band(directory, low, high)
        fit = np.arange(len(g)) % 2 == 0
        held[(low, high)], fits[(low, high)] = (g[~fit], uwe[~fit]), (g[fit], uwe[fit])
        sliver[(low, high)] = sliver_sampled(g, low, high)
        for node in nodes_of_band(g[fit], uwe[fit], low, high, rng):
            node["parent"] = [low, high]
            nodes.append(node)
    first = list(nodes)
    nodes = []
    for node in first:  # a band kept whole whose sub-bins do not follow the interpolation between its neighbours is cut
        if node["G"] != node["parent"] or node["G"][1] - node["G"][0] <= SUB + 1e-9:
            nodes.append(node)
            continue
        position = first.index(node)
        below = first[position - 1] if position > 0 else None
        above = first[position + 1] if position + 1 < len(first) else None
        subs = sub_bins(*fits[tuple(node["parent"])], node["G"][0], node["G"][1], rng)
        nodes += [dict(sub, parent=node["parent"]) for sub in subs] if misfit_split(subs, below, above) else [node]
    for node in nodes:
        node["simulated_single_star_uwe"] = float(np.interp(node["G_node"], old_centres, old_simulated))
        node["f"] = node["median_uwe"] / node["simulated_single_star_uwe"]
    centres = np.array([n["G_node"] for n in nodes])
    values = np.array([n["f"] for n in nodes])
    for node in nodes:
        g, uwe = held[tuple(node["parent"])]
        inside = (g >= node["G"][0]) & (g < node["G"][1]) if node["G"] != node["parent"] else np.ones(len(g), bool)
        ratio = uwe[inside] / (np.interp(g[inside], centres, values) * np.interp(g[inside], old_centres, old_simulated))
        node["n_heldout"] = int(inside.sum())
        node["median_ratio_heldout"] = float(np.median(ratio)) if inside.any() else float("nan")
    bands = [{"G": n["G"], "G_node": n["G_node"], "sliver_sampled": sliver[tuple(n["parent"])], "f": n["f"], "n_archive": n["n_fit"] + n["n_heldout"], "n_fit": n["n_fit"], "n_heldout": n["n_heldout"], "median_uwe_fit": n["median_uwe"],
              "simulated_single_star_uwe": n["simulated_single_star_uwe"], "median_ratio_heldout": n["median_ratio_heldout"]} for n in nodes]
    meta = json.loads(TABLE.read_text())
    meta["sample"] += "; nodes: a 0.5 mag band is cut into its 0.1 mag sub-bins when some pair of sub-bin medians of the fit half differs by more than 3 bootstrap standard deviations (validation/dr2/refit_factor.py), the node is placed at the median G of its fit-half rows (G_node), because the rows of a band with more than 5,000 stars cover only its lowest part, and the simulated single-star UWE of a node is the one of the 0.5 mag bands interpolated at that position"
    meta["bands_sha256"] = hashlib.sha256(json.dumps(bands, sort_keys=True).encode()).hexdigest()
    meta["bands"] = [{k: b[k] for k in ("G", "G_node", "sliver_sampled", "f", "n_archive")} for b in bands]
    TABLE.write_text(json.dumps(meta, indent=1) + "\n")
    (HERE / "factor_dr2_check.json").write_text(json.dumps({"bands": bands}, indent=1) + "\n")
    tested = [b for b in bands if b["n_heldout"] >= 200]
    worst = max(tested, key=lambda b: abs(b["median_ratio_heldout"] - 1))
    result = {"nodes": len(bands), "split_bands": sorted({tuple(n["parent"]) for n in nodes if n["G"] != n["parent"]}), "tested": len(tested), "worst_node": worst["G"], "worst_median_ratio": worst["median_ratio_heldout"],
              "all_within_5_percent": all(abs(b["median_ratio_heldout"] - 1) <= 0.05 for b in tested), "g_13_to_14": [{k: b[k] for k in ("G", "G_node", "f", "n_heldout", "median_ratio_heldout")} for b in bands if 13.0 <= b["G"][0] < 14.0],
              "largest_gaps_g_12_to_15": sorted(([round(a, 3), round(b, 3)] for a, b in zip(centres[:-1], centres[1:]) if 12.0 <= a < 15.0), key=lambda x: x[0] - x[1])[:4]}
    (HERE / "factor_dr2_heldout.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "g_13_to_14"}))


if __name__ == "__main__":
    main(sys.argv[1])
