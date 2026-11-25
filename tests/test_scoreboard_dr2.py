"""validation/scoreboard_dr2.py: the choice of the DR2 neighbour, the counts, and the scoring of the DR2 window against DR2's five-parameter solutions, on synthetic stars."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))
import scoreboard as sb  # noqa: E402
import scoreboard_dr2 as sd  # noqa: E402
from test_scoreboard import synthetic_rows, synthetic_scan  # noqa: E402,F401  (a fixture)


def neighbour(dr3, dr2, distance, params=31, **extra):
    row = {"dr3_source_id": str(dr3), "dr2_source_id": str(dr2), "angular_distance": str(distance), "magnitude_difference": "0.01", "pm_propagation": "true"}
    return row, {"dr2_source_id": str(dr2), "parallax": "2.0", "parallax_error": "0.1", "pmra": "1.0", "pmra_error": "0.1", "pmdec": "2.0", "pmdec_error": "0.1", "chi2": "500", "n_good": "100", "params_solved": str(params),
                 "dr2_ruwe": "1.5", **extra}


def write_batch(directory, name, candidates):
    Path(directory).mkdir(parents=True, exist_ok=True)
    neighbours = [n for n, _ in candidates]
    rows = {r["dr2_source_id"]: r for _, r in candidates}
    (Path(directory) / name).write_text(json.dumps({"dr3_ids": sorted({int(n["dr3_source_id"]) for n in neighbours}), "neighbours": neighbours, "dr2": list(rows.values())}))


def test_the_nearest_candidate_is_chosen_and_the_counts_are_stated(tmp_path):
    big = 4373465352415301632  # above 2^53: ids must stay exact
    write_batch(tmp_path, "g9_0.json", [neighbour(big, big, 150.0), neighbour(big + 1, big + 1, 20.0), neighbour(big + 1, big + 5, 30.0),  # one candidate; two, the nearest has the same id
                                        neighbour(big + 2, big + 9, 10.0), neighbour(big + 2, big + 2, 40.0),  # two candidates, the nearest has another id
                                        neighbour(big + 3, big + 3, 10.0, params=3)])  # a two-parameter DR2 solution
    chosen, counts = sd.best_neighbours(tmp_path, [big, big + 1, big + 2, big + 3, big + 4])
    assert {k: v["dr2_source_id"] for k, v in chosen.items()} == {big: str(big), big + 1: str(big + 1), big + 2: str(big + 9)}
    assert counts == {"stars": 5, "with_a_candidate": 4, "with_exactly_one": 2, "with_several": 2, "nearest_is_the_same_id_of_several": 1, "no_candidate": 1, "not_five_parameter": 1, "kept": 3}


def test_a_star_with_several_candidates_where_the_nearest_is_not_five_parameter_is_dropped_not_replaced(tmp_path):
    write_batch(tmp_path, "g9_0.json", [neighbour(1, 1, 50.0), neighbour(1, 2, 10.0, params=3)])
    chosen, counts = sd.best_neighbours(tmp_path, [1])
    assert chosen == {} and counts["not_five_parameter"] == 1 and counts["kept"] == 0


def test_a_missing_ruwe_is_nan_not_a_failure(tmp_path):
    n, r = neighbour(1, 1, 5.0)
    r["dr2_ruwe"] = ""
    write_batch(tmp_path, "g9_0.json", [(n, r)])
    chosen, _ = sd.best_neighbours(tmp_path, [1])
    assert np.isnan(chosen[1]["dr2_ruwe"]) and chosen[1]["parallax"] == 2.0


@pytest.mark.torch_cpu
def test_the_dr2_window_simulation_has_fewer_epochs_than_the_dr3_window_and_scores_end_to_end(tmp_path, synthetic_scan):  # noqa: F811
    synthetic_rows(tmp_path / "rows", n=80, seed=21)
    sb.simulate(tmp_path / "rows", tmp_path / "dr3", draws=12)
    sb.simulate(tmp_path / "rows", tmp_path / "dr2", draws=12, window="DR2")
    a, b = (np.load(sorted((tmp_path / name).glob("part-*.npz"))[0]) for name in ("dr3", "dr2"))
    assert len(b["source_id"]) > 0 and not np.array_equal(a["w_raw"][: len(b["w_raw"])], b["w_raw"]), "the DR2 window is another window, so another unit weight error"
    stars = sb.load_rows(tmp_path / "rows")
    candidates = []
    for star in stars:
        candidates.append(neighbour(star["source_id"], star["source_id"], 5.0))
    write_batch(tmp_path / "dr2cache", "g9_0.json", candidates)
    result = sd.score(tmp_path / "rows", tmp_path / "dr3", tmp_path / "dr2", tmp_path / "dr2cache", n_good_edges=[100.0, 150.0, 200.0, 250.0, 400.0], json_path=tmp_path / "board.json")
    assert result["matching"]["stars"] > 0 and result["matching"]["kept"] == result["matching"]["with_a_candidate"] - result["matching"]["not_five_parameter"]
    assert set(result["overall"]) >= {"ruwe", "offset_parallax", "pull_pmra", "error_pmdec"} and (tmp_path / "board.json").exists()
    assert "ln(UWE_DR2" in result["quantity_ruwe"]


def test_the_dr2_factors_change_the_ruwe_quantity_and_nothing_else():
    import importlib

    uf = importlib.import_module("wobbler.uwe_factor")

    table = [1.5] * (len(uf.ORBIT_EDGES) - 1)
    g = np.array([11.5, 11.5, 12.9, 13.2, 14.7])
    w = np.array([1.0, 6.0, 3.0, 1.0, 1.0])
    dr3, dr2 = sd.predicted_uwe(g, w, table, "dr3"), sd.predicted_uwe(g, w, table, "dr2")
    assert dr2[0] == pytest.approx(float(uf.uwe_factor_dr2(11.5))), "w = 1 leaves the single-star factor"
    assert dr3[0] == pytest.approx(float(uf.uwe_factor(11.5))) and dr2[0] < dr3[0], "DR2's predicted UWE at G 11.5 and w = 1 is under DR3's"
    assert dr2[1] == pytest.approx(np.sqrt(float(uf.uwe_factor_dr2(11.5)) ** 2 + 1.5**2 * 35.0)), "inside the table the orbit factor B is DR3's"
    assert dr2[3] == pytest.approx(float(uf.uwe_factor_dr2(13.2))) and dr2[4] == pytest.approx(float(uf.uwe_factor_dr2(14.7))), "beyond the table B is the single-star factor, so w = 1 gives f"
    assert float(uf.uwe_factor_dr2(11.5)) < float(uf.uwe_factor(11.5)), "DR2's factor at G 11.5 is under DR3's"
    with pytest.raises(ValueError):
        sd.predicted_uwe(g, w, table, "dr1")


@pytest.mark.torch_cpu
def test_scoring_with_dr2_factors_keeps_the_offsets_and_errors_and_changes_the_ruwe(tmp_path, synthetic_scan):  # noqa: F811
    synthetic_rows(tmp_path / "rows", n=80, seed=21)
    sb.simulate(tmp_path / "rows", tmp_path / "dr3", draws=12)
    sb.simulate(tmp_path / "rows", tmp_path / "dr2", draws=12, window="DR2")
    stars = sb.load_rows(tmp_path / "rows")
    write_batch(tmp_path / "dr2cache", "g9_0.json", [neighbour(star["source_id"], star["source_id"], 5.0) for star in stars])
    edges = [100.0, 150.0, 200.0, 250.0, 400.0]
    a = sd.score(tmp_path / "rows", tmp_path / "dr3", tmp_path / "dr2", tmp_path / "dr2cache", edges, json_path=tmp_path / "a.json", factors="dr3")
    b = sd.score(tmp_path / "rows", tmp_path / "dr3", tmp_path / "dr2", tmp_path / "dr2cache", edges, json_path=tmp_path / "b.json", factors="dr2")
    fa, fb = np.load(tmp_path / "dr2" / "stars_dr2_dr3.npz"), np.load(tmp_path / "dr2" / "stars_dr2_dr2.npz")
    assert a["factors"] == "dr3" and b["factors"] == "dr2" and np.array_equal(fa["source_id"], fb["source_id"])
    for key in fa.files:
        if key.startswith("v_") and key != "v_ruwe":
            assert np.array_equal(fa[key], fb[key], equal_nan=True), key
    assert not np.array_equal(fa["v_ruwe"], fb["v_ruwe"])


def test_the_dr2_orbit_fit_equals_fit_orbit_factor_when_given_dr3_single_star_factor():
    import importlib

    uf = importlib.import_module("wobbler.uwe_factor")
    rng = np.random.default_rng(3)
    g = rng.uniform(9.0, 13.0, 3000)
    w = rng.uniform(1.0, 8.0, 3000)
    archive = np.sqrt(uf.uwe_factor(g) ** 2 + 0.9**2 * (w**2 - 1.0)) * rng.lognormal(0.0, 0.05, 3000)
    assert np.allclose(sd.fit_orbit_table_dr2(g, w, archive, single=uf.uwe_factor), uf.fit_orbit_factor(g, w, archive), rtol=0, atol=1e-12)
    assert np.allclose(sd.fit_orbit_table_dr2(g, w, archive, single=uf.uwe_factor), 0.9, atol=0.08), "the fit recovers the factor that made the archive values"


def test_a_band_without_a_star_of_unit_weight_error_1_5_has_no_orbit_factor():
    g, w, archive = np.array([9.1, 9.2]), np.array([1.0, 1.2]), np.array([2.0, 2.1])
    table = sd.fit_orbit_table_dr2(g, w, archive)
    assert np.isnan(table[0]) and len(table) == 16


@pytest.mark.torch_cpu
def test_scoring_with_the_dr2_orbit_factor_and_its_swap_runs_end_to_end(tmp_path, synthetic_scan):  # noqa: F811
    synthetic_rows(tmp_path / "rows", n=80, seed=21)
    sb.simulate(tmp_path / "rows", tmp_path / "dr3", draws=12)
    sb.simulate(tmp_path / "rows", tmp_path / "dr2", draws=12, window="DR2")
    stars = sb.load_rows(tmp_path / "rows")
    write_batch(tmp_path / "dr2cache", "g9_0.json", [neighbour(star["source_id"], star["source_id"], 5.0) for star in stars])
    edges = [100.0, 150.0, 200.0, 250.0, 400.0]
    for swapped in (False, True):
        r = sd.score(tmp_path / "rows", tmp_path / "dr3", tmp_path / "dr2", tmp_path / "dr2cache", edges, json_path=tmp_path / f"b{swapped}.json", factors="dr2", orbit="dr2", swapped=swapped)
        assert r["orbit"] == "dr2" and r["swapped"] is swapped and r["fit_stars"] > 0 and len(r["orbit_factor_used"]) == 16
    assert (tmp_path / "dr2" / "stars_dr2_dr2_b2.npz").exists() and (tmp_path / "dr2" / "stars_dr2_dr2_b2_swapped.npz").exists()


def test_the_faint_halves_split_g_13_to_15_in_two_and_leave_the_rest_alone():
    g = np.array([10.0, 12.9, 13.0, 13.5, 14.9, 15.0, 16.0] * 40)
    half = np.array(["A", "B", "B", "B", "B", "B", "B"] * 40)
    out = sd.faint_halves(g, half)
    inside = (g >= 13) & (g < 15)
    assert (out[~inside] == half[~inside]).all()
    assert (out[inside] == "A").sum() == inside.sum() // 2 and (out[inside] == "B").sum() == inside.sum() - inside.sum() // 2
    assert (sd.faint_halves(g, half) == out).all(), "the split is seeded"


def test_the_faint_orbit_factor_is_fitted_per_band_and_replaces_f_only_in_g_13_to_15():
    rng = np.random.default_rng(0)
    g = rng.uniform(13.0, 15.0, 4000)
    w = rng.uniform(1.6, 3.0, 4000)
    f = 0.9
    truth = 0.6
    archive = np.sqrt(f**2 + truth**2 * (w**2 - 1.0))
    table = sd.fit_orbit_table_dr2(g, w, archive, single=lambda x: np.full(np.shape(x), f), edges=sd.FAINT_EDGES)
    assert len(table) == 8 and np.allclose(table, truth, atol=1e-9)
    probe_g, probe_w = np.array([12.0, 13.1, 14.9, 15.5]), np.full(4, 2.0)
    with_faint = sd.predicted_uwe(probe_g, probe_w, np.ones(16), "dr2", faint=table)
    without = sd.predicted_uwe(probe_g, probe_w, np.ones(16), "dr2")
    assert with_faint[0] == without[0] and with_faint[3] == without[3] and (with_faint[1:3] < without[1:3]).all()


def test_a_band_without_stars_of_w_1_5_or_more_gives_nan_and_not_an_error():
    table = sd.fit_orbit_table_dr2(np.array([13.1, 13.2]), np.array([1.0, 1.1]), np.array([1.0, 1.0]), single=lambda x: np.ones(np.shape(x)), edges=sd.FAINT_EDGES)
    assert np.isnan(table).all()
