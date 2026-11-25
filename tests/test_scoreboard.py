"""validation/scoreboard.py: the halves, the statistics, the bootstrap, the improvement rule and the simulation of a star, on synthetic stars with known answers (no archive, no scan data)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))
import scoreboard as sb  # noqa: E402


def test_the_halves_split_the_bright_stars_with_the_seeded_permutation_and_every_faint_star_is_a_test_star():
    g = np.concatenate([np.linspace(9, 12.99, 101), np.linspace(13, 15.99, 60)])
    half = sb.halves(g)
    bright = np.flatnonzero(g < 13)
    assert (half[bright] == "A").sum() == len(bright) // 2 and (half[bright] == "B").sum() == len(bright) - len(bright) // 2
    expected = np.random.default_rng(1692).permutation(len(bright))[: len(bright) // 2]
    assert set(bright[expected]) == set(np.flatnonzero(half == "A"))
    assert (half[g >= 13] == "B").all(), "no star of G 13 and above is in a fit"
    assert (sb.halves(g) == half).all()
    # the halves of the bright stars do not depend on which faint stars are cached
    assert (sb.halves(g[g < 13]) == half[g < 13]).all()


def test_a_stratum_with_one_star_does_not_fail():
    assert list(sb.halves([10.0])) == ["B"] and len(sb.halves([])) == 0 and list(sb.halves([14.0])) == ["B"]


def test_the_width_of_a_gaussian_is_its_sigma_and_the_error_falls_as_one_over_root_n():
    rng = np.random.default_rng(0)
    errors = {}
    for n in (2000, 8000):
        x = rng.normal(0.3, 2.0, n)
        result = sb.summary(x, np.random.default_rng(1).integers(0, n, size=(200, n)))
        assert result["width"] == pytest.approx(2.0, rel=0.06) and result["median"] == pytest.approx(0.3, abs=0.15)
        assert result["median_error"] == pytest.approx(2.0 * 1.25 / np.sqrt(n), rel=0.3)
        errors[n] = result["width_error"]
    assert errors[2000] / errors[8000] == pytest.approx(2.0, rel=0.35)


def stars_of(n, seed=0):
    rng = np.random.default_rng(seed)
    return {"g": rng.uniform(9, 16, n), "period": rng.uniform(5, 3000, n), "ecc": rng.uniform(0, 1, n), "ecl_lat": rng.uniform(-90, 90, n), "n_good": rng.integers(60, 400, n).astype(float)}


def test_every_star_is_in_exactly_one_bin_of_each_variable():
    stars = stars_of(3000)
    edges = sb.n_good_edges_of(stars["n_good"])
    for name, group in sb.bins_of(stars, edges).items():
        total = sum(mask.astype(int) for _, mask in group)
        assert (total == 1).all(), name


def test_the_quantities_follow_their_definitions():
    stars = {"ruwe": np.array([3.0]), "parallax": np.array([1.5]), "orb_parallax": np.array([1.0]), "parallax_error": np.array([0.1]), "pmra": np.array([0.0]), "orb_pmra": np.array([0.0]), "pmra_error": np.array([0.2]),
             "pmdec": np.array([2.0]), "orb_pmdec": np.array([1.0]), "pmdec_error": np.array([0.5])}
    predicted = {"ruwe": np.array([2.0]), "d_parallax": np.array([0.3]), "d_pmra": np.array([0.0]), "d_pmdec": np.array([1.0]), "s_parallax": np.array([0.2]), "s_pmra": np.array([0.2]), "s_pmdec": np.array([0.25]),
                 "o16_parallax": np.array([0.1]), "o84_parallax": np.array([0.5]), "o16_pmra": np.array([-0.2]), "o84_pmra": np.array([0.2]), "o16_pmdec": np.array([0.5]), "o84_pmdec": np.array([1.5])}
    q = sb.quantities(stars, predicted)
    assert q["ruwe"][0] == pytest.approx(np.log(1.5))
    assert q["offset_parallax"][0] == pytest.approx((0.5 - 0.3) / 0.1) and q["zero_parallax"][0] == pytest.approx(5.0)
    assert q["pull_parallax"][0] == pytest.approx((0.5 - 0.3) / 0.2) and q["pull_pmdec"][0] == pytest.approx(0.0) and q["pull_pmra"][0] == pytest.approx(0.0)
    assert q["offset_pmdec"][0] == pytest.approx(0.0) and q["error_pmdec"][0] == pytest.approx(np.log(2.0)) and q["error_parallax"][0] == pytest.approx(np.log(0.5))


def per_star_file(path, ids, values, stars, edges):
    np.savez(path, source_id=ids, n_good_edges=np.array(edges), **stars, **{"v_" + k: v for k, v in values.items()})


def test_the_improvement_rule(tmp_path):
    n = 4000
    stars, rng = stars_of(n, 3), np.random.default_rng(4)
    ids, edges = np.arange(n), sb.n_good_edges_of(stars["n_good"])
    base = rng.normal(0.0, 1.0, n)
    per_star_file(tmp_path / "old.npz", ids, {"ruwe": base}, stars, edges)
    # identical: no improvement, no change
    per_star_file(tmp_path / "same.npz", ids, {"ruwe": base}, stars, edges)
    same = sb.compare(np.load(tmp_path / "old.npz"), np.load(tmp_path / "same.npz"))["ruwe"]
    assert not same["improved"] and same["overall"]["width_change"] == 0.0 and not same["regressions"]
    # every star's value shrunk by 20 percent: W falls by 20 percent everywhere
    per_star_file(tmp_path / "narrow.npz", ids, {"ruwe": 0.8 * base}, stars, edges)
    narrow = sb.compare(np.load(tmp_path / "old.npz"), np.load(tmp_path / "narrow.npz"))["ruwe"]
    assert narrow["improved"] and narrow["overall"]["width_change"] == pytest.approx(-0.2 * narrow["overall"]["width_old"], rel=1e-9)
    # narrower overall but three times wider in one bin of eccentricity: not an improvement
    wide = 0.8 * base
    mask = stars["ecc"] < 0.1
    wide[mask] = 3.0 * base[mask]
    per_star_file(tmp_path / "mixed.npz", ids, {"ruwe": wide}, stars, edges)
    mixed = sb.compare(np.load(tmp_path / "old.npz"), np.load(tmp_path / "mixed.npz"))["ruwe"]
    assert not mixed["improved"] and mixed["bins"]["eccentricity: 0 to 0.1"]["width_change"] > 0
    # a shift of the median in one bin is a regression even where W falls
    shifted = 0.8 * base
    shifted[stars["ecc"] >= 0.7] += 1.0
    per_star_file(tmp_path / "shifted.npz", ids, {"ruwe": shifted}, stars, edges)
    moved = sb.compare(np.load(tmp_path / "old.npz"), np.load(tmp_path / "shifted.npz"))["ruwe"]
    assert "eccentricity: 0.7 to 1" in moved["regressions"]


def test_compare_pairs_the_stars_by_source_id_when_a_file_is_not_sorted(tmp_path):
    """The per-star file of a whole run holds the scored half of G below 13 and then the stars of G 13 and above, each block sorted by source_id, so the file is not sorted as a whole; a pairing by searchsorted on it paired other stars (a first comparison of two runs gave a width of 0.0788 for the old run where it is 0.098)."""
    n = 4000
    stars, rng = stars_of(n, 3), np.random.default_rng(4)
    ids, edges = np.arange(n), sb.n_good_edges_of(stars["n_good"])
    base = rng.normal(0.0, 1.0, n)
    order = np.concatenate([np.arange(2000, n), np.arange(2000)])  # two sorted blocks, the second block first
    per_star_file(tmp_path / "old.npz", ids[order], {"ruwe": base[order]}, {key: value[order] for key, value in stars.items()}, edges)
    per_star_file(tmp_path / "new.npz", ids, {"ruwe": 0.8 * base}, stars, edges)
    report = sb.compare(np.load(tmp_path / "old.npz"), np.load(tmp_path / "new.npz"))["ruwe"]
    assert report["overall"]["n"] == n and report["improved"]
    assert report["overall"]["width_change"] == pytest.approx(-0.2 * report["overall"]["width_old"], rel=1e-9)


def test_a_bin_below_the_minimum_is_not_compared(tmp_path):
    n = 600
    stars = stars_of(n, 5)
    stars["ecc"][:50] = 0.05  # 50 stars in the first bin of eccentricity
    stars["ecc"][50:] = 0.9
    ids, edges = np.arange(n), sb.n_good_edges_of(stars["n_good"])
    base = np.random.default_rng(6).normal(size=n)
    per_star_file(tmp_path / "a.npz", ids, {"ruwe": base}, stars, edges)
    per_star_file(tmp_path / "b.npz", ids, {"ruwe": 5 * base}, stars, edges)
    report = sb.compare(np.load(tmp_path / "a.npz"), np.load(tmp_path / "b.npz"))["ruwe"]
    assert "eccentricity: 0 to 0.1" not in report["bins"] and "eccentricity: 0.7 to 1" in report["bins"]


def synthetic_rows(directory, n=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n):
        g = float(rng.uniform(9.5, 15.5))
        rows.append({"source_id": str(1000 + k), "type": "Orbital", "period": str(rng.uniform(100, 900)), "ecc": str(rng.uniform(0, 0.6)), "t_peri": str(rng.uniform(-200, 200)),
                     "ti_a": str(rng.uniform(0.3, 2)), "ti_b": str(rng.uniform(-1, 1)), "ti_f": str(rng.uniform(-1, 1)), "ti_g": str(rng.uniform(-1, 1)), "orb_parallax": "2.0", "orb_pmra": "1.0", "orb_pmdec": "2.0", "ra": str(rng.uniform(0, 360)), "dec": str(rng.uniform(-60, 60)),
                     "g": str(g), "parallax": "2.0", "parallax_error": "0.05", "pmra": "1.0", "pmra_error": "0.06", "pmdec": "2.0", "pmdec_error": "0.05", "nu_eff": "1.5", "ruwe": str(rng.uniform(1.0, 6.0)),
                     "chi2": str(rng.uniform(200, 3000)), "n_good": str(int(rng.integers(100, 300))), "ecl_lat": str(rng.uniform(-80, 80))})
    Path(directory).mkdir(parents=True, exist_ok=True)
    (Path(directory) / "g.json").write_text(json.dumps({"rows": rows}))


@pytest.fixture
def synthetic_scan(monkeypatch):
    import wobbler as wb

    def from_position(ra, dec, window="DR3", device=None):
        rng = np.random.default_rng(int(abs(ra) * 100 + abs(dec)))
        return wb.Scan(jd=np.sort(rng.uniform(2456891.5, 2457902.0, 70)), psi=rng.uniform(0, np.pi, 70), plx_factor=rng.uniform(-1, 1, 70), device="cpu")

    monkeypatch.setattr(wb.Scan, "from_position", staticmethod(from_position))


pytestmark_torch = pytest.mark.torch_cpu


@pytest.mark.torch_cpu
def test_simulate_resumes_and_a_one_shot_run_gives_the_same_stars(tmp_path, synthetic_scan):
    synthetic_rows(tmp_path / "rows")
    first = sb.simulate(tmp_path / "rows", tmp_path / "out", limit=10, draws=12, part_size=5)
    assert first[0] == 10
    rest = sb.simulate(tmp_path / "rows", tmp_path / "out", draws=12)
    again = sb.simulate(tmp_path / "rows", tmp_path / "out", draws=12)
    assert again[0] == 0, "nothing is simulated twice"
    sb.simulate(tmp_path / "rows", tmp_path / "once", draws=12)
    a, b = (sb.join(tmp_path / "rows", tmp_path / name) for name in ("out", "once"))
    order_a, order_b = np.argsort(a[0]["source_id"]), np.argsort(b[0]["source_id"])
    np.testing.assert_array_equal(a[0]["source_id"][order_a], b[0]["source_id"][order_b])
    np.testing.assert_allclose(a[1]["w_raw"][order_a], b[1]["w_raw"][order_b], rtol=0, atol=0)
    half_a = sb.halves([s["g"] for s in sb.load_rows(tmp_path / "rows")])
    assert first[0] + rest[0] == len(half_a), "every star of G below 13 and every star of G 13 and above is simulated"


@pytest.mark.torch_cpu
def test_a_star_with_no_orbit_signal_has_no_offset_and_a_unit_weight_error_near_one(synthetic_scan):
    import wobbler as wb

    star = {"source_id": 7.0, "period": 300.0, "ecc": 0.2, "t_peri": 0.0, "ti_a": 1e-9, "ti_b": 1e-9, "ti_f": 1e-9, "ti_g": 1e-9, "orb_parallax": 2.0, "ra": 100.0, "dec": 20.0, "g": 14.5, "nu_eff": 1.5}
    result = sb.simulate_star(wb, star, 60, "cpu")
    assert abs(result["d_parallax"]) < 0.5 * result["s_parallax"] and abs(result["d_pmra"]) < 0.5 * result["s_pmra"] and abs(result["d_pmdec"]) < 0.5 * result["s_pmdec"]
    assert 0.8 < result["w_raw"] < 1.4 and result["w16"] <= result["w_raw"] <= result["w84"] and result["n_ok"] >= 30
    for p in sb.PARAMETERS:
        assert result["o16_" + p] <= result["d_" + p] <= result["o84_" + p], "the median of the draws lies between their 16th and 84th percentile"


@pytest.mark.torch_cpu
def test_score_writes_the_table_and_the_per_star_file(tmp_path, synthetic_scan):
    synthetic_rows(tmp_path / "rows", n=60, seed=2)
    sb.simulate(tmp_path / "rows", tmp_path / "out", draws=12)
    result = sb.score(tmp_path / "rows", tmp_path / "out")
    assert len(result["orbit_factor_fitted_on_half_A"]) == 16
    assert result["stars"]["scored"] > 0 and (tmp_path / "out" / "stars.npz").exists() and (tmp_path / "out" / "scoreboard.json").exists()
    assert set(result["overall"]) >= {"ruwe", "offset_parallax", "zero_pmdec", "error_pmra"}
    assert all(set(cell) == {"n"} for group in result["bins"].values() for cell in (c["ruwe"] for c in group.values()))  # 60 stars: every bin is under the minimum


@pytest.mark.torch_cpu
def test_a_named_star_row_orders_its_percentiles_and_matches_the_definitions(tmp_path, synthetic_scan):
    import wobbler as wb

    synthetic_rows(tmp_path, n=1, seed=9)
    row = json.loads((tmp_path / "g.json").read_text())["rows"][0]
    gaia = {key: row[key] for key in ("g", "parallax", "parallax_error", "pmra", "pmra_error", "pmdec", "pmdec_error", "nu_eff", "ruwe", "ra", "dec")}
    orbit = {key: row[key] for key in ("source_id", "type", "period", "ecc", "t_peri", "ti_a", "ti_b", "ti_f", "ti_g", "orb_parallax", "orb_pmra", "orb_pmdec")}
    star = sb.star_of({"gaia_source": [gaia], "nss_two_body_orbit": [orbit]})
    table = np.full(16, 1.3)
    result = sb.star_row(wb, star, table, draws=40, device="cpu")
    low, mid, high = result["ruwe_16_50_84"]
    assert low <= mid <= high and result["ln_median_predicted_over_archive_ruwe"] == pytest.approx(np.log(mid / star["ruwe"]))
    for p in sb.PARAMETERS:
        a, b, c = result[p]["predicted_offset_16_50_84"]
        assert a <= b <= c and result[p]["archive_offset"] == pytest.approx(star[p] - star["orb_" + p]) and result[p]["predicted_error"] > 0
    assert result["draws_that_fit"] >= 20


@pytest.mark.torch_cpu
def test_the_swapped_score_fits_on_half_b_and_scores_the_bright_stars_of_half_a(tmp_path, synthetic_scan):
    synthetic_rows(tmp_path / "rows", n=80, seed=4)
    sb.simulate(tmp_path / "rows", tmp_path / "out", draws=12)
    stars = sb.load_rows(tmp_path / "rows")
    half = sb.halves([s["g"] for s in stars])
    bright = np.array([s["g"] < 13 for s in stars])
    result = sb.score(tmp_path / "rows", tmp_path / "out", swapped=True)
    assert result["stars"]["swapped"] and result["stars"]["scored"] == int(((half == "A") & bright).sum()) and result["stars"]["fitting_half_below_G13"] == int(((half == "B") & bright).sum())
    assert "orbit_factor_fitted_on_half_B" in result and (tmp_path / "out" / "stars_swapped.npz").exists() and (tmp_path / "out" / "scoreboard_swapped.json").exists()


@pytest.mark.torch_cpu
def test_a_seed_offset_changes_the_noise_and_the_same_offset_reproduces_it(synthetic_scan):
    import wobbler as wb

    star = {"source_id": 11.0, "period": 300.0, "ecc": 0.2, "t_peri": 0.0, "ti_a": 1.5, "ti_b": 0.4, "ti_f": -0.3, "ti_g": 0.8, "orb_parallax": 2.0, "ra": 100.0, "dec": 20.0, "g": 12.0, "nu_eff": 1.5}
    first, again, other = (sb.simulate_star(wb, star, 40, "cpu", offset) for offset in (0, 0, 1000))
    assert first == again and first["w_raw"] != other["w_raw"] and first["d_pmra"] != other["d_pmra"]


@pytest.mark.torch_cpu
def test_g_max_limits_the_simulation_to_brighter_stars_and_a_later_run_adds_the_rest(tmp_path, synthetic_scan):
    synthetic_rows(tmp_path / "rows", n=30, seed=6)
    g = np.array([s["g"] for s in sb.load_rows(tmp_path / "rows")])
    bright = sb.simulate(tmp_path / "rows", tmp_path / "out", draws=12, g_max=13.0)
    assert bright[0] == int((g < 13).sum())
    rest = sb.simulate(tmp_path / "rows", tmp_path / "out", draws=12)
    assert rest[0] == int((g >= 13).sum())


def test_two_source_ids_that_differ_in_the_last_digit_stay_two_stars(tmp_path):
    """Gaia source ids are above 2^53, so a float64 merges neighbouring ids: the rows of two stars with ids 1 apart must stay two."""
    synthetic_rows(tmp_path / "rows", n=2, seed=1)
    path = tmp_path / "rows" / "g.json"
    rows = json.loads(path.read_text())["rows"]
    rows[0]["source_id"], rows[1]["source_id"] = "4373465352415301632", "4373465352415301633"
    path.write_text(json.dumps({"rows": rows}))
    stars = sb.load_rows(tmp_path / "rows")
    assert [s["source_id"] for s in stars] == [4373465352415301632, 4373465352415301633]
    assert all(isinstance(s["source_id"], int) for s in stars)


@pytest.mark.torch_cpu
def test_the_simulated_ids_are_exact_integers(tmp_path, synthetic_scan):
    synthetic_rows(tmp_path / "rows", n=3, seed=1)
    path = tmp_path / "rows" / "g.json"
    rows = json.loads(path.read_text())["rows"]
    for k, row in enumerate(rows):
        row["source_id"] = str(4373465352415301632 + k)
    path.write_text(json.dumps({"rows": rows}))
    sb.simulate(tmp_path / "rows", tmp_path / "out", draws=12)
    stars, predicted, half = sb.join(tmp_path / "rows", tmp_path / "out")
    assert sorted(stars["source_id"].tolist()) == [4373465352415301632 + k for k in range(3)] and stars["source_id"].dtype == np.int64


@pytest.mark.torch_cpu
def test_the_good_observation_edges_can_be_pinned(tmp_path, synthetic_scan):
    synthetic_rows(tmp_path / "rows", n=40, seed=8)
    sb.simulate(tmp_path / "rows", tmp_path / "out", draws=12)
    result = sb.score(tmp_path / "rows", tmp_path / "out", n_good_edges=[100.0, 150.0, 200.0, 250.0, 400.0])
    assert result["n_good_edges"] == [100.0, 150.0, 200.0, 250.0, 400.0]
    assert list(result["bins"]["good observations"]) == ["100 to 150", "150 to 200", "200 to 250", "250 to 400"]


@pytest.mark.torch_cpu
def test_simulate_with_the_exact_scans_of_the_stars_gives_the_stars_of_a_run_that_takes_the_same_scans_from_their_position(tmp_path, synthetic_scan, monkeypatch):
    """The scans come from a directory of GOST batches (batch_NNNN.csv, a target a star named by its source id) and not from the nearest node; the same simulation code runs on both."""
    import wobbler as wb

    synthetic_rows(tmp_path / "rows")
    reference = tmp_path / "reference"
    sb.simulate(tmp_path / "rows", reference, draws=12)
    lines = ["Target, ra[rad], dec[rad], scanAngle[rad], parallaxFactorAlongScan, ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]"]
    for star in sb.load_rows(tmp_path / "rows"):
        s = wb.Scan.from_position(star["ra"], star["dec"], device="cpu")
        lines += [f"{int(star['source_id'])},0,0,{psi!r},{plx!r},{jd!r}" for jd, psi, plx in zip(s.jd.tolist(), s.psi.tolist(), s.plx_factor.tolist())]
    (tmp_path / "scans").mkdir()
    (tmp_path / "scans" / "batch_0000.csv").write_text("\n".join(lines) + "\n")

    def refused(*arguments, **keywords):
        raise AssertionError("the scan of the nearest node was used")

    monkeypatch.setattr(wb.Scan, "from_position", staticmethod(refused))
    exact = tmp_path / "exact"
    assert sb.simulate(tmp_path / "rows", exact, draws=12, scans=tmp_path / "scans")[1] == 0
    a, b = (sb.join(tmp_path / "rows", directory) for directory in (reference, exact))
    order_a, order_b = np.argsort(a[0]["source_id"]), np.argsort(b[0]["source_id"])
    np.testing.assert_array_equal(a[0]["source_id"][order_a], b[0]["source_id"][order_b])
    np.testing.assert_allclose(a[1]["w_raw"][order_a], b[1]["w_raw"][order_b], rtol=0, atol=0)


@pytest.mark.torch_cpu
def test_a_star_that_has_no_scan_in_the_directory_is_dropped_and_counted_and_the_others_are_simulated(tmp_path, synthetic_scan):
    synthetic_rows(tmp_path / "rows")
    stars = sb.load_rows(tmp_path / "rows")
    import wobbler as wb

    lines = ["Target, ra[rad], dec[rad], scanAngle[rad], parallaxFactorAlongScan, ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]"]
    for star in stars[1:]:
        s = wb.Scan.from_position(star["ra"], star["dec"], device="cpu")
        lines += [f"{int(star['source_id'])},0,0,{psi!r},{plx!r},{jd!r}" for jd, psi, plx in zip(s.jd.tolist(), s.psi.tolist(), s.plx_factor.tolist())]
    (tmp_path / "scans").mkdir()
    (tmp_path / "scans" / "batch_0000.csv").write_text("\n".join(lines) + "\n")
    done, dropped = sb.simulate(tmp_path / "rows", tmp_path / "out", draws=12, scans=tmp_path / "scans")
    assert done == len(stars) and dropped == 1
