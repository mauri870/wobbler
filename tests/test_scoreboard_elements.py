"""validation/scoreboard_elements.py: the covariance of an orbit solution from its errors and corr_vec, the draws of its elements, and the cases through the fit, checked against simulate_fit."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))
import scoreboard_elements as se  # noqa: E402
from test_scoreboard import synthetic_scan  # noqa: E402,F401  (a fixture)


def random_correlation(n, seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    c = a @ a.T
    d = np.sqrt(np.diag(c))
    return c / d[:, None] / d[None, :]


def column_major_upper(r):
    n = len(r)
    return [r[i, j] for j in range(1, n) for i in range(j)]


@pytest.mark.parametrize("n", [12, 15])
def test_the_correlation_is_rebuilt_from_the_column_major_upper_triangle_without_the_diagonal(n):
    r = random_correlation(n, n)
    vector = column_major_upper(r)
    assert len(vector) == n * (n - 1) // 2 == (66 if n == 12 else 105)
    text = "(" + ", ".join(repr(float(x)) for x in vector) + ")"
    np.testing.assert_allclose(se.correlation(se.parse_vector(text), n), r, atol=1e-15)


def test_the_first_values_are_the_pairs_in_the_documented_order():
    r = np.eye(4)
    r[0, 1] = r[1, 0] = 0.1
    r[0, 2] = r[2, 0] = 0.2
    r[1, 2] = r[2, 1] = 0.3
    r[0, 3] = r[3, 0] = 0.4
    assert column_major_upper(r)[:4] == [0.1, 0.2, 0.3, 0.4]
    np.testing.assert_array_equal(se.correlation(np.array([0.1, 0.2, 0.3, 0.4, 0.0, 0.0]), 4), r)


def row_of(kind="Orbital", seed=1):
    names = se.ORBITAL if kind == "Orbital" else se.ASTROSPECTROSB1
    r = random_correlation(len(names), seed)
    errors = {"ra_error": 0.5, "dec_error": 0.4, "parallax_error": 0.05, "pmra_error": 0.06, "pmdec_error": 0.07, "a_error": 0.3, "b_error": 0.2, "f_error": 0.25, "g_error": 0.35, "c_error": 0.1, "h_error": 0.12,
              "period_error": 3.0, "t_peri_error": 9.0, "ecc_error": 0.04, "velocity_error": 0.5}
    return {"type": kind, "bit_index": "8191" if kind == "Orbital" else "65535", "corr_vec": "(" + ", ".join(repr(float(x)) for x in column_major_upper(r)) + ")", **{k: str(v) for k, v in errors.items()}}, r, errors


STAR = {"orb_parallax": 2.0, "orb_pmra": 5.0, "orb_pmdec": -3.0, "ti_a": 1.2, "ti_b": 0.7, "ti_f": -0.5, "ti_g": 0.9, "ecc": 0.3, "period": 300.0, "t_peri": 20.0}


@pytest.mark.parametrize("kind", ["Orbital", "AstroSpectroSB1"])
def test_the_covariance_of_the_astrometric_parameters_is_the_scaled_submatrix(kind):
    row, r, errors = row_of(kind)
    mean, cov = se.mean_and_covariance(STAR, row)
    assert len(mean) == 10 == len(se.USED)
    names = se.ORBITAL if kind == "Orbital" else se.ASTROSPECTROSB1
    index = [names.index(n) for n in se.USED]
    sigma = np.array([errors[se.ERROR_OF[n]] for n in se.USED])
    np.testing.assert_allclose(cov, r[np.ix_(index, index)] * sigma[:, None] * sigma[None, :], rtol=1e-12)
    np.testing.assert_allclose(mean, [2.0, 5.0, -3.0, 1.2, 0.7, -0.5, 0.9, 0.3, 300.0, 20.0])


def test_a_parameter_with_no_error_is_held_fixed():
    row, _, _ = row_of("Orbital")
    row["g_error"], row["ecc_error"] = "", ""
    mean, cov = se.mean_and_covariance(STAR, row)
    for name in ("G", "ecc"):
        k = se.USED.index(name)
        assert cov[k, k] == 0 and not cov[k].any() and not cov[:, k].any()


def test_the_draws_follow_the_covariance_and_the_clipping_is_counted():
    row, _, _ = row_of("Orbital", seed=3)
    mean, cov = se.mean_and_covariance(STAR, row)
    draws, clipped = se.sample(mean, cov, 20000, np.random.default_rng(0))
    assert draws.shape == (20000, 10) and clipped == 0
    np.testing.assert_allclose(draws.mean(axis=0), mean, atol=5 * np.sqrt(np.diag(cov) / 20000).max() * 4)
    np.testing.assert_allclose(np.cov(draws.T), cov, rtol=0.1, atol=0.03 * np.abs(cov).max())
    wide = cov.copy()
    wide[7, 7] = 1.0  # an eccentricity error of 1: many draws fall outside 0 to 0.99
    draws, clipped = se.sample(mean, wide, 2000, np.random.default_rng(1))
    assert clipped > 0 and draws[:, 7].min() >= 0 and draws[:, 7].max() <= 0.99 and draws[:, 8].min() >= 1.0


def test_a_covariance_of_zero_gives_the_nominal_elements_in_every_draw():
    mean = np.array([2.0, 5.0, -3.0, 1.2, 0.7, -0.5, 0.9, 0.3, 300.0, 20.0])
    draws, clipped = se.sample(mean, np.zeros((10, 10)), 5, np.random.default_rng(0))
    np.testing.assert_array_equal(draws, np.tile(mean, (5, 1))) and clipped == 0


@pytest.mark.torch_cpu
def test_the_cases_through_the_fit_equal_simulate_fit_when_the_covariance_is_zero():
    """The differential test: the same noise, the nominal orbit in every case; offsets, unit weight error and ok must agree with simulate_fit."""
    import wobbler as wb

    rng = np.random.default_rng(5)
    scan = wb.Scan(jd=np.sort(rng.uniform(2456891.5, 2457902.0, 70)), psi=rng.uniform(0, np.pi, 70), plx_factor=rng.uniform(-1, 1, 70), device="cpu")
    cases, epochs = 6, 70
    noise = {"keep": rng.uniform(size=(cases, epochs)) > 0.1, "z_err": rng.normal(size=(cases, epochs)), "z_extra": rng.normal(size=(cases, epochs)), "extra": rng.uniform(0, 0.04, cases)}
    mean = np.array([2.0, 5.0, -3.0, 1.2, 0.7, -0.5, 0.9, 0.3, 300.0, 20.0])
    elements = wb.campbell(1.2, 0.7, -0.5, 0.9)
    incl, node, omega, a0 = (float(elements[k]) for k in ("inclination", "node", "omega", "a"))
    orbit = wb.Orbit(period_days=300.0, ecc=0.3, t_peri_days=20.0, omega=omega)
    reference = wb.simulate_fit(scan, orbit, a1_au=a0 * np.sin(incl) / 2.0, parallax_mas=np.array([2.0]), G=11.5, n_orient=cases, inclination=incl, node=node, windows={"DR3": wb.WINDOWS["DR3"]},
                                pm=(5.0, -3.0), noise=noise, nu_eff=1.5, uwe_factor=1.0, device="cpu", parameters=True).windows["DR3"]
    mine = se.cases_through_the_fit(wb, scan, 11.5, 1.5, np.tile(mean, (cases, 1)), mean[:3], noise, "cpu")
    np.testing.assert_allclose(mine["uwe"], reference["uwe"][0].numpy(), rtol=1e-9)
    np.testing.assert_allclose(mine["d_parallax"], reference["parallax_mas"][0].numpy() - 2.0, atol=1e-9)
    np.testing.assert_allclose(mine["d_pmra"], reference["pmra_mas_yr"][0].numpy() - 5.0, atol=1e-9)
    np.testing.assert_allclose(mine["d_pmdec"], reference["pmdec_mas_yr"][0].numpy() + 3.0, atol=1e-9)
    assert (mine["ok"] == reference["ok"][0].numpy()).all()


@pytest.mark.torch_cpu
def test_simulate_and_score_run_end_to_end_on_synthetic_stars_and_resume(tmp_path, synthetic_scan):  # noqa: F811
    import json

    from test_scoreboard import synthetic_rows

    synthetic_rows(tmp_path / "rows", n=60, seed=11)
    stars = sb_load(tmp_path / "rows")
    orbit_rows = []
    for k, star in enumerate(stars):
        row, _, _ = row_of("Orbital" if k % 2 else "AstroSpectroSB1", seed=k)
        orbit_rows.append({"source_id": str(star["source_id"]), **row})
    (tmp_path / "orbits").mkdir()
    (tmp_path / "orbits" / "g.json").write_text(json.dumps({"rows": orbit_rows}))
    first = se.simulate(tmp_path / "rows", tmp_path / "orbits", tmp_path / "out", limit=3, draws=16, part_size=2)
    rest = se.simulate(tmp_path / "rows", tmp_path / "orbits", tmp_path / "out", draws=16)
    assert first == 3 and se.simulate(tmp_path / "rows", tmp_path / "orbits", tmp_path / "out", draws=16) == 0
    half = se.sb.halves([s["g"] for s in stars])
    assert first + rest == int(sum(1 for s, h in zip(stars, half) if s["g"] < 13 and h == "B"))
    result = se.score(tmp_path / "rows", tmp_path / "out", (1.1, 1.0, 1.0))
    assert result["stars"] == first + rest and set(result["overall"]) >= {"z_el_parallax", "pull_el_pmra", "pull_el_scan_pmdec"} and (tmp_path / "out" / "elements.json").exists()


def sb_load(path):
    return se.sb.load_rows(path)


@pytest.mark.parametrize("kind, bit_index, unfitted", [("Orbital", "8179", ("G", "ecc")), ("AstroSpectroSB1", "65435", ("G", "ecc"))])
def test_a_solution_with_unfitted_parameters_has_a_finite_covariance_with_those_held_fixed(kind, bit_index, unfitted):
    """bit_index 8179 and 65435 leave G, the eccentricity (and C) unfitted: their errors are empty and their correlations NaN."""
    names = se.ORBITAL if kind == "Orbital" else se.ASTROSPECTROSB1
    row, r, _ = row_of(kind, seed=7)
    r = r.copy()
    for name in unfitted + (("C",) if kind == "AstroSpectroSB1" else ()):
        k = names.index(name)
        r[k, :] = np.nan
        r[:, k] = np.nan
    row["bit_index"] = bit_index
    row["corr_vec"] = "(" + ", ".join("NaN" if np.isnan(x) else repr(float(x)) for x in column_major_upper(r)) + ")"
    for name in unfitted:
        row[se.ERROR_OF[name]] = ""
    mean, cov = se.mean_and_covariance(STAR, row)
    assert np.isfinite(cov).all()
    for name in unfitted:
        k = se.USED.index(name)
        assert not cov[k].any() and not cov[:, k].any()
    draws, _ = se.sample(mean, cov, 50, np.random.default_rng(0))
    assert np.isfinite(draws).all() and (draws[:, se.USED.index("G")] == mean[se.USED.index("G")]).all()


@pytest.mark.torch_cpu
def test_a_noise_scale_widens_the_offsets_by_that_factor_and_one_changes_nothing():
    import wobbler as wb

    rng = np.random.default_rng(2)
    scan = wb.Scan(jd=np.sort(rng.uniform(2456891.5, 2457902.0, 70)), psi=rng.uniform(0, np.pi, 70), plx_factor=rng.uniform(-1, 1, 70), device="cpu")
    cases = 4000
    noise = {"keep": np.ones((cases, 70), dtype=bool), "z_err": rng.normal(size=(cases, 70)), "z_extra": np.zeros((cases, 70)), "extra": np.zeros(cases)}
    mean = np.array([2.0, 5.0, -3.0, 1e-9, 1e-9, 1e-9, 1e-9, 0.3, 300.0, 20.0])  # no orbit signal: the offsets are the noise
    samples = np.tile(mean, (cases, 1))
    one = se.cases_through_the_fit(wb, scan, 12.0, 1.5, samples, mean[:3], noise, "cpu")
    same = se.cases_through_the_fit(wb, scan, 12.0, 1.5, samples, mean[:3], noise, "cpu", noise_scale=1.0)
    double = se.cases_through_the_fit(wb, scan, 12.0, 1.5, samples, mean[:3], noise, "cpu", noise_scale=2.0)
    for key in ("d_parallax", "d_pmra", "d_pmdec", "uwe"):
        np.testing.assert_array_equal(one[key], same[key])
    for key in ("d_parallax", "d_pmra", "d_pmdec"):
        assert np.std(double[key]) / np.std(one[key]) == pytest.approx(2.0, rel=0.02), key
    assert np.median(double["uwe"]) > 1.1 * np.median(one["uwe"]), "the weights stay at the raw error, so the unit weight error grows with the noise (by less than the factor: the unbinned conversion adds a constant)"
