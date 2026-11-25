"""The unit weight error of a star with an orbit (wobbler.uwe_factor.calibrated_uwe, orbit_factor, fit_orbit_factor) and its use as the default of simulate_fit.
Function tests: the checks are on a table, on a recorded subsample of Gaia DR3 orbit stars and on the arithmetic of simulate_fit, not on a light curve; a txtar archive has no place for an array of 12,000 stars."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import wobbler as wb
from wobbler.uwe_factor import calibrated_uwe, fit_orbit_factor, orbit_factor

pytestmark = pytest.mark.torch_cpu
HERE = Path(__file__).resolve().parent
TABLE = Path(__file__).resolve().parents[1] / "src" / "wobbler" / "tables" / "uwe_orbit_factor_dr3.json"
SAMPLE = HERE / "golden" / "orbit_stars_dr3.csv.gz"
EDGES9 = [9.0, 10.5, 10.75, 11.0, 11.25, 11.5, 11.75, 12.0, 12.5, 13.0]
ORBIT = wb.Orbit(period_days=325.006, ecc=0.2103, t_peri_days=-110.37, omega=1.1048)


def recorded():
    with gzip.open(SAMPLE, "rt") as handle:
        header = handle.readline().strip().split(",")
        rows = [line.strip().split(",") for line in handle]
    part = np.array([r[0] for r in rows])
    values = np.array([[float(x) for x in r[1:]] for r in rows])
    return {name: values[part == name] for name in ("fit", "test")}, header


def nine_bins(g, archive, predicted):
    ratio = archive / predicted
    return [float(np.median(ratio[(g >= low) & (g < high)])) for low, high in zip(EDGES9[:-1], EDGES9[1:])]


def test_a_star_with_no_orbit_signal_keeps_the_single_star_factor():
    for g in (9.5, 11.23, 12.57):
        assert float(calibrated_uwe(g, 1.0)) == pytest.approx(float(wb.uwe_factor(g)), rel=1e-12)


@pytest.mark.parametrize("w", [0.6, 0.9, 1.0, 1.4, 3.0])
def test_where_the_orbit_factor_is_the_single_star_factor_the_form_is_the_product_for_a_raw_value_of_any_size(w):
    """From G 13 on B is f, so f^2 + f^2 (w^2 - 1) = (f w)^2 also for w under 1: a clamp of w^2 - 1 at 0 made single stars a few percent high there."""
    for g in (13.0, 13.77, 16.0):
        assert float(calibrated_uwe(g, w)) == pytest.approx(float(wb.uwe_factor(g)) * w, rel=1e-12)


def test_a_strong_orbit_carries_the_orbit_factor_and_not_the_single_star_factor():
    g = 11.23
    assert float(calibrated_uwe(g, 500.0)) / 500.0 == pytest.approx(float(orbit_factor(g)), rel=1e-3)
    assert float(orbit_factor(g)) < float(wb.uwe_factor(g)) - 0.3, "the orbit factor of BH3's G is far under the single-star factor"


@pytest.mark.parametrize("g", [13.0, 13.77, 15.0, 17.5])
def test_beyond_the_sample_the_orbit_factor_is_the_single_star_factor(g):
    assert float(orbit_factor(g)) == pytest.approx(float(wb.uwe_factor(g)), rel=1e-12)


def test_the_orbit_factor_takes_arrays_and_holds_the_first_band_below_the_sample():
    out = orbit_factor(np.array([[9.1, 11.3], [12.9, 14.0]]))
    assert out.shape == (2, 2) and float(orbit_factor(5.0)) == pytest.approx(float(orbit_factor(9.1)))


def test_the_table_has_the_hash_of_its_bands_recorded_and_no_gap_between_bands():
    meta = json.loads(TABLE.read_text())
    canonical = json.dumps(meta["bands"], sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode()).hexdigest() == meta["bands_sha256"]
    edges = [band["G"] for band in meta["bands"]]
    assert edges[0][0] == 9.0 and edges[-1][1] == 13.0 and all(abs(edges[i][1] - edges[i + 1][0]) < 1e-9 for i in range(len(edges) - 1))
    assert all(band["n_archive"] >= 500 for band in meta["bands"])


def test_a_table_fit_on_the_recorded_fitting_stars_holds_on_the_recorded_held_out_stars_where_the_product_does_not():
    """The held-out statistic of the table: in every one of the nine bins of G the median of the archive's unit weight error over the predicted one is within 0.95 to 1.05; the product f x w, which this replaces as the default, misses."""
    parts, header = recorded()
    assert header == ["part", "G", "w", "uwe_archive"] and len(parts["fit"]) == len(parts["test"]) == 6000
    g_fit, w_fit, a_fit = parts["fit"].T
    table = fit_orbit_factor(g_fit, w_fit, a_fit)
    g, w, archive = parts["test"].T
    form = nine_bins(g, archive, calibrated_uwe(g, w, bands=table))
    product = nine_bins(g, archive, wb.uwe_factor(g) * w)
    assert all(0.95 <= x <= 1.05 for x in form), form
    assert any(not 0.95 <= x <= 1.05 for x in product), product
    assert max(abs(x - 1) for x in form) < max(abs(x - 1) for x in product) / 2, (form, product)


def test_the_shipped_table_is_close_to_a_table_fit_on_the_recorded_stars():
    parts, _ = recorded()
    g, w, archive = np.concatenate([parts["fit"], parts["test"]]).T
    refit = fit_orbit_factor(g, w, archive)
    shipped = np.array([band["B"] for band in json.loads(TABLE.read_text())["bands"]])
    assert np.max(np.abs(refit / shipped - 1)) < 0.03, (refit, shipped)


KWARGS = dict(a1_au=1.0, parallax_mas=np.array([2.0, 2.0]), G=np.array([11.23, 12.57]), n_orient=400, seed=5, device="cpu", nu_eff=np.array([1.4755, 1.4599]))


def scan():
    return wb.Scan(np.linspace(2456900.0, 2457800.0, 150), np.linspace(0.1, 3.0, 150), np.cos(np.linspace(0, 20, 150)), device="cpu")


def test_simulate_fit_applies_the_orbit_form_to_the_raw_unit_weight_error_of_every_case_by_default():
    default = wb.simulate_fit(scan(), ORBIT, **KWARGS)
    raw = wb.simulate_fit(scan(), ORBIT, uwe_factor=1.0, **KWARGS)
    g = np.repeat(KWARGS["G"][:, None], 400, axis=1)
    expected_uwe = calibrated_uwe(g, raw.uwe.numpy())
    assert np.allclose(default.uwe.numpy(), expected_uwe, rtol=1e-12, equal_nan=True)
    u0 = np.asarray(wb.u0(g, np.repeat(KWARGS["nu_eff"][:, None], 400, axis=1), device="cpu"))
    assert np.allclose(default.ruwe.numpy(), expected_uwe / u0, rtol=1e-10, equal_nan=True)


def test_simulate_fit_still_multiplies_by_the_single_star_factor_when_given_the_function():
    product = wb.simulate_fit(scan(), ORBIT, uwe_factor=wb.uwe_factor, **KWARGS)
    raw = wb.simulate_fit(scan(), ORBIT, uwe_factor=1.0, **KWARGS)
    g = np.repeat(KWARGS["G"][:, None], 400, axis=1)
    assert np.allclose(product.uwe.numpy(), raw.uwe.numpy() * wb.uwe_factor(g), rtol=1e-12, equal_nan=True)
