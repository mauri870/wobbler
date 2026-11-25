"""The UWE factor of Gaia DR3 (wobbler.uwe_factor): its table, its interpolation, its hash, and its use in simulate_fit.
Function tests: the checks are on a table and on the numbers of three stars, not on a light curve."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import wobbler as wb

pytestmark = pytest.mark.torch_cpu
TABLE = Path(__file__).resolve().parents[1] / "src" / "wobbler" / "tables" / "uwe_factor_dr3.json"
ORBIT = wb.Orbit(period_days=325.006, ecc=0.2103, t_peri_days=-110.37, omega=1.1048)


def bands():
    return json.loads(TABLE.read_text())["bands"]


def test_the_factor_at_the_centre_of_every_band_is_the_value_of_the_band():
    for band in bands():
        centre = 0.5 * (band["G"][0] + band["G"][1])
        assert float(wb.uwe_factor(centre)) == pytest.approx(band["f"], rel=1e-12)


@pytest.mark.parametrize("name, g, expected", [("Gaia BH1", 13.771998, 1.0175), ("the dark companion of a star of G 12.57", 12.573492, 1.2655)])
def test_the_factor_at_the_g_of_two_known_stars_is_the_interpolation_of_the_neighbouring_bands(name, g, expected):
    assert float(wb.uwe_factor(g)) == pytest.approx(expected, abs=5e-4), name


def test_the_factor_peaks_near_g_11_3_and_holds_its_end_values_outside_the_table():
    assert float(wb.uwe_factor(11.35)) == pytest.approx(1.91, abs=0.02)
    first, last = bands()[0], bands()[-1]
    assert float(wb.uwe_factor(2.0)) == pytest.approx(first["f"]) and float(wb.uwe_factor(25.0)) == pytest.approx(last["f"])


def test_the_factor_takes_arrays_and_keeps_their_shape():
    out = wb.uwe_factor(np.array([[9.0, 11.3], [13.0, 16.0]]))
    assert out.shape == (2, 2) and bool((out >= 1.0).all())


def test_the_table_has_the_hash_of_its_bands_recorded_and_no_gap_between_bands():
    meta = json.loads(TABLE.read_text())
    canonical = json.dumps(meta["bands"], sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode()).hexdigest() == meta["bands_sha256"]
    edges = [band["G"] for band in meta["bands"]]
    assert all(edges[i][1] <= edges[i + 1][0] + 1e-9 for i in range(len(edges) - 1)), "bands overlap"
    assert all(band["n_archive"] >= 100 for band in meta["bands"])


def test_simulate_fit_with_the_function_gives_the_ruwe_of_the_array_of_its_values_and_one_star_rises_with_it():
    scan = wb.Scan(np.linspace(2456900.0, 2457800.0, 150), np.linspace(0.1, 3.0, 150), np.cos(np.linspace(0, 20, 150)), device="cpu")
    g = np.array([11.35, 12.573492, 13.771998])
    kwargs = dict(a1_au=0.0, parallax_mas=np.array([2.0, 2.0, 2.0]), G=g, n_orient=300, seed=5, device="cpu", nu_eff=1.47)
    by_function = wb.simulate_fit(scan, ORBIT, uwe_factor=wb.uwe_factor, **kwargs)
    by_array = wb.simulate_fit(scan, ORBIT, uwe_factor=wb.uwe_factor(g), **kwargs)
    plain = wb.simulate_fit(scan, ORBIT, uwe_factor=1.0, **kwargs)
    assert np.array_equal(by_function.ruwe.numpy(), by_array.ruwe.numpy(), equal_nan=True)
    rise = np.nanmedian(by_function.ruwe.numpy(), axis=1) / np.nanmedian(plain.ruwe.numpy(), axis=1)
    f = wb.uwe_factor(g)
    assert np.allclose(rise, f, rtol=1e-9), f"the simulated unit weight error is multiplied by the factor: {rise} against {f}"
    assert rise[0] > rise[1] > rise[2], "the rise follows the factor, largest at the peak"
    assert np.allclose(np.nanmedian(by_function.uwe.numpy(), axis=1), np.nanmedian(plain.uwe.numpy(), axis=1) * f, rtol=1e-9)


def test_the_default_of_simulate_fit_for_a_star_with_no_orbit_is_the_single_star_factor_within_two_percent():
    scan = wb.Scan(np.linspace(2456900.0, 2457800.0, 150), np.linspace(0.1, 3.0, 150), np.cos(np.linspace(0, 20, 150)), device="cpu")
    kwargs = dict(a1_au=0.0, parallax_mas=np.array([2.0]), G=11.35, n_orient=500, seed=3, device="cpu")
    default = wb.simulate_fit(scan, ORBIT, **kwargs)
    calibrated = wb.simulate_fit(scan, ORBIT, uwe_factor=wb.uwe_factor, **kwargs)
    raw = wb.simulate_fit(scan, ORBIT, uwe_factor=1.0, **kwargs)
    assert np.nanmean(default.uwe.numpy()) == pytest.approx(np.nanmean(calibrated.uwe.numpy()), rel=0.02), "with no orbit the default is the product up to the spread of the raw value"
    assert np.nanmean(default.uwe.numpy()) > 1.5 * np.nanmean(raw.uwe.numpy()), "at the peak of the factor the raw simulation is far below the archive's"
