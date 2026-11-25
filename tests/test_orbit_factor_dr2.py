"""The orbit factor of Gaia DR2 (wobbler.uwe_factor.orbit_factor_dr2, calibrated_uwe_dr2): the table, its lookup and its use beside the single-star factor of DR2. Function tests on the table; the fit is validation/scoreboard_dr2.py."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from wobbler.uwe_factor import calibrated_uwe_dr2, orbit_factor_dr2, uwe_factor_dr2

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "src" / "wobbler" / "tables" / "uwe_orbit_factor_dr2.json"
sys.path.insert(0, str(ROOT / "validation"))


def test_the_table_has_contiguous_bands_of_a_quarter_magnitude_from_g_9_to_15():
    bands = json.loads(TABLE.read_text())["bands"]
    edges = [b["G"][0] for b in bands] + [bands[-1]["G"][1]]
    assert len(bands) == 24 and np.allclose(np.diff(edges), 0.25) and edges[0] == 9.0 and edges[-1] == 15.0
    assert all(0.5 < b["B"] < 1.0 for b in bands)


def test_the_factor_is_the_band_value_inside_the_table_and_the_single_star_factor_outside_it():
    bands = json.loads(TABLE.read_text())["bands"]
    middle = np.array([0.5 * (b["G"][0] + b["G"][1]) for b in bands])
    assert np.array_equal(orbit_factor_dr2(middle), [b["B"] for b in bands])
    assert float(orbit_factor_dr2(8.0)) == bands[0]["B"]
    assert float(orbit_factor_dr2(15.0)) == float(uwe_factor_dr2(15.0)) and float(orbit_factor_dr2(17.3)) == float(uwe_factor_dr2(17.3))


def test_a_star_with_no_orbit_signal_gets_the_single_star_factor_and_a_strong_one_gets_b_times_w():
    assert float(calibrated_uwe_dr2(12.0, 1.0)) == pytest.approx(float(uwe_factor_dr2(12.0)))
    assert float(calibrated_uwe_dr2(12.0, 50.0)) == pytest.approx(float(orbit_factor_dr2(12.0)) * 50.0, rel=2e-3)


def test_the_prediction_equals_the_one_of_the_scoring_script_and_keeps_the_shape():
    import scoreboard_dr2 as sd

    bands = json.loads(TABLE.read_text())["bands"]
    bright, faint = [b["B"] for b in bands[:16]], [b["B"] for b in bands[16:]]
    g = np.array([[9.5, 12.3, 13.1], [14.9, 15.5, 8.0]])
    w = np.array([[1.2, 3.0, 2.0], [5.0, 1.0, 4.0]])
    assert np.allclose(calibrated_uwe_dr2(g, w), sd.predicted_uwe(g, w, bright, "dr2", faint), rtol=1e-12)
    assert calibrated_uwe_dr2(g, w).shape == (2, 3) and np.asarray(orbit_factor_dr2(12.0)).shape == ()


@pytest.mark.torch_cpu
def test_the_documented_call_gives_the_dr2_ruwe_of_the_scoring_scripts_prediction():
    """The call of the README: a star through simulate_fit with uwe_factor=1, then calibrated_uwe_dr2(G, uwe of the DR2 window) over u0_dr2(G, bp_rp), against the scoring script's prediction for the same inputs and seed."""
    import scoreboard_dr2 as sd
    import wobbler as wb
    from wobbler.ruwe import u0_dr2

    rng = np.random.default_rng(1)
    scan = wb.Scan(jd=np.sort(rng.uniform(2456891.5, 2457531.5, 60)), psi=rng.uniform(0, np.pi, 60), plx_factor=rng.uniform(-1, 1, 60), device="cpu")
    orbit = wb.Orbit(period_days=300.0, ecc=0.2, t_peri_days=-50.0, omega=1.0)
    bands = json.loads(TABLE.read_text())["bands"]
    bright, faint = [b["B"] for b in bands[:16]], [b["B"] for b in bands[16:]]
    for G in (11.0, 13.6):
        fit = wb.simulate_fit(scan, orbit, a1_au=0.5, parallax_mas=np.array([2.0]), G=G, n_orient=20, seed=5, uwe_factor=1.0, device="cpu")
        uwe = fit.windows["DR2"]["uwe"]
        ruwe = calibrated_uwe_dr2(G, uwe) / u0_dr2(G, 1.0)
        expected = sd.predicted_uwe(G, np.asarray(uwe), bright, "dr2", faint) / float(u0_dr2(G, 1.0))
        ok = fit.windows["DR2"]["ok"].numpy()
        assert np.shape(ruwe) == (1, 20) and ok.sum() >= 10
        assert np.allclose(np.asarray(ruwe)[ok], expected[ok], rtol=1e-12, atol=0)
        assert np.all(np.asarray(ruwe)[ok] > 0.5)
