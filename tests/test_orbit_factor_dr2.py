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
