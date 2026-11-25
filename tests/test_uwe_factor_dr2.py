"""The single-star factor of Gaia DR2 (wobbler.uwe_factor.uwe_factor_dr2): the table, its interpolation and its use beside the DR3 factor.
Function tests on the table; the build and the held-out check are in validation/dr2/."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from wobbler.uwe_factor import uwe_factor, uwe_factor_dr2

TABLE = Path(__file__).resolve().parents[1] / "src" / "wobbler" / "tables" / "uwe_factor_dr2.json"


def test_the_table_hash_is_the_one_of_its_bands():
    meta = json.loads(TABLE.read_text())
    check = json.loads((TABLE.parents[3] / "validation" / "dr2" / "factor_dr2_check.json").read_text())["bands"]
    assert hashlib.sha256(json.dumps(check, sort_keys=True).encode()).hexdigest() == meta["bands_sha256"]


def test_the_factor_equals_the_node_value_at_the_node_position_and_interpolates_linearly_between_nodes():
    bands = json.loads(TABLE.read_text())["bands"]
    centres = np.array([b.get("G_node", 0.5 * (b["G"][0] + b["G"][1])) for b in bands])
    values = np.array([b["f"] for b in bands])
    assert np.allclose(uwe_factor_dr2(centres), values, rtol=0, atol=1e-12)
    middle = 0.5 * (centres[20] + centres[21])
    assert float(uwe_factor_dr2(middle)) == pytest.approx(0.5 * (values[20] + values[21]), rel=1e-12)


def test_the_factor_holds_its_end_values_outside_the_bands():
    bands = json.loads(TABLE.read_text())["bands"]
    assert float(uwe_factor_dr2(1.0)) == pytest.approx(bands[0]["f"]) and float(uwe_factor_dr2(25.0)) == pytest.approx(bands[-1]["f"])


def test_the_dr2_factor_is_not_the_dr3_factor():
    """DR2's unit weight error of an ordinary star differs from DR3's: at G 11.5 the DR3 factor is 1.7 and the DR2 factor 1.3, and at G 14 DR2's is under 1 (the bands of validation/dr2/factor_dr2_check.json)."""
    assert float(uwe_factor(11.5)) > 1.5 > float(uwe_factor_dr2(11.5)) > 1.2
    assert float(uwe_factor_dr2(14.0)) < 0.9


def test_a_number_gives_a_zero_dimensional_array_and_an_array_keeps_its_shape():
    assert np.asarray(uwe_factor_dr2(12.0)).shape == () and uwe_factor_dr2(np.array([[10.0, 12.0], [14.0, 16.0]])).shape == (2, 2)


def test_a_node_with_a_position_is_placed_there_and_not_at_its_band_centre(tmp_path):
    import importlib
    module = importlib.import_module("wobbler.uwe_factor")
    table = tmp_path / "t.json"
    table.write_text(json.dumps({"bands": [{"G": [13.0, 13.5], "G_node": 13.01, "f": 1.0}, {"G": [13.5, 14.0], "G_node": 13.51, "f": 0.5}]}))
    centres, values = module._nodes.__wrapped__(table)
    assert list(centres) == [13.01, 13.51] and list(values) == [1.0, 0.5]


def test_the_nodes_of_the_table_are_in_increasing_order_of_position():
    bands = json.loads(TABLE.read_text())["bands"]
    positions = np.array([b.get("G_node", 0.5 * (b["G"][0] + b["G"][1])) for b in bands])
    assert (np.diff(positions) > 0).all()
