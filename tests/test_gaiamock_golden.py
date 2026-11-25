"""wobbler against numbers recorded from gaiamock 1.0.0 for constant inputs (tests/golden/record_gaiamock.py), so that the conventions (which of node and omega is which, east and north, the sign of the along-scan projection, the epoch error table) are checked without gaiamock installed."""
import json
from pathlib import Path

import numpy as np
import pytest

from wobbler import epoch_error, kepler, photocentre, thiele_innes

pytestmark = pytest.mark.torch_cpu

GOLDEN = json.loads((Path(__file__).parent / "golden" / "gaiamock_golden.json").read_text())
T_YR, PSI = np.array(GOLDEN["t_yr"]), np.array(GOLDEN["psi"])


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: f"P{c['period']}")
def test_the_orbit_agrees_with_gaiamock(case):
    constants = thiele_innes(case["a0"], case["node"], case["omega"], case["inclination"])
    assert np.allclose(constants, case["thiele_innes"], rtol=1e-12, atol=1e-12)
    mean = 2 * np.pi / case["period"] * (T_YR * 365.25 - case["t_peri"])
    E = kepler(mean, case["ecc"])
    assert np.allclose(np.sin(E), case["sin_E"], atol=1e-8) and np.allclose(np.cos(E), case["cos_E"], atol=1e-8)
    x, y = photocentre(T_YR, case["period"], case["ecc"], case["t_peri"], case["a0"], case["inclination"], case["node"], case["omega"])
    along_scan = y * np.cos(PSI) + x * np.sin(PSI)
    assert np.allclose(along_scan, case["delta_eta"], atol=1e-8), np.max(np.abs(along_scan - case["delta_eta"]))


def test_the_epoch_error_agrees_with_gaiamock_for_a_faint_and_two_bright_stars():
    assert np.allclose([epoch_error(g) for g in GOLDEN["G"]], GOLDEN["epoch_error"], rtol=1e-12)
    assert np.allclose(epoch_error(np.array(GOLDEN["G"])), GOLDEN["epoch_error"], rtol=1e-12)
