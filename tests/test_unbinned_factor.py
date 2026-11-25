"""The single-star factor of the unbinned mode (wobbler.unbinned_factor, fit_unbinned_factor), its table, its held-out check on a recorded sample of Gaia DR3 single stars and its use as the default of simulate_fit(binned=False).
Function tests: the checks are on a table, on a recorded sample of 19,600 stars and on the arithmetic of simulate_fit, not on a light curve; a txtar archive has no place for an array of that size."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import stats

import wobbler as wb
from wobbler.uwe_factor import UNBINNED_EDGES, fit_unbinned_factor, unbinned_factor

from test_unbinned import make_scan, synthetic_map

pytestmark = pytest.mark.torch_cpu
HERE = Path(__file__).resolve().parent
TABLE = HERE.parent / "src" / "wobbler" / "tables" / "uwe_factor_unbinned_dr3.json"
SAMPLE = HERE / "golden" / "single_stars_dr3.csv.gz"


def recorded():
    with gzip.open(SAMPLE, "rt") as handle:
        handle.readline()
        rows = [line.strip().split(",") for line in handle]
    part = np.array([r[0] for r in rows])
    values = np.array([[float(x) for x in r[1:]] for r in rows])
    return part, values[:, 0], values[:, 1], values[:, 2], values[:, 3]


def test_the_table_has_the_hash_of_its_bands_no_gap_and_the_factor_at_a_centre_is_the_value_of_its_band():
    meta = json.loads(TABLE.read_text())
    assert hashlib.sha256(json.dumps(meta["bands"], sort_keys=True, separators=(",", ":")).encode()).hexdigest() == meta["bands_sha256"]
    assert [b["G"][0] for b in meta["bands"]] == list(UNBINNED_EDGES[:-1]) and meta["bands"][-1]["G"][1] == 16.0
    assert all(b["n_archive"] == 700 for b in meta["bands"])
    for band in meta["bands"]:
        assert float(unbinned_factor(0.5 * (band["G"][0] + band["G"][1]))) == pytest.approx(band["f"], rel=1e-12)


def test_the_factor_holds_its_end_values_outside_the_sample_and_takes_arrays():
    bands = json.loads(TABLE.read_text())["bands"]
    assert float(unbinned_factor(5.0)) == pytest.approx(bands[0]["f"]) and float(unbinned_factor(19.0)) == pytest.approx(bands[-1]["f"])
    out = unbinned_factor(np.array([[9.0, 11.3], [13.0, 16.0]]))
    assert out.shape == (2, 2) and bool((out > 0.9).all())


def test_the_table_is_the_fit_of_the_recorded_sample():
    _, g, archive, _, raw = recorded()
    expected = [b["f"] for b in json.loads(TABLE.read_text())["bands"]]
    assert fit_unbinned_factor(g, archive, raw) == pytest.approx(expected, abs=5e-6)  # the fixture rounds to 6 decimals


def test_held_out_the_factor_makes_the_unbinned_mode_closer_to_the_archive_than_the_binned_mode_in_every_band():
    """Fitted on the half of the recorded stars marked fit, tested on the other half: the numbers of the table's check (D 0.097 against 0.237 on the test half)."""
    part, g, archive, binned, raw = recorded()
    fit, test = part == "fit", part == "test"
    centres = np.array([0.5 * (a + b) for a, b in zip(UNBINNED_EDGES[:-1], UNBINNED_EDGES[1:])])
    held = np.interp(g, centres, fit_unbinned_factor(g[fit], archive[fit], raw[fit])) * raw
    pooled_unbinned, pooled_binned = stats.ks_2samp(archive[test], held[test]).statistic, stats.ks_2samp(archive[test], binned[test]).statistic
    assert pooled_unbinned == pytest.approx(0.097, abs=0.005) and pooled_binned == pytest.approx(0.237, abs=0.005), (pooled_unbinned, pooled_binned)
    closer = 0
    for low, high in zip(UNBINNED_EDGES[:-1], UNBINNED_EDGES[1:]):
        band = test & (g >= low) & (g < high)
        closer += stats.ks_2samp(archive[band], held[band]).statistic < stats.ks_2samp(archive[band], binned[band]).statistic
    assert closer == 28, f"closer in {closer} of 28 bands"


def test_simulate_fit_unbinned_applies_the_factor_by_default_and_uwe_factor_1_gives_the_raw_value(tmp_path):
    path, _ = synthetic_map(tmp_path, 6)
    scan, orbit = make_scan(6, 600), wb.Orbit(period_days=185.0, ecc=0.45, t_peri_days=-1.1, omega=0.22)
    kw = dict(a1_au=1.0, parallax_mas=np.array([2.0]), G=11.3, n_orient=5, binned=False, position=(200.0, 30.0), sky_map=path, seed=2, device="cpu", windows={"all": (0, 1e9)})
    default, raw = wb.simulate_fit(scan, orbit, **kw), wb.simulate_fit(scan, orbit, uwe_factor=1, **kw)
    factor = float(unbinned_factor(11.3))
    assert factor > 1.5
    assert torch.allclose(default.uwe, raw.uwe * factor, rtol=1e-12), "the default multiplies the raw value by the factor of the star's G"
