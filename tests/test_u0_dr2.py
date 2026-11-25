"""The normalisation of the unit weight error of Gaia DR2 (wobbler.ruwe.u0_dr2): RUWE = UWE / u0(G, BP-RP) with u0 the table of DR2_RUWE_V1.zip (table_u0_g_col.txt).
Function tests: the checks are on a table and on a recorded sample of DR2 stars with the archive's own ruwe, not on a light curve."""
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from wobbler.ruwe import u0_dr2

pytestmark = pytest.mark.torch_cpu
TABLES = Path(__file__).resolve().parents[1] / "src" / "wobbler" / "tables"
SAMPLE = Path(__file__).resolve().parent / "golden" / "dr2_ruwe_stars.csv.gz"


def table():
    with gzip.open(TABLES / "u0_dr2_p5.csv.gz", "rt") as handle:
        rows = np.loadtxt(handle, delimiter=",", skiprows=1)
    g, c = np.unique(rows[:, 0]), np.unique(rows[:, 1])
    return g, c, rows[:, 2].reshape(len(g), len(c))


def test_the_table_is_the_one_the_sidecar_names():
    meta = json.loads((TABLES / "u0_dr2.json").read_text())
    assert hashlib.sha256((TABLES / "u0_dr2_p5.csv.gz").read_bytes()).hexdigest() == meta["sha256"]
    g, c, grid = table()
    assert (len(g), len(c)) == (1741, 111) and g[0] == pytest.approx(3.60) and g[-1] == pytest.approx(21.00) and c[0] == pytest.approx(-1.0) and c[-1] == pytest.approx(10.0)


def test_u0_dr2_equals_the_table_at_its_nodes():
    g, c, grid = table()
    rng = np.random.default_rng(1)
    i, j = rng.integers(0, len(g), 200), rng.integers(0, len(c), 200)
    assert np.allclose(u0_dr2(g[i], c[j]).numpy(), grid[i, j], rtol=0, atol=1e-12)


def test_u0_dr2_at_the_middle_of_four_nodes_is_their_mean():
    g, c, grid = table()
    i, j = 700, 40
    middle = u0_dr2((g[i] + g[i + 1]) / 2, (c[j] + c[j + 1]) / 2).numpy()
    assert float(middle) == pytest.approx(float(grid[i : i + 2, j : j + 2].mean()), rel=1e-12)


def test_u0_dr2_outside_the_grid_holds_the_value_of_the_edge():
    g, c, grid = table()
    assert float(u0_dr2(2.0, 0.5)) == pytest.approx(float(u0_dr2(g[0], 0.5)))
    assert float(u0_dr2(25.0, 0.5)) == pytest.approx(float(u0_dr2(g[-1], 0.5)))
    assert float(u0_dr2(15.0, -3.0)) == pytest.approx(float(u0_dr2(15.0, c[0])))
    assert float(u0_dr2(15.0, 14.0)) == pytest.approx(float(u0_dr2(15.0, c[-1])))


def test_u0_dr2_broadcasts_a_magnitude_against_a_colour():
    out = u0_dr2(np.array([[12.0], [15.0]]), np.array([0.5, 1.0, 1.5]))
    assert tuple(out.shape) == (2, 3)


def test_the_unit_weight_error_over_u0_dr2_reproduces_the_archives_ruwe_on_1000_dr2_stars():
    """The oracle is the archive: the median absolute relative difference under 0.003 and the 99th percentile under 0.02."""
    with gzip.open(SAMPLE, "rt") as handle:
        rows = list(csv.DictReader(handle))
    g = np.array([float(r["phot_g_mean_mag"]) for r in rows])
    colour = np.array([float(r["bp_rp"]) for r in rows])
    chi2 = np.array([float(r["astrometric_chi2_al"]) for r in rows])
    n_good = np.array([float(r["astrometric_n_good_obs_al"]) for r in rows])
    archive = np.array([float(r["ruwe"]) for r in rows])
    mine = np.sqrt(chi2 / (n_good - 5.0)) / u0_dr2(g, colour).numpy()
    difference = np.abs(mine / archive - 1.0)
    assert np.median(difference) < 0.003 and np.percentile(difference, 99) < 0.02, (np.median(difference), np.percentile(difference, 99), float(difference.max()))
