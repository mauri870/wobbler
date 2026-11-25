"""The RUWE normalisation u0(G, nu_eff) of Gaia DR3 (Lindegren 2023, RUWE_DR3.zip, table_u0_g_c_p5.txt) and its use in simulate_fit: ruwe = uwe / u0, with the unit weight error `uwe` that gaiamock calls RUWE; and the factor between the formal and the real epoch error. The table is package data with its source and hash.
Function tests: the checks are on a table, on the archive rows of three stars and on arrays of random inputs, not on a light curve."""
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import wobbler as wb
from wobbler import ruwe as ruwe_module

def git(directory, *arguments):
    return subprocess.run(["git", "-C", str(directory), *arguments], check=True, capture_output=True, text=True).stdout.strip()

pytestmark = pytest.mark.torch_cpu
DATA = Path(__file__).resolve().parents[1] / "src" / "wobbler" / "tables"
ORBIT = wb.Orbit(period_days=325.006, ecc=0.2103, t_peri_days=-110.37, omega=1.1048)
# the archive rows of Gaia DR3 for Gaia BH3, Gaia BH1 and a third star, read from gaiadr3.gaia_source by its source id (columns: phot_g_mean_mag, nu_eff_used_in_astrometry, and u0 = sqrt(astrometric_chi2_al / (astrometric_n_good_obs_al - 5)) / ruwe)
ARCHIVE = {"BH3": (11.231088, 1.4755152, 1.8127571302563215), "BH1": (13.771998, 1.4811925, 1.0019748885909814), "a star of G 12.57": (12.573492, 1.4598968, 1.1984160903511196)}


def table_rows():
    with gzip.open(DATA / "u0_dr3_p5.csv.gz", "rt") as handle:
        return np.loadtxt(handle, delimiter=",", skiprows=1)


@pytest.mark.parametrize("star", list(ARCHIVE))
def test_u0_reproduces_the_archive_rows_of_three_stars_to_0_2_percent(star):
    g, nu_eff, expected = ARCHIVE[star]
    assert float(ruwe_module.u0(g, nu_eff, device="cpu")) == pytest.approx(expected, rel=2e-3)


def test_u0_is_the_table_at_its_nodes_bilinear_between_them_and_the_nearest_value_outside():
    rows = table_rows()
    for index in (0, 1, 5000, len(rows) // 2, len(rows) - 1):
        g, c, value = rows[index]
        assert float(ruwe_module.u0(g, c, device="cpu")) == pytest.approx(value, rel=1e-12)
    corner = {(g, c): v for g, c, v in rows if g in (11.00, 11.02) and c in (1.40, 1.41)}
    assert len(corner) == 4
    expected = np.mean(list(corner.values()))
    assert float(ruwe_module.u0(11.01, 1.405, device="cpu")) == pytest.approx(expected, rel=1e-12)
    assert float(ruwe_module.u0(2.0, 1.5, device="cpu")) == pytest.approx(float(ruwe_module.u0(4.0, 1.5, device="cpu")), rel=1e-14), "brighter than the table: its first magnitude"
    assert float(ruwe_module.u0(25.0, 3.0, device="cpu")) == pytest.approx(float(ruwe_module.u0(21.0, 2.0, device="cpu")), rel=1e-14), "fainter and redder: the last node"


def test_u0_takes_arrays_and_returns_a_tensor_of_the_same_shape():
    out = ruwe_module.u0(np.array([[11.0, 12.0], [13.0, 14.0]]), 1.45, device="cpu")
    assert tuple(out.shape) == (2, 2) and bool((out > 0.9).all())


def test_the_table_in_the_package_has_the_hash_of_its_source_recorded_with_its_provenance():
    meta = json.loads((DATA / "u0_dr3_p5.json").read_text())
    assert meta["zip_sha256"] == "37cecf81826f82cca890c8f83a58dc22f71cb95560972071bbc628ce343eacb5"
    with gzip.open(DATA / "u0_dr3_p5.csv.gz", "rb") as handle:
        assert hashlib.sha256(handle.read()).hexdigest() == meta["member_sha256"]
    assert "cosmos.esa.int" in meta["source"] and meta["member"] == "table_u0_g_c_p5.txt"


def test_simulate_fit_gives_ruwe_as_uwe_over_u0_with_nu_eff_and_no_ruwe_without_it():
    scan = wb.Scan(np.linspace(2456900.0, 2457800.0, 120), np.linspace(0.1, 3.0, 120), np.cos(np.linspace(0, 20, 120)), device="cpu")
    kwargs = dict(a1_au=1.2, parallax_mas=np.array([2.0, 3.0]), G=np.array([11.231088, 13.771998]), n_orient=6, seed=1, device="cpu")
    plain = wb.simulate_fit(scan, ORBIT, **kwargs)
    normed = wb.simulate_fit(scan, ORBIT, nu_eff=np.array([1.4755152, 1.4811925]), **kwargs)
    with pytest.raises(AttributeError, match="nu_eff"):
        plain.ruwe
    for name in plain.windows:
        window, other = plain.windows[name], normed.windows[name]
        assert "ruwe" not in window and "uwe" in window
        assert np.allclose(other["uwe"].numpy(), window["uwe"].numpy(), equal_nan=True), "the unit weight error does not depend on the normalisation"
        u0 = ruwe_module.u0(np.array([11.231088, 13.771998]), np.array([1.4755152, 1.4811925]), device="cpu").numpy()
        assert np.allclose(other["ruwe"].numpy(), other["uwe"].numpy() / u0[:, None], equal_nan=True)


def test_the_formal_error_factor_scales_the_unit_weight_error_of_a_single_star_and_one_is_the_raw_behaviour():
    scan = wb.Scan(np.linspace(2456900.0, 2457800.0, 150), np.linspace(0.1, 3.0, 150), np.cos(np.linspace(0, 20, 150)), device="cpu")
    kwargs = dict(a1_au=0.0, parallax_mas=np.array([2.0]), G=15.0, n_orient=2000, seed=3, device="cpu", windows={"DR3": (2456891.5, 2457902.0)}, uwe_factor=1.0)
    base = wb.simulate_fit(scan, ORBIT, **kwargs)
    unit = wb.simulate_fit(scan, ORBIT, formal_error_factor=1.0, **kwargs)
    assert np.array_equal(base.windows["DR3"]["uwe"].numpy(), unit.windows["DR3"]["uwe"].numpy(), equal_nan=True)
    scaled = wb.simulate_fit(scan, ORBIT, formal_error_factor=1.5, **kwargs)
    n = base.windows["DR3"]["n_epochs"].numpy().astype(float)
    uwe1, uwe_f = base.windows["DR3"]["uwe"].numpy(), scaled.windows["DR3"]["uwe"].numpy()
    # the residuals are the same (a uniform weight leaves the solution alone) and the reduced chi-square rises by f^2: uwe^2 = (7 n + chi2_reduced (n - 5)) / (8 n - 5) of the 8 CCD observations of an epoch
    expected = np.sqrt((7 * n + 1.5**2 * (uwe1**2 * (8 * n - 5) - 7 * n)) / (8 * n - 5))
    assert np.allclose(uwe_f, expected, rtol=1e-9, equal_nan=True)
    assert np.nanmean(uwe_f) > np.nanmean(uwe1) * 1.05, "the unit weight error of a single star rises with the factor"


def test_the_fit_divides_its_unit_weight_error_by_the_u0_it_is_given_case_by_case():
    rng = np.random.default_rng(8)
    t = np.linspace(-1.2, 1.4, 90)
    psi, plx = rng.uniform(0, np.pi, 90), rng.uniform(-0.9, 0.9, 90)
    observed = rng.normal(0, 0.1, (5, 90))
    u0 = np.array([1.0, 1.2, 1.5, 2.0, 0.9])
    fit = wb.fit_five_parameter(t, psi, plx, observed, 0.1, u0=u0, device="cpu")
    plain = wb.fit_five_parameter(t, psi, plx, observed, 0.1, device="cpu")
    assert np.allclose(fit["ruwe"].numpy(), plain["uwe"].numpy() / u0) and np.array_equal(plain["ruwe"].numpy(), plain["uwe"].numpy())


def test_every_table_file_of_the_package_is_tracked_by_git_and_not_ignored():
    # The u0 table of the first merge sat in a directory that .gitignore excludes, passed the checks in its own worktree and was absent from master.
    root = DATA.parents[2]
    if not (root / ".git").exists():
        pytest.skip("not a git checkout (a source distribution has no repository)")
    for path in sorted(DATA.iterdir()):
        try:
            git(root, "ls-files", "--error-unmatch", str(path))
        except subprocess.CalledProcessError:
            pytest.fail(f"{path} is not tracked by git (ignored by .gitignore?)")
