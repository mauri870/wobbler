"""The unbinned mode of gaiamock (https://github.com/kareemelbadry/gaiamock, README section "A modified version to predict RUWE more reliably for small orbits and single stars"): one measurement per CCD, the epoch errors rescaled by the median RUWE of the sky position, and a fit whose reduced chi-square is not corrected for binning. Each test compares with gaiamock on identical inputs or checks a case that would hide a failure."""
import numpy as np
import pytest
import torch

from wobbler import Scan, fit_five_parameter, observe, simulate_fit
from wobbler import unbinned
from wobbler.orbit import Orbit

pytestmark = pytest.mark.torch_cpu


def synthetic_map(tmp_path, seed=0):
    """A stand-in for healpix_16_med_ruwe.npz: 3,072 values of the median RUWE around 1."""
    values = np.random.default_rng(seed).uniform(0.9, 1.3, 12 * 16**2)
    path = tmp_path / "healpix_16_med_ruwe.npz"
    np.savez(path, med_ruwe=values)
    return path, values


def make_scan(seed, n=1400):
    rng = np.random.default_rng(seed)
    return Scan(jd=np.sort(rng.uniform(2456891.5, 2457902.0, n)), psi=rng.uniform(0, np.pi, n), plx_factor=rng.uniform(-0.9, 0.9, n))


def test_the_ring_pixel_of_a_position_agrees_with_healpy_including_the_poles_and_the_equatorial_boundary():
    healpy = pytest.importorskip("healpy")
    rng = np.random.default_rng(1)
    ra, dec = rng.uniform(0, 360, 20_000), np.degrees(np.arcsin(rng.uniform(-1, 1, 20_000)))
    edge = np.degrees(np.arcsin(np.array([2 / 3, -2 / 3, 1.0, -1.0, 0.0, 2 / 3 + 1e-9, -2 / 3 - 1e-9])))
    ra, dec = np.concatenate([ra, np.linspace(0, 359.9, edge.size)]), np.concatenate([dec, edge])
    expected = healpy.ang2pix(16, np.radians(90.0 - dec), np.radians(ra), nest=False)
    assert np.array_equal(unbinned.ang2pix_ring(16, ra, dec), expected)


def test_the_unbinned_fit_agrees_with_gaiamock_at_fixed_abscissae_to_1e_11():
    gaiamock = pytest.importorskip("gaiamock")
    rng = np.random.default_rng(2)
    scan, cases = make_scan(2), 30
    n = len(scan)
    parameters = dict(parallax=rng.uniform(0.5, 4, cases), pmra=rng.normal(0, 5, cases), pmdec=rng.normal(0, 5, cases), a0_mas=rng.uniform(0, 4, cases), inclination=np.arccos(rng.uniform(0, 1, cases)), node=rng.uniform(0, 2 * np.pi, cases),
                      period_days=rng.uniform(100, 900, cases), ecc=rng.uniform(0, 0.8, cases), t_peri_days=rng.uniform(-400, 400, cases), omega=rng.uniform(0, 2 * np.pi, cases))
    error = 0.13
    observed = observe(scan, error=error * 1.1, z_err=rng.normal(size=(cases, n)), extra=np.zeros(cases), z_extra=np.zeros((cases, n)), device="cpu", **parameters)
    keep = rng.uniform(size=(cases, n)) > 0.1
    t = scan.t_yr()
    batched = {name: value.numpy() if hasattr(value, "numpy") else value for name, value in fit_five_parameter(t, scan.psi, scan.plx_factor, observed, error, keep=keep, binned=False, device="cpu").items()}
    observed, t, psi, plx = observed.numpy(), t.numpy(), scan.psi.numpy(), scan.plx_factor.numpy()
    worst = 0.0
    for i in range(cases):
        k = keep[i]
        expected = gaiamock.fit_5par_solution_only(t_ast_yr=t[k], psi=psi[k], plx_factor=plx[k], ast_obs=observed[i, k], ast_err=error * np.ones(k.sum()), binned=False)
        assert batched["ok"][i]
        assert batched["ruwe"][i] == pytest.approx(expected[10], rel=1e-11), i
        assert batched["parameter_errors"][i, 4] == pytest.approx(expected[9], rel=1e-11), i
        assert np.allclose(batched["parameters"][i], expected[:5], rtol=1e-11, atol=1e-13), i
        worst = max(worst, abs(batched["ruwe"][i] / expected[10] - 1))
    assert worst < 1e-11, f"the largest relative difference of the RUWE is {worst:.2e}"


def test_the_unbinned_reduced_chi_square_is_not_corrected_and_differs_from_the_binned_one():
    scan, rng = make_scan(3, 200), np.random.default_rng(3)
    observed = rng.normal(size=(5, len(scan))) * 0.2
    binned = fit_five_parameter(scan.t_yr(), scan.psi, scan.plx_factor, observed, 0.1, device="cpu")
    plain = fit_five_parameter(scan.t_yr(), scan.psi, scan.plx_factor, observed, 0.1, binned=False, device="cpu")
    reduced = plain["chi2"] / plain["degrees_of_freedom"]
    assert torch.allclose(plain["uwe"], reduced.sqrt(), rtol=1e-13)
    assert (binned["uwe"] != plain["uwe"]).all() and (binned["uwe"] < plain["uwe"]).all(), "binning a poor fit lowers the corrected chi-square"


def test_the_realistic_error_agrees_with_gaiamock_for_a_map_and_a_draw(tmp_path, monkeypatch):
    gaiamock = pytest.importorskip("gaiamock")
    path, values = synthetic_map(tmp_path)
    monkeypatch.setattr(gaiamock.gaiamock, "data_path", lambda *parts: str(path))
    ccd = gaiamock.al_uncertainty_per_ccd_interp(12.3)
    for ra, dec in ((10.0, 20.0), (266.4, -28.9), (359.99, 89.9), (0.0, -89.9)):
        np.random.seed(7)
        expected = gaiamock.get_realistic_epoch_astrometry_errors(ra, dec, 12.3)
        jitter = np.random.RandomState(7).randn()
        ours = unbinned.realistic_error(ccd, ra, dec, jitter, unbinned.read_sky_map(path))
        assert ours == pytest.approx(expected, rel=1e-14, abs=1e-15), (ra, dec)
        assert ours - ccd - 0.01 * jitter == pytest.approx((values[unbinned.ang2pix_ring(16, ra, dec)] - 1) * 0.125, abs=1e-14), "the map term is the pixel's median RUWE minus 1, times 0.125"


def test_the_noise_and_abscissae_of_a_whole_star_agree_with_gaiamock_at_the_same_draws(tmp_path, monkeypatch):
    """gaiamock.predict_astrometry_binary_in_terms_of_a0(binned=False) on a synthetic table of CCD epochs, against wobbler.observe given the draws that gaiamock makes in the order it makes them (the rejection, the error's jitter, the noise)."""
    gaiamock = pytest.importorskip("gaiamock")
    from astropy.table import Table

    path, _ = synthetic_map(tmp_path, 4)
    scan = make_scan(5, 900)
    table = Table({"ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]": scan.jd.numpy(), "scanAngle[rad]": scan.psi.numpy(), "parallaxFactorAlongScan": scan.plx_factor.numpy()})
    monkeypatch.setattr(gaiamock.gaiamock, "get_gost_one_position", lambda ra, dec, data_release, binned=True: table)
    monkeypatch.setattr(gaiamock.gaiamock, "data_path", lambda *parts: str(path))
    ra, dec, G = 123.4, -33.3, 13.7
    kw = dict(parallax=2.0, pmra=3.0, pmdec=-4.0, period=300.0, Tp=-20.0, ecc=0.4, omega=1.1, inc=1.0, w=0.7, a0_mas=1.5)
    np.random.seed(11)
    t, psi, plx, observed, error = gaiamock.predict_astrometry_binary_in_terms_of_a0(ra=ra, dec=dec, phot_g_mean_mag=G, data_release="dr3", c_funcs=gaiamock.read_in_C_functions(), binned=False, **{k: v for k, v in kw.items()})
    draws = np.random.RandomState(11)
    keep = draws.uniform(0, 1, len(scan)) > 0.1
    jitter = draws.randn()
    z = draws.randn(keep.sum())
    assert keep.sum() == len(t)
    ccd = gaiamock.al_uncertainty_per_ccd_interp(G)
    noise = unbinned.realistic_error(ccd, ra, dec, jitter, unbinned.read_sky_map(path))
    kept = Scan(jd=scan.jd[keep], psi=scan.psi[keep], plx_factor=scan.plx_factor[keep])
    ours = observe(kept, parallax=kw["parallax"], pmra=kw["pmra"], pmdec=kw["pmdec"], a0_mas=kw["a0_mas"], inclination=kw["inc"], node=kw["omega"], period_days=kw["period"], ecc=kw["ecc"], t_peri_days=kw["Tp"], omega=kw["w"], error=noise, z_err=z[None, :], extra=0.0, z_extra=0.0, device="cpu")
    assert np.allclose(ours.numpy()[0], observed, rtol=0, atol=1e-7), np.max(np.abs(ours.numpy()[0] - observed))
    assert np.allclose(error, ccd), "the returned uncertainties are the expected ones, the per-CCD error"


def test_a_missing_map_names_the_file_and_how_to_fetch_it(tmp_path, monkeypatch):
    monkeypatch.setenv("GAIAMOCK_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    with pytest.raises(FileNotFoundError, match=r"healpix_16_med_ruwe\.npz.*gaiamock-download"):
        unbinned.read_sky_map()


def test_the_map_is_found_in_the_data_directory_of_gaiamock_and_a_wrong_size_is_refused(tmp_path, monkeypatch):
    path, values = synthetic_map(tmp_path)
    monkeypatch.setenv("GAIAMOCK_DATA_DIR", str(tmp_path))
    assert np.array_equal(unbinned.read_sky_map(), values)
    bad = tmp_path / "bad.npz"
    np.savez(bad, med_ruwe=values[:100])
    with pytest.raises(ValueError, match="3072"):
        unbinned.read_sky_map(bad)


def test_simulate_fit_unbinned_needs_a_position_and_is_the_composition_of_its_parts(tmp_path):
    path, _ = synthetic_map(tmp_path, 6)
    scan, orbit = make_scan(6, 600), Orbit(period_days=185.0, ecc=0.45, t_peri_days=-1.1, omega=0.22)
    with pytest.raises(ValueError, match="position"):
        simulate_fit(scan, orbit, a1_au=1.0, parallax_mas=np.array([2.0]), G=13.8, n_orient=4, binned=False, device="cpu")
    cases = 6
    rng = np.random.default_rng(8)
    noise = {"keep": rng.uniform(size=(cases, len(scan))) > 0.1, "z_err": rng.normal(size=(cases, len(scan))), "jitter": rng.normal(size=cases)}
    fit = simulate_fit(scan, orbit, a1_au=1.0, parallax_mas=np.array([2.0]), G=13.8, n_orient=cases, binned=False, uwe_factor=1, position=(200.0, 30.0), sky_map=path, noise=noise, seed=1, device="cpu", windows={"all": (0, 1e9)})
    from wobbler.gaiafit import epoch_error

    ccd = float(epoch_error(13.8, device="cpu")) * 8**0.5
    sky = unbinned.sky_extra_noise(200.0, 30.0, unbinned.read_sky_map(path))
    error = ccd + sky + 0.01 * noise["jitter"]
    observed = observe(scan, parallax=np.full(cases, 2.0), pmra=0.0, pmdec=0.0, a0_mas=fit.a0_mas.numpy().ravel(), inclination=fit.inclination.numpy().ravel(), node=fit.node.numpy().ravel(), period_days=185.0, ecc=0.45, t_peri_days=-1.1, omega=0.22,
                       error=error[:, None], z_err=noise["z_err"], extra=0.0, z_extra=0.0, device="cpu")
    direct = fit_five_parameter(scan.t_yr(), scan.psi, scan.plx_factor, observed, ccd, keep=noise["keep"], binned=False, errors=False, device="cpu")
    assert torch.allclose(fit.windows["all"]["uwe"].reshape(-1), direct["uwe"], rtol=1e-12), "the raw unbinned unit weight error, with no calibration factor"
    assert torch.allclose(fit.windows["all"]["parallax_mas"].reshape(-1), direct["parameters"][:, 4], rtol=1e-12)


def test_the_binned_default_is_unchanged_by_the_flag():
    scan, orbit = make_scan(9, 300), Orbit(period_days=185.0, ecc=0.45, t_peri_days=-1.1, omega=0.22)
    kw = dict(a1_au=1.0, parallax_mas=np.array([2.0]), G=13.8, n_orient=20, seed=3, device="cpu", windows={"all": (0, 1e9)})
    assert torch.equal(simulate_fit(scan, orbit, **kw).uwe, simulate_fit(scan, orbit, binned=True, **kw).uwe)


def test_scan_from_position_asks_gaiamock_for_the_ccd_epochs_when_not_binned(monkeypatch):
    pytest.importorskip("gaiamock")
    import gaiamock.gaiamock as module
    from astropy.table import Table

    seen = []

    def fake(ra, dec, data_release, binned=True):
        seen.append(binned)
        return Table({"ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]": [2457000.0, 2456900.0], "scanAngle[rad]": [0.1, 0.2], "parallaxFactorAlongScan": [0.3, 0.4]})

    monkeypatch.setattr(module, "get_gost_one_position", fake)
    assert len(Scan.from_position(10.0, 20.0, binned=False)) == 2 and seen == [False]
    Scan.from_position(10.0, 20.0)
    assert seen == [False, True]


def test_a_scan_from_position_carries_its_mode_and_position_into_simulate_fit(tmp_path, monkeypatch):
    pytest.importorskip("gaiamock")
    import gaiamock.gaiamock as module
    from astropy.table import Table

    path, _ = synthetic_map(tmp_path, 6)
    monkeypatch.setenv("GAIAMOCK_DATA_DIR", str(tmp_path))
    scan = make_scan(6, 600)
    table = Table({"ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]": scan.jd.numpy(), "scanAngle[rad]": scan.psi.numpy(), "parallaxFactorAlongScan": scan.plx_factor.numpy()})
    monkeypatch.setattr(module, "get_gost_one_position", lambda ra, dec, data_release, binned=True: table)
    orbit = Orbit(period_days=185.0, ecc=0.45, t_peri_days=-1.1, omega=0.22)
    kw = dict(a1_au=1.0, parallax_mas=np.array([2.0]), G=13.8, n_orient=6, seed=1, device="cpu", windows={"all": (0, 1e9)})
    from_position = Scan.from_position(200.0, 30.0, device="cpu", binned=False)
    assert from_position.binned is False and from_position.position == (200.0, 30.0) and from_position.to("cpu").position == (200.0, 30.0)
    implicit = simulate_fit(from_position, orbit, **kw)
    explicit = simulate_fit(from_position, orbit, binned=False, position=(200.0, 30.0), **kw)
    assert torch.equal(implicit.uwe, explicit.uwe)
    binned_default = simulate_fit(Scan.from_position(200.0, 30.0, device="cpu"), orbit, **kw)
    assert not torch.equal(binned_default.uwe, implicit.uwe), "a binned scan keeps the binned model"
