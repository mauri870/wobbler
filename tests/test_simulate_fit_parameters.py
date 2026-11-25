"""simulate_fit(parameters=True) returns the fitted proper motion and the formal errors of the five parameters beside what it always returns, and changes nothing else."""
import numpy as np
import pytest
import torch

from wobbler import Orbit, Scan, simulate_fit
from wobbler import gaiafit

pytestmark = pytest.mark.torch_cpu

EXTRA = ("pmra_mas_yr", "pmdec_mas_yr", "parallax_error_mas", "pmra_error_mas_yr", "pmdec_error_mas_yr")
ORBIT = Orbit(period_days=185.59, ecc=0.45, t_peri_days=-1.1, omega=0.22)


def scan(epochs=70, seed=3):
    rng = np.random.default_rng(seed)
    return Scan(jd=np.sort(rng.uniform(2456891.5, 2457902.0, epochs)), psi=rng.uniform(0, np.pi, epochs), plx_factor=rng.uniform(-1, 1, epochs), device="cpu")


def run(**kwargs):
    return simulate_fit(scan(), ORBIT, a1_au=1.0, parallax_mas=np.array([2.0, 3.0]), G=13.0, n_orient=4, seed=5, nu_eff=1.5, device="cpu", uwe_factor=1.0, **kwargs)


def test_the_default_result_has_none_of_the_extra_keys_and_the_keyword_leaves_the_rest_unchanged():
    plain, extended = run().windows["DR3"], run(parameters=True).windows["DR3"]
    assert not set(EXTRA) & set(plain)
    assert set(EXTRA) <= set(extended)
    for key in plain:
        assert torch.equal(plain[key], extended[key]), key
    assert all(extended[key].shape == (2, 4) for key in EXTRA)


def test_the_extra_values_are_those_of_a_direct_fit_of_the_same_draws():
    """The differential test: the same epochs, the same noise (noise-free here) and the same errors through fit_five_parameter."""
    sc = scan()
    cases, epochs = 3, len(sc)
    noise = {"keep": np.ones((cases, epochs), dtype=bool), "z_err": np.zeros((cases, epochs)), "z_extra": np.zeros((cases, epochs)), "extra": np.zeros(cases)}
    sim = simulate_fit(sc, ORBIT, a1_au=1.0, parallax_mas=np.array([2.0]), G=13.0, n_orient=cases, noise=noise, pm=(7.0, -4.0), seed=1, device="cpu", uwe_factor=1.0, parameters=True).windows["DR3"]
    error = gaiafit.epoch_error(np.full(cases, 13.0), device="cpu").numpy()[:, None]
    observed = gaiafit.observe(sc, torch.full((cases,), 2.0, dtype=torch.float64), 7.0, -4.0, torch.zeros(cases, dtype=torch.float64), torch.zeros(cases, dtype=torch.float64), torch.zeros(cases, dtype=torch.float64),
                               ORBIT.period_days, ORBIT.ecc, ORBIT.t_peri_days, ORBIT.omega, error, torch.zeros(cases, epochs, dtype=torch.float64), torch.zeros(cases, dtype=torch.float64), torch.zeros(cases, epochs, dtype=torch.float64), device="cpu")
    # with a zero orbit (a0 = 0) the fit returns the barycentre's motion and parallax exactly, and the errors are those of the direct fit
    a0_zero = simulate_fit(sc, ORBIT, a1_au=0.0, parallax_mas=np.array([2.0]), G=13.0, n_orient=cases, noise=noise, pm=(7.0, -4.0), seed=1, device="cpu", uwe_factor=1.0, parameters=True).windows["DR3"]
    np.testing.assert_allclose(a0_zero["pmra_mas_yr"].numpy(), 7.0, atol=1e-8)
    np.testing.assert_allclose(a0_zero["pmdec_mas_yr"].numpy(), -4.0, atol=1e-8)
    np.testing.assert_allclose(a0_zero["parallax_mas"].numpy(), 2.0, atol=1e-8)
    direct = gaiafit.fit_five_parameter(sc.t_yr(), sc.psi, sc.plx_factor, observed, error, keep=torch.as_tensor(noise["keep"]), errors=True, device="cpu")
    errors = direct["parameter_errors"].numpy()
    np.testing.assert_allclose(a0_zero["pmra_error_mas_yr"].numpy().reshape(-1), errors[:, 1], rtol=1e-12)
    np.testing.assert_allclose(a0_zero["pmdec_error_mas_yr"].numpy().reshape(-1), errors[:, 3], rtol=1e-12)
    np.testing.assert_allclose(a0_zero["parallax_error_mas"].numpy().reshape(-1), errors[:, 4], rtol=1e-12)
    assert (a0_zero["parallax_error_mas"] > 0).all() and sim["pmra_mas_yr"].shape == (1, cases)
