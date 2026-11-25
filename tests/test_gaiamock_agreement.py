"""wobbler against gaiamock (El-Badry et al. 2024, https://github.com/kareemelbadry/gaiamock) on identical inputs: the abscissae are built by wobbler.observe from constructed orbits and noise, and gaiamock's fit_5par_solution_only and solve_kepler_eqn_on_array run on the same numbers. Skipped where gaiamock is not installed."""
import numpy as np
import pytest

gaiamock = pytest.importorskip("gaiamock")

from wobbler import fit_five_parameter, kepler, observe, scan  # noqa: E402

pytestmark = pytest.mark.torch_cpu


def make_cases(seed, cases=40, n=190):
    rng = np.random.default_rng(seed)
    jd = np.sort(rng.uniform(2456891.5, 2457902.0, n))
    s = scan.Scan(jd=jd, psi=rng.uniform(0, np.pi, n), plx_factor=rng.uniform(-0.9, 0.9, n))
    parameters = dict(parallax=rng.uniform(0.5, 4, cases), pmra=rng.normal(0, 5, cases), pmdec=rng.normal(0, 5, cases), a0_mas=rng.uniform(0, 4, cases), inclination=np.arccos(rng.uniform(0, 1, cases)), node=rng.uniform(0, 2 * np.pi, cases),
                      period_days=rng.uniform(100, 900, cases), ecc=rng.uniform(0, 0.8, cases), t_peri_days=rng.uniform(-400, 400, cases), omega=rng.uniform(0, 2 * np.pi, cases))
    return s, parameters, rng.normal(size=(cases, n)), rng.normal(size=(cases, n)), rng.uniform(0, 0.04, cases), rng.uniform(size=(cases, n)) > 0.1


def test_kepler_agrees_with_gaiamocks_compiled_solver_in_sine_and_cosine():
    """gaiamock returns the eccentric anomaly wrapped into one revolution while wobbler returns the one that carries the mean anomaly's revolutions; the orbit uses sin E and cos E, which agree to the solver's tolerance."""
    c_funcs = gaiamock.read_in_C_functions()
    rng = np.random.default_rng(1)
    for ecc in (0.0, 0.21, 0.6, 0.9):
        mean = rng.uniform(-10, 10, 50)
        ours, theirs = kepler(mean, ecc, device="cpu").numpy(), gaiamock.solve_kepler_eqn_on_array(mean, ecc, c_funcs)
        assert np.max(np.abs(np.sin(ours) - np.sin(theirs))) < 1e-8 and np.max(np.abs(np.cos(ours) - np.cos(theirs))) < 1e-8, ecc


def test_the_five_parameter_fit_agrees_with_gaiamock_case_by_case_with_masks():
    s, parameters, z_err, z_extra, extra, keep = make_cases(2)
    error = gaiamock.al_uncertainty_per_ccd_interp(13.0) / np.sqrt(8)
    observed = observe(s, error=error, z_err=z_err, extra=extra, z_extra=z_extra, device="cpu", **parameters)
    t = s.t_yr()
    batched = {name: value.numpy() if hasattr(value, "numpy") else value for name, value in fit_five_parameter(t, s.psi, s.plx_factor, observed, error, keep=keep, device="cpu").items()}
    observed, t, psi, plx = observed.numpy(), t.numpy(), s.psi.numpy(), s.plx_factor.numpy()
    for i in range(len(observed)):
        k = keep[i]
        expected = gaiamock.fit_5par_solution_only(t_ast_yr=t[k], psi=psi[k], plx_factor=plx[k], ast_obs=observed[i, k], ast_err=error * np.ones(k.sum()))
        assert batched["ok"][i] == (expected[0] != 0 or expected[4] != 0)
        if batched["ok"][i]:
            assert np.allclose(batched["parameters"][i], expected[:5], rtol=1e-9, atol=1e-12), i
            assert batched["ruwe"][i] == pytest.approx(expected[10], rel=1e-10), i
            assert batched["parameter_errors"][i, 4] == pytest.approx(expected[9], rel=1e-10), i
