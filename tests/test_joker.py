"""wobbler.joker: the Joker-method rejection sampler of a single-lined binary's orbit from radial velocities (Price-Whelan et al. 2017) on torch.
The marginal likelihood is compared with the values recorded from thejoker 1.3.1 (the seven APOGEE epochs of J0008+0128, recorded below) and with a dense computation of the same covariance; the sampler with the NumPy sampler of oracle_joker.py on a case that accepts many draws."""
import numpy as np
import pytest
import torch

import wobbler
from wobbler import joker

pytestmark = pytest.mark.torch_cpu

TIME = np.array([56644.0, 56666.0, 56848.0, 57654.0, 57674.0, 57677.0, 58033.0])
VELOCITY = np.array([-27.265276, -27.288692, -37.029631, -36.870444, -35.909287, -35.361665, -34.156430])
ERROR = np.array([0.027409, 0.026189, 0.024986, 0.032949, 0.341447, 0.028945, 0.036848])
JITTER = 0.1
# P (d), e, omega (rad), M0 (rad, thejoker's convention: the mean anomaly is n (t - t_ref) - M0)
DRAWS = np.array([[1217.19, 0.09, 0.62, -1.0], [614.0, 0.06, 1.1, 0.4], [1220.3, 0.15, 2.0, -0.7], [300.0, 0.3, 4.0, 2.5], [2500.0, 0.5, 5.5, -3.0], [90.0, 0.02, 0.1, 1.9]])
GOLDEN = np.array([-4671.23801876, -2047.25477503, -1992.96768965, -4836.5506028, -4845.61904432, -3897.95710219])  # thejoker 1.3.1 marginal_ln_likelihood


def likelihood(draws, time, velocity, variance, **kwargs):
    period, e, omega, m0 = (torch.as_tensor(column, dtype=torch.float64) for column in draws.T)
    return joker.marginal_ln_likelihood_of_draws(period, e, omega, -m0, time, velocity, variance, **kwargs)


def test_the_marginal_likelihood_equals_thejokers_recorded_values_on_shared_draws():
    ours = likelihood(DRAWS, TIME, VELOCITY, ERROR**2 + JITTER**2, device="cpu").numpy()
    np.testing.assert_allclose(ours, GOLDEN, rtol=1e-8)


def test_the_sign_of_m0_matters():
    period, e, omega, m0 = (torch.as_tensor(column, dtype=torch.float64) for column in DRAWS.T)
    wrong = joker.marginal_ln_likelihood_of_draws(period, e, omega, m0, TIME, VELOCITY, ERROR**2 + JITTER**2, device="cpu").numpy()
    assert not np.allclose(wrong, GOLDEN, rtol=1e-3), "the convention check has no power"


def test_the_rank_two_formula_equals_the_dense_covariance_on_random_draws():
    rng = np.random.default_rng(3)
    n, count = 12, 9
    basis = rng.normal(size=(n, count))
    sigma_k = rng.uniform(1.0, 50.0, n)
    velocity = rng.normal(size=count) * 10
    variance = rng.uniform(0.01, 4.0, count)
    covariance = np.diag(variance)[None] + sigma_k[:, None, None] ** 2 * basis[:, :, None] * basis[:, None, :] + joker.SIGMA_V0**2
    sign, logdet = np.linalg.slogdet(covariance)
    solved = np.linalg.solve(covariance, np.broadcast_to(velocity, (n, count))[:, :, None])[:, :, 0]
    dense = -0.5 * (solved @ velocity) - 0.5 * logdet - 0.5 * count * np.log(2 * np.pi)
    ours = joker.marginal_ln_likelihood(torch.as_tensor(basis), torch.as_tensor(sigma_k), torch.as_tensor(velocity), torch.as_tensor(variance)).numpy()
    np.testing.assert_allclose(ours, dense, rtol=1e-10)


def synthetic(seed=5, epochs=5, error=6.0, period=300.0, ecc=0.3, k=8.0):
    rng = np.random.default_rng(seed)
    time = np.sort(rng.uniform(0, 600, epochs))
    omega, m0 = 0.7, 1.2
    import oracle_joker as reference

    shape = reference.shape(time, np.array([period]), np.array([ecc]), np.array([omega]), np.array([m0]), float(time.min()))[0]
    velocity = k * shape + 3.0 + rng.normal(size=epochs) * error
    return time, velocity, np.full(epochs, error)


def test_the_sampler_agrees_with_the_numpy_sampler_on_a_case_that_accepts_many_draws():
    from scipy import stats

    import oracle_joker as reference

    time, velocity, error = synthetic()
    ours = joker.sample(time, velocity, error, draws=400_000, seed=1, chunk=100_000, device="cpu")
    theirs, _ = reference.sample(time, velocity, error, draws=400_000, seed=2, chunk=100_000)
    assert len(ours["period"]) > 200 and len(theirs["period"]) > 200, (len(ours["period"]), len(theirs["period"]))
    for name in ("period", "eccentricity", "K", "v0", "ln_likelihood"):
        result = stats.ks_2samp(ours[name], theirs[name])
        assert result.pvalue > 0.01, (name, result)


def test_the_same_seed_and_chunk_give_the_same_samples_and_another_seed_does_not():
    time, velocity, error = synthetic()
    a = joker.sample(time, velocity, error, draws=50_000, seed=7, chunk=10_000, device="cpu")
    b = joker.sample(time, velocity, error, draws=50_000, seed=7, chunk=10_000, device="cpu")
    c = joker.sample(time, velocity, error, draws=50_000, seed=8, chunk=10_000, device="cpu")
    assert np.array_equal(a["period"], b["period"]) and not np.array_equal(a["period"], c["period"][: len(a["period"])]) 


def test_the_best_draw_is_always_kept_and_no_draw_gives_no_sample():
    """A rejection sampler accepts a draw with probability exp(ln L - ln L_best), so the best draw is kept whatever the data."""
    time, velocity, error = synthetic(error=0.001)
    result = joker.sample(time, velocity + 1e3 * np.sin(time), error, draws=20_000, seed=1, chunk=10_000, device="cpu")
    assert result["draws"] == 20_000 and len(result["period"]) >= 1
    empty = joker.sample(time, velocity, error, draws=0, device="cpu")
    assert empty["draws"] == 0 and all(len(v) == 0 for k, v in empty.items() if k != "draws")


def test_the_partial_last_chunk_draws_no_more_than_the_requested_count():
    time, velocity, error = synthetic()
    result = joker.sample(time, velocity, error, draws=25_000, seed=3, chunk=10_000, device="cpu")
    assert result["draws"] == 25_000
    assert np.all(result["period"] >= 20.0) and np.all(result["period"] <= 20000.0)


def test_the_package_exports_the_sampler():
    assert wobbler.joker is joker
