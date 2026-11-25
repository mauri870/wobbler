"""A NumPy rejection sampler for the orbit of a single-lined binary from radial velocities, after The Joker (Price-Whelan et al. 2017, ApJ 837, 20, arXiv:1610.07602), kept as the reference that tests/test_joker.py compares the torch sampler with.
Method (paper, section 2): draw the nonlinear parameters (P, e, omega, M0) from their priors; the linear parameters (K, v0) have Gaussian priors and are marginalised analytically, which gives the marginal likelihood of each draw; a draw is kept with probability proportional to its marginal likelihood (rejection sampling); the kept draws get (K, v0) from their conditional Gaussian.
Priors: ln P uniform between PERIOD_MIN and PERIOD_MAX; e from Beta(0.867, 3.03) (Kipping 2013, MNRAS 434, L51); omega and M0 uniform; v0 ~ N(0, 100 km/s); K ~ N(0, sigma_K) with sigma_K = 30 km/s (P / 1 yr)^(-1/3) / sqrt(1 - e^2). Excess velocity variance (jitter) enters the errors in quadrature as a fixed value.
"""
import numpy as np

SIGMA_V0 = 100.0
SIGMA_K0 = 30.0
YEAR = 365.25
DROP_BELOW_BEST = 30.0  # in ln likelihood


def kepler(mean_anomaly, eccentricity):
    """Eccentric anomaly E from M = E - e sin E by Newton's method (vectorised; 40 iterations cover e up to 0.99)."""
    anomaly = mean_anomaly + eccentricity * np.sin(mean_anomaly)
    for _ in range(40):
        anomaly = anomaly - (anomaly - eccentricity * np.sin(anomaly) - mean_anomaly) / (1.0 - eccentricity * np.cos(anomaly))
    return anomaly


def shape(time, period, eccentricity, omega, m0, t_ref):
    """The unit-amplitude velocity curve cos(omega + f) + e cos(omega) at each time (rows: draws, columns: times)."""
    mean = 2.0 * np.pi * (time[None, :] - t_ref) / period[:, None] + m0[:, None]
    e = eccentricity[:, None]
    anomaly = kepler(np.mod(mean, 2.0 * np.pi), e)
    true = 2.0 * np.arctan2(np.sqrt(1.0 + e) * np.sin(anomaly / 2.0), np.sqrt(1.0 - e) * np.cos(anomaly / 2.0))
    return np.cos(omega[:, None] + true) + e * np.cos(omega[:, None])


def marginal_ln_likelihood(basis, sigma_k, velocity, variance):
    """ln p(velocity | P, e, omega, M0) with (K, v0) marginalised, for each row of `basis` (the unit-amplitude curves of `shape`) and its K prior width `sigma_k`: y = K basis + v0 + noise, so y ~ N(0, C)
    with C = N + sigma_k^2 b b^T + SIGMA_V0^2 1 1^T and N the diagonal of `variance` (errors and jitter squared)."""
    n, count = basis.shape
    covariance = np.broadcast_to(np.diag(variance), (n, count, count)).copy()
    covariance += (sigma_k[:, None, None] ** 2) * basis[:, :, None] * basis[:, None, :] + SIGMA_V0**2
    sign, logdet = np.linalg.slogdet(covariance)
    solved = np.linalg.solve(covariance, np.broadcast_to(velocity, (n, count))[:, :, None])[:, :, 0]
    return -0.5 * (solved @ velocity) - 0.5 * logdet - 0.5 * count * np.log(2 * np.pi)


def sample(time, velocity, error, draws=2_000_000, seed=0, period_min=20.0, period_max=20000.0, jitter=0.0, maximum=1000, chunk=200_000):
    """Return a dict of posterior samples (period, eccentricity, omega, m0, K, v0, ln_likelihood) and the count of prior draws."""
    rng = np.random.default_rng(seed)
    time = np.asarray(time, float)
    velocity = np.asarray(velocity, float)
    variance = np.asarray(error, float) ** 2 + jitter**2
    t_ref = float(time.min())
    kept = []
    best = []
    for start in range(0, draws, chunk):
        n = min(chunk, draws - start)
        period = np.exp(rng.uniform(np.log(period_min), np.log(period_max), n))
        e = rng.beta(0.867, 3.03, n)
        omega = rng.uniform(0, 2 * np.pi, n)
        m0 = rng.uniform(0, 2 * np.pi, n)
        basis = shape(time, period, e, omega, m0, t_ref)  # n x N
        sigma_k = SIGMA_K0 * (period / YEAR) ** (-1.0 / 3.0) / np.sqrt(1.0 - e**2)
        count = len(time)
        ln_like = marginal_ln_likelihood(basis, sigma_k, velocity, variance)
        best.append((ln_like, period, e, omega, m0, basis, sigma_k))
        kept.append(ln_like.max())
        # a draw more than DROP_BELOW_BEST under the best so far is accepted with a probability under 1e-13 whatever the later chunks bring (the best only rises): its basis is not kept
        floor = max(kept) - DROP_BELOW_BEST
        best = [tuple(part[keep] for part in entry) for entry in best for keep in [entry[0] > floor] if keep.any()]
    top = max(kept)
    out = {k: [] for k in ("period", "eccentricity", "omega", "m0", "K", "v0", "ln_likelihood")}
    for ln_like, period, e, omega, m0, basis, sigma_k in best:
        accept = np.where(rng.uniform(size=len(ln_like)) < np.exp(ln_like - top))[0]
        for i in accept:
            design = np.column_stack([basis[i], np.ones(len(time))])
            precision = design.T @ (design / variance[:, None]) + np.diag([1.0 / sigma_k[i] ** 2, 1.0 / SIGMA_V0**2])
            mean = np.linalg.solve(precision, design.T @ (velocity / variance))
            k, v0 = rng.multivariate_normal(mean, np.linalg.inv(precision))
            for name, value in zip(out, (period[i], e[i], omega[i], m0[i], k, v0, ln_like[i])):
                out[name].append(float(value))
    return {k: np.array(v) for k, v in out.items()}, draws


def mass_function(period, eccentricity, k):
    """f(m) in solar masses from P in days, e and K in km/s: 1.0361e-7 (1 - e^2)^(3/2) K^3 P (Torres-style constant from G and the solar mass; cross-checked against Gaia's 0.537 for J0008+0128)."""
    return 1.0361e-7 * (1.0 - eccentricity**2) ** 1.5 * np.abs(k) ** 3 * period
