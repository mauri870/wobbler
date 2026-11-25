"""A rejection sampler for the orbit of a single-lined binary from radial velocities after The Joker (Price-Whelan et al. 2017, ApJ 837, 20, arXiv:1610.07602), on torch: the draws, the curves and the marginal likelihood of every draw are computed in slices on one device.

Method (paper, section 2): the nonlinear parameters (P, e, omega, M0) are drawn from their priors; the linear parameters (K, v0) have Gaussian priors and are marginalised analytically, which gives the marginal likelihood of each draw; a draw is kept with probability proportional to its marginal likelihood; the kept draws get (K, v0) from their conditional Gaussian.
Priors (the paper's defaults and thejoker's): ln P uniform between `period_min` and `period_max`; e from Beta(0.867, 3.03) (Kipping 2013, MNRAS 434, L51); omega and M0 uniform; v0 ~ N(0, 100 km/s); K ~ N(0, sigma_K) with sigma_K = 30 km/s (P / 1 yr)^(-1/3) / sqrt(1 - e^2). Excess velocity variance enters the errors in quadrature as a fixed value (thejoker's likelihood ignores the jitter of its prior, https://thejoker.readthedocs.io).
The covariance of the velocities is C = N + sigma_K^2 b b^T + sigma_v0^2 1 1^T, with N the diagonal of the variances and b the unit-amplitude curve, a rank-two update of a diagonal, so its determinant and its inverse are computed from two-by-two matrices (the matrix determinant lemma and the Woodbury identity) at a cost linear in the epochs.
The random draws come from torch generators seeded by the seed and the slice index: a run repeats on its device with its slice size, and a CPU and a GPU give different draws for the same seed.
"""
import numpy as np
import torch

from ._compile import hot
from ._tensor import as_tensor, pick_device
from .orbit import kepler, true_anomaly

SIGMA_V0 = 100.0
SIGMA_K0 = 30.0
YEAR = 365.25
DROP_BELOW_BEST = 30.0  # ln likelihood: a draw this far under the best so far is accepted with a probability under 1e-13 whatever the later slices bring
BETA_A, BETA_B = 0.867, 3.03
ECC_LIMIT = 1.0 - 1e-12


def shape(time, period, ecc, omega, m0, t_ref, device=None):
    """The unit-amplitude velocity curve cos(omega + f) + e cos(omega) of each draw (rows) at each time (columns); m0 is the mean anomaly at t_ref."""
    device = pick_device(time, period, ecc, omega, m0, device=device)
    time, period, ecc, omega, m0 = (as_tensor(x, device) for x in (time, period, ecc, omega, m0))
    mean = 2 * torch.pi * (time[None, :] - t_ref) / period[:, None] + m0[:, None]
    e = ecc[:, None]
    true = true_anomaly(kepler(torch.remainder(mean, 2 * torch.pi), e.expand_as(mean), device=device), e, device=device)
    return torch.cos(omega[:, None] + true) + e * torch.cos(omega[:, None])


def _ln_likelihood(basis, sigma_k, velocity, variance):
    inverse = 1.0 / variance
    sigma_k2 = sigma_k**2
    m11 = (basis**2 * inverse).sum(1)
    m12 = (basis * inverse).sum(1)
    m22 = inverse.sum()
    a = velocity * inverse
    quadratic0 = (velocity * a).sum()
    u1 = (basis * a).sum(1)
    u2 = a.sum()
    # I + Lambda M, Lambda = diag(sigma_k^2, sigma_v0^2): its determinant, and (Lambda^-1 + M)^-1 u by Cramer's rule
    det = (1.0 + sigma_k2 * m11) * (1.0 + SIGMA_V0**2 * m22) - sigma_k2 * m12 * SIGMA_V0**2 * m12
    p11, p12, p22 = 1.0 / sigma_k2 + m11, m12, 1.0 / SIGMA_V0**2 + m22
    precision_det = p11 * p22 - p12 * p12
    quadratic = quadratic0 - (p22 * u1 * u1 - 2 * p12 * u1 * u2 + p11 * u2 * u2) / precision_det
    count = basis.shape[1]
    return -0.5 * quadratic - 0.5 * (torch.log(variance).sum() + torch.log(det)) - 0.5 * count * np.log(2 * np.pi)


_ln_likelihood_fused = hot(_ln_likelihood)


def marginal_ln_likelihood(basis, sigma_k, velocity, variance, device=None):
    """ln p(velocity | P, e, omega, M0) with (K, v0) marginalised, for each row of `basis` (the unit-amplitude curves of `shape`) and its K prior width `sigma_k`: y = K basis + v0 + noise with noise variances `variance` (errors and jitter squared)."""
    device = pick_device(basis, sigma_k, velocity, variance, device=device)
    basis, sigma_k, velocity, variance = (as_tensor(x, device) for x in (basis, sigma_k, velocity, variance))
    return _ln_likelihood_fused(basis, sigma_k, velocity, variance)


def sigma_k_of(period, ecc):
    """The width of K's prior: 30 km/s (P / 1 yr)^(-1/3) / sqrt(1 - e^2)."""
    return SIGMA_K0 * (period / YEAR) ** (-1.0 / 3.0) / torch.sqrt(1.0 - ecc**2)


def marginal_ln_likelihood_of_draws(period, ecc, omega, m0, time, velocity, variance, t_ref=None, device=None):
    """The marginal ln likelihood of draws of (P, e, omega, M0) (arrays of one length) at the epochs `time` with velocities and variances."""
    device = pick_device(period, ecc, omega, m0, time, device=device)
    period, ecc, omega, m0, time, velocity, variance = (as_tensor(x, device) for x in (period, ecc, omega, m0, time, velocity, variance))
    t_ref = float(time.min()) if t_ref is None else t_ref
    return marginal_ln_likelihood(shape(time, period, ecc, omega, m0, t_ref, device=device), sigma_k_of(period, ecc), velocity, variance, device=device)


def _draw(count, seed, index, period_min, period_max, device):
    """The prior draws of one slice. The generators are seeded by the seed and the slice index; the Beta draws use the global generator of the device inside a forked state."""
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) * 1_000_003 + index)
    low, high = np.log(period_min), np.log(period_max)
    period = torch.exp(low + (high - low) * torch.rand(count, generator=generator, device=device, dtype=torch.float64))
    omega = 2 * torch.pi * torch.rand(count, generator=generator, device=device, dtype=torch.float64)
    m0 = 2 * torch.pi * torch.rand(count, generator=generator, device=device, dtype=torch.float64)
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(int(seed) * 1_000_003 + index)
        e = torch.distributions.Beta(torch.tensor(BETA_A, dtype=torch.float64, device=device), torch.tensor(BETA_B, dtype=torch.float64, device=device)).sample((count,))
    return period, e.clamp(max=ECC_LIMIT), omega, m0, generator


def sample(time, velocity, error, draws=2_000_000, seed=0, period_min=20.0, period_max=20000.0, jitter=0.0, chunk=None, device=None):
    """Posterior samples of the orbit by rejection from `draws` prior draws, as a dict of NumPy arrays: period, eccentricity, omega, m0, K, v0, ln_likelihood, and `draws` (the count of prior draws). The draws are made and scored in slices of `chunk` (default 262,144 on a GPU and 32,768 on a CPU), each of one fixed size, the last one padded and masked."""
    device = pick_device(time, velocity, error, device=device)
    time, velocity, error = (as_tensor(x, device) for x in (time, velocity, error))
    variance = error**2 + jitter**2
    t_ref = float(time.min())
    chunk = int(chunk or (262_144 if device.type == "cuda" else 32_768))
    if draws <= 0:
        return {**{name: np.zeros(0) for name in ("period", "eccentricity", "omega", "m0", "K", "v0", "ln_likelihood")}, "draws": 0}
    candidates, best = [], -np.inf
    for index, start in enumerate(range(0, draws, chunk)):
        period, e, omega, m0, generator = _draw(chunk, seed, index, period_min, period_max, device)
        ln_like = marginal_ln_likelihood_of_draws(period, e, omega, m0, time, velocity, variance, t_ref=t_ref, device=device)
        ln_like = torch.where(torch.arange(chunk, device=device) < draws - start, ln_like, torch.full_like(ln_like, -torch.inf))
        best = max(best, float(ln_like.max()))
        keep = ln_like > best - DROP_BELOW_BEST
        candidates.append([x[keep] for x in (ln_like, period, e, omega, m0)])
    ln_like, period, e, omega, m0 = (torch.cat([c[i] for c in candidates]) for i in range(5))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) * 1_000_003 + 999_999_937)
    accept = torch.rand(len(ln_like), generator=generator, device=device, dtype=torch.float64) < torch.exp(ln_like - best)
    ln_like, period, e, omega, m0 = (x[accept] for x in (ln_like, period, e, omega, m0))
    k, v0 = _linear(shape(time, period, e, omega, m0, t_ref, device=device), sigma_k_of(period, e), velocity, variance, generator)
    out = {"period": period, "eccentricity": e, "omega": omega, "m0": m0, "K": k, "v0": v0, "ln_likelihood": ln_like}
    result = {name: value.cpu().numpy() for name, value in out.items()}
    result["draws"] = draws
    return result


def _linear(basis, sigma_k, velocity, variance, generator):
    """K and v0 of each accepted draw from their conditional Gaussian: precision [[sum b^2/var + 1/sigma_k^2, sum b/var], [sum b/var, sum 1/var + 1/sigma_v0^2]], mean precision^-1 [sum b y/var, sum y/var]."""
    inverse = 1.0 / variance
    p11 = (basis**2 * inverse).sum(1) + 1.0 / sigma_k**2
    p12 = (basis * inverse).sum(1)
    p22 = inverse.sum() + 1.0 / SIGMA_V0**2
    r1 = (basis * (velocity * inverse)).sum(1)
    r2 = (velocity * inverse).sum()
    det = p11 * p22 - p12 * p12
    mean_k, mean_v = (p22 * r1 - p12 * r2) / det, (p11 * r2 - p12 * r1) / det
    # covariance = precision^-1 = [[p22, -p12], [-p12, p11]] / det, its Cholesky factor by hand
    c11, c12, c22 = p22 / det, -p12 / det, p11 / det
    l11 = torch.sqrt(c11)
    l21 = c12 / l11
    l22 = torch.sqrt(c22 - l21**2)
    z = torch.randn(2, len(det), generator=generator, device=basis.device, dtype=torch.float64)
    return mean_k + l11 * z[0], mean_v + l21 * z[0] + l22 * z[1]
