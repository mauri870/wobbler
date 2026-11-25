"""The grid search for an astrometric orbit in along-scan abscissae, for many target vectors at once.

The abscissa of a star on an orbit is linear in the parameters of a base model (the five-parameter single-star motion) and in the four Thiele-Innes elements A, B, F, G when the period P, the eccentricity e and the time of periastron T_p are fixed: w = base p + A cos(theta) X + B sin(theta) X + F cos(theta) Y + G sin(theta) Y, with X = cos E - e, Y = sqrt(1 - e^2) sin E and theta the scan angle (Halbwachs et al. 2023, A&A 674, A9, arXiv:2206.05726, equations A.1 and 10). The search takes a grid over (P, e, T_p), projects the base model out of the four columns and of the data once, and solves the four elements by the normal equations at every grid point.
The columns depend on the epochs and the grid only, so the grid is evaluated once for any number of target vectors (the data and, for a null distribution, simulated data at the same epochs): the work per extra vector is the product of the columns with it.
Grid: frequencies in steps of 1 / (step_share x the span of the epochs) in two bands, the long band from 1 / period_range[1] to 1 / period_range[0] with `eccentricities` x `phases` grid points a frequency, and, when `floor` is below period_range[0], a short band up to 1 / floor with `short_eccentricities` x `short_phases`; the phases are fractions of the period after the first epoch.
"""
import math

import numpy as np
import torch

from ._tensor import as_tensor, pick_device
from .orbit import kepler

ECCENTRICITIES = (0.0, 0.2, 0.4, 0.6, 0.75, 0.9)
SHORT_ECCENTRICITIES = (0.0, 0.3, 0.6)
ELEMENTS_AT_A_TIME = 32_000_000  # the number of doubles of the columns of one chunk of frequencies (256 MB)


def _bands(span, period_range, floor, eccentricities, phases, short_eccentricities, short_phases, step_share):
    step = 1.0 / (step_share * span)
    bands = [(1.0 / period_range[1], 1.0 / period_range[0], eccentricities, phases)]
    if floor < period_range[0]:
        bands.append((1.0 / period_range[0] + step, 1.0 / floor, short_eccentricities, short_phases))
    return bands, step


def grid_search(jd, theta_deg, sigma, base, targets, period_range=(100.0, 10000.0), floor=10.0, eccentricities=ECCENTRICITIES, phases=16, short_eccentricities=SHORT_ECCENTRICITIES, short_phases=8, step_share=6.0, device=None, chunk=None):
    """The best orbit of the grid for each column of `targets`.

    jd (days), theta_deg (scan angle, degrees) and sigma (mas) are arrays of the N epochs; base is the N x p design matrix of the base model (unweighted); targets is N x K abscissae (mas). Returns a dict of tensors of length K: chi2 (the weighted residual after the base model and the best orbit), chi2_base (the residual after the base model alone), period, eccentricity and periastron (the epoch of periastron, in the unit of jd) of the best grid point. `chunk` is the number of frequencies evaluated at once (default: as many as keep the columns under ELEMENTS_AT_A_TIME doubles).
    """
    device = pick_device(jd, theta_deg, sigma, base, targets, device=device)
    jd_np = np.asarray(jd, dtype=float)
    jd_t, theta, sigma_t, base_t, target = (as_tensor(x, device) for x in (jd, theta_deg, sigma, base, targets))
    target = target.reshape(len(jd_np), -1)
    count = target.shape[1]
    weighted_base = base_t / sigma_t[:, None]
    q, _ = torch.linalg.qr(weighted_base)
    t = target / sigma_t[:, None]
    t = t - q @ (q.T @ t)
    chi2_base = (t * t).sum(0)
    cosine, sine = torch.cos(torch.deg2rad(theta)), torch.sin(torch.deg2rad(theta))
    bands, step = _bands(float(np.ptp(jd_np)), period_range, floor, eccentricities, phases, short_eccentricities, short_phases, step_share)
    best = torch.full((count,), math.inf, dtype=torch.float64, device=device)
    where = torch.zeros((count, 3), dtype=torch.float64, device=device)
    first = float(jd_np.min())
    for low, high, band_eccentricities, band_phases in bands:
        ecc = torch.as_tensor(np.repeat(np.asarray(band_eccentricities, float), band_phases), device=device)
        fraction = torch.as_tensor(np.tile(np.arange(band_phases) / band_phases, len(band_eccentricities)), device=device)
        frequencies = np.arange(low, high + step, step)
        grid = len(ecc)
        size = int(chunk or max(1, ELEMENTS_AT_A_TIME // (grid * len(jd_np) * 4)))
        for start in range(0, len(frequencies), size):
            period = torch.as_tensor(1.0 / frequencies[start:start + size], device=device)  # (c,)
            tp = first + period[:, None] * fraction[None, :]  # (c, G)
            mean = 2 * math.pi * (jd_t[None, None, :] - tp[:, :, None]) / period[:, None, None]
            anomaly = kepler(mean, ecc[None, :, None].expand_as(mean), device=device)
            x, y = torch.cos(anomaly) - ecc[None, :, None], torch.sqrt(1 - ecc**2)[None, :, None] * torch.sin(anomaly)
            columns = torch.stack([cosine * x, sine * x, cosine * y, sine * y], dim=-1) / sigma_t[None, None, :, None]  # (c, G, N, 4)
            columns = columns - torch.einsum("np,cgpj->cgnj", q, torch.einsum("np,cgnj->cgpj", q, columns))
            normal = torch.einsum("cgni,cgnj->cgij", columns, columns) + 1e-12 * torch.eye(4, dtype=torch.float64, device=device)
            rhs = torch.einsum("cgni,nk->cgik", columns, t)
            solution = torch.linalg.solve(normal, rhs)
            chi2 = chi2_base[None, None, :] - (solution * rhs).sum(-2)  # (c, G, K)
            flat = chi2.reshape(-1, count)
            value, index = flat.min(0)
            improved = value < best
            if bool(improved.any()):
                c_index, g_index = index // grid, index % grid
                candidate = torch.stack([period[c_index], ecc[g_index % grid], tp[c_index, g_index % grid]], dim=1)
                where = torch.where(improved[:, None], candidate, where)
                best = torch.where(improved, value, best)
    return {"chi2": best, "chi2_base": chi2_base, "period": where[:, 0], "eccentricity": where[:, 1], "periastron": where[:, 2]}
