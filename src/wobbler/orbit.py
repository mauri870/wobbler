"""Orbits on tensors: Kepler's equation, the Thiele-Innes constants and their inverse, the radial velocity and the photocentre orbit. Every function broadcasts over its arguments (NumPy arrays, numbers or tensors, float64), so a batch of cases and a batch of epochs are two axes of the same call, and returns tensors on the device of its inputs (`device` overrides).

Conventions are those of gaiamock (El-Badry et al. 2024, arXiv:2411.00088; https://github.com/kareemelbadry/gaiamock, gaiamock/gaiamock.py): `node` is the longitude of the ascending node (gaiamock's omega), `omega` the argument of periastron (gaiamock's w), the inclination is in radians, x is the offset to the east and y to the north.
"""
from dataclasses import dataclass

import torch

from ._compile import hot
from ._tensor import as_tensor, pick_device

DAYS_PER_YEAR = 365.25
TWO_PI = 2 * torch.pi
DANBY_UPDATES = 3  # updates of Danby's iteration that bring the anomaly to rounding for every eccentricity up to DANBY_ECC_LIMIT (tested: the error against the converged solution is under 1e-13)
DANBY_ECC_LIMIT = 0.8


@dataclass(frozen=True)
class Orbit:
    """The elements of one orbit or of an array of orbits: period (days), eccentricity, time of periastron (days from the reference epoch JD 2457389.0, Gaia's 2016.0, unless a function takes another reference_jd) and argument of periastron (radians)."""

    period_days: object
    ecc: object
    t_peri_days: object
    omega: object


def _danby(anomaly, e, wrapped):
    """DANBY_UPDATES updates of Danby's fourth-order iteration (Danby 1988, Fundamentals of Celestial Mechanics, section 6.6)."""
    for _ in range(DANBY_UPDATES):
        sine, cosine = torch.sin(anomaly), torch.cos(anomaly)
        f = anomaly - e * sine - wrapped
        f1, f2, f3 = 1 - e * cosine, e * sine, e * cosine
        d1 = -f / f1
        d2 = -f / (f1 + 0.5 * d1 * f2)
        d3 = -f / (f1 + 0.5 * d2 * f2 + d2 * d2 * f3 / 6)
        anomaly = anomaly + d3
    return anomaly


_danby_fused = hot(_danby)


def kepler(mean_anomaly, ecc, tolerance=1e-12, max_iterations=60, device=None):
    """The eccentric anomaly E of E - e sin E = M, for mean anomalies (radians, any range) and eccentricities in [0, 1): Danby's fourth-order iteration from the start M + 0.85 e sign(sin M). For a batch whose largest eccentricity is at most 0.8 it makes three updates and no check (the three updates bring every anomaly to rounding, tested against the converged solution); above, it iterates until the largest residual is under `tolerance` (or max_iterations). The result carries the revolutions of the mean anomaly (E - M is periodic)."""
    device = pick_device(mean_anomaly, ecc, device=device)
    mean, e = torch.broadcast_tensors(as_tensor(mean_anomaly, device), as_tensor(ecc, device))
    if bool(((e < 0) | (e >= 1)).any()):
        raise ValueError("the eccentricity must be in [0, 1)")
    wrapped = torch.remainder(mean, TWO_PI)
    anomaly = wrapped + 0.85 * e * torch.sign(torch.sin(wrapped))
    if mean.numel() == 0 or float(e.max()) <= DANBY_ECC_LIMIT:
        anomaly = _danby_fused(anomaly, e, wrapped)
    else:
        for _ in range(max_iterations):
            sine, cosine = torch.sin(anomaly), torch.cos(anomaly)
            f = anomaly - e * sine - wrapped
            if float(f.abs().max()) < tolerance:
                break
            f1, f2, f3 = 1 - e * cosine, e * sine, e * cosine
            d1 = -f / f1
            d2 = -f / (f1 + 0.5 * d1 * f2)
            d3 = -f / (f1 + 0.5 * d2 * f2 + d2 * d2 * f3 / 6)
            anomaly = anomaly + d3
    return anomaly + (mean - wrapped)


def thiele_innes(a, node, omega, inclination, device=None):
    """(A, B, F, G), the Thiele-Innes constants of a semi-major axis `a` (any unit) with the node, argument of periastron and inclination: the lines of gaiamock.predict_astrometry_binary_in_terms_of_a0."""
    device = pick_device(a, node, omega, inclination, device=device)
    a, node, omega, inclination = (as_tensor(x, device) for x in (a, node, omega, inclination))
    co, so, cn, sn, ci = torch.cos(omega), torch.sin(omega), torch.cos(node), torch.sin(node), torch.cos(inclination)
    return (a * (co * cn - so * sn * ci), a * (co * sn + so * cn * ci), -a * (so * cn + co * sn * ci), -a * (so * sn - co * cn * ci))


def campbell(A, B, F, G, device=None):
    """The elements {a, node, omega, inclination} of the Thiele-Innes constants (the inverse of thiele_innes): the sum and difference of omega and node from the two combinations A + G, B - F and A - G, B + F (Heintz 1978; Halbwachs et al. 2023, A&A 674, A9, appendix A).
    omega and node are determined up to adding pi to both, which leaves the constants unchanged."""
    device = pick_device(A, B, F, G, device=device)
    A, B, F, G = (as_tensor(x, device) for x in (A, B, F, G))
    plus, minus = torch.hypot(A + G, B - F), torch.hypot(A - G, B + F)  # a (1 + cos i) and a (1 - cos i)
    total = torch.atan2(B - F, A + G)  # omega + node
    difference = torch.atan2(-(B + F), A - G)  # omega - node
    return {"a": 0.5 * (plus + minus), "node": torch.remainder(0.5 * (total - difference), TWO_PI), "omega": torch.remainder(0.5 * (total + difference), TWO_PI), "inclination": 2 * torch.atan2(torch.sqrt(minus), torch.sqrt(plus))}


def true_anomaly(eccentric_anomaly, ecc, device=None):
    device = pick_device(eccentric_anomaly, ecc, device=device)
    eccentric_anomaly, ecc = as_tensor(eccentric_anomaly, device), as_tensor(ecc, device)
    return 2 * torch.atan2(torch.sqrt(1 + ecc) * torch.sin(eccentric_anomaly / 2), torch.sqrt(1 - ecc) * torch.cos(eccentric_anomaly / 2))


def radial_velocity(t_days, period_days, ecc, t_peri_days, omega, k, device=None):
    """The radial velocity k (cos(omega + f) + e cos omega) of a single-lined orbit at times t_days (days from the reference epoch); k is the semi-amplitude, in any velocity unit."""
    device = pick_device(t_days, period_days, ecc, t_peri_days, omega, k, device=device)
    t_days, period_days, ecc, t_peri_days, omega, k = (as_tensor(x, device) for x in (t_days, period_days, ecc, t_peri_days, omega, k))
    anomaly = kepler(2 * torch.pi / period_days * (t_days - t_peri_days), ecc)
    return k * (torch.cos(omega + true_anomaly(anomaly, ecc)) + ecc * torch.cos(omega))


def photocentre(t_yr, period_days, ecc, t_peri_days, a0_mas, inclination, node, omega, device=None):
    """(x, y), the east and north offsets of the photocentre from the barycentre at times t_yr (years from the reference epoch), for a photocentre orbit of semi-major axis a0_mas: x = B X + G Y, y = A X + F Y with X = cos E - e and Y = sqrt(1 - e^2) sin E,
    as gaiamock.predict_astrometry_binary_in_terms_of_a0."""
    device = pick_device(t_yr, period_days, ecc, t_peri_days, a0_mas, inclination, node, omega, device=device)
    t_yr, period_days, ecc, t_peri_days, a0_mas, inclination, node, omega = (as_tensor(x, device) for x in (t_yr, period_days, ecc, t_peri_days, a0_mas, inclination, node, omega))
    anomaly = kepler(2 * torch.pi / period_days * (t_yr * DAYS_PER_YEAR - t_peri_days), ecc)
    A, B, F, G = thiele_innes(a0_mas, node, omega, inclination)
    X, Y = torch.cos(anomaly) - ecc, torch.sqrt(1 - ecc**2) * torch.sin(anomaly)
    return B * X + G * Y, A * X + F * Y
