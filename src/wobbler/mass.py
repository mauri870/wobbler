"""The mass function of a single-lined orbit and the companion masses that follow from it, on tensors (NumPy arrays, numbers or tensors in, float64 tensors out)."""
import torch

from ._tensor import as_tensor, pick_device

G_SI, SOLAR_MASS_KG = 6.6743e-11, 1.98840987e30  # CODATA 2018 and IAU 2015 nominal solar mass parameter over G


def mass_function(period_days, k_kms, ecc, device=None):
    """f(m) = P K^3 (1 - e^2)^(3/2) / (2 pi G), in solar masses, for the period in days, the semi-amplitude K in km/s and the eccentricity (Hilditch 2001, An Introduction to Close Binary Stars, section 2.3)."""
    device = pick_device(period_days, k_kms, ecc, device=device)
    period_s = as_tensor(period_days, device) * 86400.0
    k_ms = as_tensor(k_kms, device) * 1000.0
    return period_s * k_ms**3 * (1 - as_tensor(ecc, device) ** 2) ** 1.5 / (2 * torch.pi * G_SI) / SOLAR_MASS_KG


def min_companion_mass(f, m1, iterations=100, device=None):
    """m2 of m2^3 / (m1 + m2)^2 = f (an edge-on orbit), by bisection on tensors: the least companion mass for the mass function f and the primary's mass m1 (solar masses). A zero mass function has the zero mass."""
    device = pick_device(f, m1, device=device)
    f, m1 = torch.broadcast_tensors(as_tensor(f, device), as_tensor(m1, device))
    low = torch.zeros_like(f)
    high = torch.clamp(2 * (f + torch.sqrt(f * m1)), min=1.0)
    for _ in range(60):  # the bracket doubles until the cube over the square exceeds f
        small = high**3 / (m1 + high) ** 2 < f
        if not bool(small.any()):
            break
        high = torch.where(small, 2 * high, high)
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        below = mid**3 / (m1 + mid) ** 2 < f
        low, high = torch.where(below, mid, low), torch.where(below, high, mid)
    return 0.5 * (low + high) * (f > 0)


def companion_mass(f, m1, inclination, device=None):
    """m2 of m2^3 sin^3 i = f (m1 + m2)^2 at the inclination i (radians), the least mass divided out by sin^3 i."""
    device = pick_device(f, m1, inclination, device=device)
    return min_companion_mass(as_tensor(f, device) / torch.sin(as_tensor(inclination, device)) ** 3, m1, device=device)


def share_inclined_below(limit, device=None):
    """The share of isotropically oriented orbits whose inclination is under `limit` (radians, between 0 and pi / 2): 1 - cos(limit)."""
    device = pick_device(limit, device=device)
    return 1 - torch.cos(torch.clamp(as_tensor(limit, device), 0.0, torch.pi / 2))


def photocentre_ratio(flux_ratio, mass_ratio, device=None):
    """R = a0 / a1, the photocentre's orbit over the primary's orbit about the barycentre, for a companion with the flux ratio F2 / F1 and the mass ratio q = M2 / M1: 1 - l (1 + 1 / q) with l = (F2 / F1) / (1 + F2 / F1) the companion's share of the light. R is 1 for a dark companion, 0 where the light follows the mass, and negative where the companion carries more light than its share of the mass (the photocentre is then on the companion's side)."""
    device = pick_device(flux_ratio, mass_ratio, device=device)
    flux_ratio, mass_ratio = as_tensor(flux_ratio, device), as_tensor(mass_ratio, device)
    return 1 - flux_ratio / (1 + flux_ratio) * (1 + 1 / mass_ratio)


def flux_ratio_from_ratio(ratio, mass_ratio, device=None):
    """The flux ratio F2 / F1 that gives the photocentre ratio R = a0 / a1 at the mass ratio q = M2 / M1 (the inverse of photocentre_ratio; l = (1 - R) / (1 + 1 / q) is the companion's share of the light, and R of 1 gives 0)."""
    device = pick_device(ratio, mass_ratio, device=device)
    light = (1 - as_tensor(ratio, device)) / (1 + 1 / as_tensor(mass_ratio, device))
    return light / (1 - light)


AU_KM = 1.495978707e8  # kilometres in an astronomical unit (IAU 2012 Resolution B2)


def reflex_semi_major_axis(k_kms, period_days, ecc, device=None):
    """a1 sin i in AU: the primary's reflex semi-major axis about the barycentre from the semi-amplitude K (km/s), the period (days) and the eccentricity, K P sqrt(1 - e^2) / (2 pi), with no mass assumed (Hilditch 2001, An Introduction to Close Binary Stars, section 2.3). Multiplied by the parallax in mas it is the wobble in mas."""
    device = pick_device(k_kms, period_days, ecc, device=device)
    k, period, e = (as_tensor(x, device) for x in (k_kms, period_days, ecc))
    return k * period * 86400.0 * torch.sqrt(1 - e**2) / (2 * torch.pi) / AU_KM
