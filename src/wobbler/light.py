"""The orbit and mass functions of wobbler in NumPy, which need no torch: Kepler's equation, the Thiele-Innes constants and their inverse, the radial velocity, the photocentre orbit, the mass function, the companion masses, the photocentre ratio and the reflex semi-major axis.

Each function has the arguments, units and conventions of the function of the same name in wobbler.orbit or wobbler.mass, takes NumPy arrays or numbers (broadcast against each other) and returns NumPy arrays; tests/test_light.py checks every one against the torch function on random inputs. For `pip install wobbler` without torch, `import wobbler.light` is all that is needed.

Conventions are those of gaiamock (El-Badry et al. 2024, OJAp 7, 100, arXiv:2411.00088): `node` is the longitude of the ascending node, `omega` the argument of periastron, the inclination is in radians, x is the offset to the east and y to the north.
"""
import numpy as np

DAYS_PER_YEAR = 365.25
TWO_PI = 2 * np.pi
DANBY_UPDATES = 3  # enough for every eccentricity up to DANBY_ECC_LIMIT
DANBY_ECC_LIMIT = 0.8
G_SI, SOLAR_MASS_KG = 6.6743e-11, 1.98840987e30  # CODATA 2018 and the IAU 2015 nominal solar mass parameter over G
AU_KM = 1.495978707e8  # IAU 2012 Resolution B2


def _danby(anomaly, e, wrapped):
    for _ in range(DANBY_UPDATES):
        sine, cosine = np.sin(anomaly), np.cos(anomaly)
        f = anomaly - e * sine - wrapped
        f1, f2, f3 = 1 - e * cosine, e * sine, e * cosine
        d1 = -f / f1
        d2 = -f / (f1 + 0.5 * d1 * f2)
        d3 = -f / (f1 + 0.5 * d2 * f2 + d2 * d2 * f3 / 6)
        anomaly = anomaly + d3
    return anomaly


def kepler(mean_anomaly, ecc, tolerance=1e-12, max_iterations=60):
    """Matches `wobbler.orbit.kepler`, in NumPy. The eccentric anomaly E of E - e sin E = M, for mean anomalies (radians, any range) and eccentricities in [0, 1): Danby's fourth-order iteration. For a batch whose largest eccentricity is at most 0.8 it makes three updates; above, it iterates until the largest residual is under `tolerance`."""
    mean, e = np.broadcast_arrays(np.asarray(mean_anomaly, dtype=float), np.asarray(ecc, dtype=float))
    if np.any((e < 0) | (e >= 1)):
        raise ValueError("the eccentricity must be in [0, 1)")
    wrapped = np.remainder(mean, TWO_PI)
    anomaly = wrapped + 0.85 * e * np.sign(np.sin(wrapped))
    if mean.size == 0 or float(e.max()) <= DANBY_ECC_LIMIT:
        anomaly = _danby(anomaly, e, wrapped)
    else:
        for _ in range(max_iterations):
            if float(np.abs(anomaly - e * np.sin(anomaly) - wrapped).max()) < tolerance:
                break
            anomaly = anomaly + _step(anomaly, e, wrapped)
    return anomaly + (mean - wrapped)


def _step(anomaly, e, wrapped):
    sine, cosine = np.sin(anomaly), np.cos(anomaly)
    f = anomaly - e * sine - wrapped
    f1, f2, f3 = 1 - e * cosine, e * sine, e * cosine
    d1 = -f / f1
    d2 = -f / (f1 + 0.5 * d1 * f2)
    return -f / (f1 + 0.5 * d2 * f2 + d2 * d2 * f3 / 6)


def thiele_innes(a, node, omega, inclination):
    """Matches `wobbler.orbit.thiele_innes`, in NumPy. (A, B, F, G), the Thiele-Innes constants of a semi-major axis `a` (any unit) with the node, argument of periastron and inclination."""
    a, node, omega, inclination = (np.asarray(x, dtype=float) for x in (a, node, omega, inclination))
    co, so, cn, sn, ci = np.cos(omega), np.sin(omega), np.cos(node), np.sin(node), np.cos(inclination)
    return (a * (co * cn - so * sn * ci), a * (co * sn + so * cn * ci), -a * (so * cn + co * sn * ci), -a * (so * sn - co * cn * ci))


def campbell(A, B, F, G):
    """Matches `wobbler.orbit.campbell`, in NumPy. The elements {a, node, omega, inclination} of the Thiele-Innes constants (the inverse of thiele_innes; Heintz 1978; Halbwachs et al. 2023, A&A 674, A9, appendix A). omega and node are determined up to adding pi to both."""
    A, B, F, G = (np.asarray(x, dtype=float) for x in (A, B, F, G))
    plus, minus = np.hypot(A + G, B - F), np.hypot(A - G, B + F)  # a (1 + cos i) and a (1 - cos i)
    total = np.arctan2(B - F, A + G)  # omega + node
    difference = np.arctan2(-(B + F), A - G)  # omega - node
    return {"a": 0.5 * (plus + minus), "node": np.remainder(0.5 * (total - difference), TWO_PI), "omega": np.remainder(0.5 * (total + difference), TWO_PI), "inclination": 2 * np.arctan2(np.sqrt(minus), np.sqrt(plus))}


def true_anomaly(eccentric_anomaly, ecc):
    """Matches `wobbler.orbit.true_anomaly`, in NumPy."""
    eccentric_anomaly, ecc = np.asarray(eccentric_anomaly, dtype=float), np.asarray(ecc, dtype=float)
    return 2 * np.arctan2(np.sqrt(1 + ecc) * np.sin(eccentric_anomaly / 2), np.sqrt(1 - ecc) * np.cos(eccentric_anomaly / 2))


def radial_velocity(t_days, period_days, ecc, t_peri_days, omega, k):
    """Matches `wobbler.orbit.radial_velocity`, in NumPy. The radial velocity k (cos(omega + f) + e cos omega) of a single-lined orbit at times t_days (days from the reference epoch); k is the semi-amplitude, in any velocity unit."""
    t_days, period_days, ecc, t_peri_days, omega, k = (np.asarray(x, dtype=float) for x in (t_days, period_days, ecc, t_peri_days, omega, k))
    anomaly = kepler(TWO_PI / period_days * (t_days - t_peri_days), ecc)
    return k * (np.cos(omega + true_anomaly(anomaly, ecc)) + ecc * np.cos(omega))


def photocentre(t_yr, period_days, ecc, t_peri_days, a0_mas, inclination, node, omega):
    """Matches `wobbler.orbit.photocentre`, in NumPy. (x, y), the east and north offsets of the photocentre from the barycentre at times t_yr (years from the reference epoch), for a photocentre orbit of semi-major axis a0_mas."""
    t_yr, period_days, ecc, t_peri_days = (np.asarray(x, dtype=float) for x in (t_yr, period_days, ecc, t_peri_days))
    anomaly = kepler(TWO_PI / period_days * (t_yr * DAYS_PER_YEAR - t_peri_days), ecc)
    A, B, F, G = thiele_innes(a0_mas, node, omega, inclination)
    X, Y = np.cos(anomaly) - ecc, np.sqrt(1 - ecc**2) * np.sin(anomaly)
    return B * X + G * Y, A * X + F * Y


def mass_function(period_days, k_kms, ecc):
    """Matches `wobbler.mass.mass_function`, in NumPy. f(m) = P K^3 (1 - e^2)^(3/2) / (2 pi G), in solar masses, for the period in days, the semi-amplitude K in km/s and the eccentricity."""
    period_s = np.asarray(period_days, dtype=float) * 86400.0
    k_ms = np.asarray(k_kms, dtype=float) * 1000.0
    return period_s * k_ms**3 * (1 - np.asarray(ecc, dtype=float) ** 2) ** 1.5 / (TWO_PI * G_SI) / SOLAR_MASS_KG


def min_companion_mass(f, m1, iterations=100):
    """Matches `wobbler.mass.min_companion_mass`, in NumPy. m2 of m2^3 / (m1 + m2)^2 = f (an edge-on orbit), by bisection: the least companion mass for the mass function f and the primary's mass m1 (solar masses). A zero mass function has the zero mass."""
    f, m1 = np.broadcast_arrays(np.asarray(f, dtype=float), np.asarray(m1, dtype=float))
    low = np.zeros_like(f)
    high = np.maximum(2 * (f + np.sqrt(f * m1)), 1.0)
    for _ in range(60):
        small = high**3 / (m1 + high) ** 2 < f
        if not small.any():
            break
        high = np.where(small, 2 * high, high)
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        below = mid**3 / (m1 + mid) ** 2 < f
        low, high = np.where(below, mid, low), np.where(below, high, mid)
    return 0.5 * (low + high) * (f > 0)


def companion_mass(f, m1, inclination):
    """Matches `wobbler.mass.companion_mass`, in NumPy. m2 of m2^3 sin^3 i = f (m1 + m2)^2 at the inclination i (radians)."""
    return min_companion_mass(np.asarray(f, dtype=float) / np.sin(np.asarray(inclination, dtype=float)) ** 3, m1)


def share_inclined_below(limit):
    """Matches `wobbler.mass.share_inclined_below`, in NumPy. The share of isotropically oriented orbits whose inclination is under `limit` (radians, between 0 and pi / 2): 1 - cos(limit)."""
    return 1 - np.cos(np.clip(np.asarray(limit, dtype=float), 0.0, np.pi / 2))


def photocentre_ratio(flux_ratio, mass_ratio):
    """Matches `wobbler.mass.photocentre_ratio`, in NumPy. R = a0 / a1, the photocentre's orbit over the primary's orbit about the barycentre, for a companion with the flux ratio F2 / F1 and the mass ratio q = M2 / M1: 1 - l (1 + 1 / q) with l = (F2 / F1) / (1 + F2 / F1). R is 1 for a dark companion, 0 where the light follows the mass, and negative where the companion carries more light than its share of the mass."""
    flux_ratio, mass_ratio = np.asarray(flux_ratio, dtype=float), np.asarray(mass_ratio, dtype=float)
    return 1 - flux_ratio / (1 + flux_ratio) * (1 + 1 / mass_ratio)


def flux_ratio_from_ratio(ratio, mass_ratio):
    """Matches `wobbler.mass.flux_ratio_from_ratio`, in NumPy. The flux ratio F2 / F1 that gives the photocentre ratio R = a0 / a1 at the mass ratio q = M2 / M1 (the inverse of photocentre_ratio)."""
    light = (1 - np.asarray(ratio, dtype=float)) / (1 + 1 / np.asarray(mass_ratio, dtype=float))
    return light / (1 - light)


def reflex_semi_major_axis(k_kms, period_days, ecc):
    """Matches `wobbler.mass.reflex_semi_major_axis`, in NumPy. a1 sin i in AU: the primary's reflex semi-major axis about the barycentre from the semi-amplitude K (km/s), the period (days) and the eccentricity, K P sqrt(1 - e^2) / (2 pi), with no mass assumed. Multiplied by the parallax in mas it is the wobble in mas."""
    k, period, e = (np.asarray(x, dtype=float) for x in (k_kms, period_days, ecc))
    return k * period * 86400.0 * np.sqrt(1 - e**2) / TWO_PI / AU_KM
