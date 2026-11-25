"""The NumPy implementation of wobbler as it was before the port to torch, frozen as the oracle of the tests: orbit, mass, scan and gaiafit in one module, edited only to be one file (the imports of its siblings and the backend shim). The torch package is checked against it and against the numbers recorded from it (tests/golden/numpy_outputs.npz). Never edited."""
import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def array_namespace(*arrays):
    return np


# ---- scan.py ----
COLUMNS = ("ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]", "scanAngle[rad]", "parallaxFactorAlongScan")
DR3_REFERENCE_JD = 2457389.0  # the reference epoch of DR3 astrometry (J2016.0): gaiamock.rescale_times_astrometry for 'dr3'
DAYS_PER_YEAR = 365.25


@dataclass(frozen=True)
class Scan:
    jd: np.ndarray
    psi: np.ndarray
    plx_factor: np.ndarray

    @classmethod
    def from_gost(cls, path):
        """The scan of the table at `path`, sorted by time."""
        with open(Path(path), newline="") as handle:
            reader = csv.DictReader(handle, skipinitialspace=True)
            missing = [name for name in COLUMNS if name not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"{path} lacks the columns {missing}; it has {reader.fieldnames}")
            rows = [(float(r[COLUMNS[0]]), float(r[COLUMNS[1]]), float(r[COLUMNS[2]])) for r in reader]
        order = np.argsort([r[0] for r in rows], kind="stable")
        table = np.array(rows)[order]
        return cls(jd=table[:, 0], psi=table[:, 1], plx_factor=table[:, 2])

    def window(self, low_jd, high_jd):
        """Boolean mask of the epochs with low_jd < jd < high_jd."""
        return (self.jd > low_jd) & (self.jd < high_jd)

    def t_yr(self, reference_jd=DR3_REFERENCE_JD):
        """The times in years from the reference epoch."""
        return (self.jd - reference_jd) / DAYS_PER_YEAR


# ---- orbit.py ----
DAYS_PER_YEAR = 365.25


@dataclass(frozen=True)
class Orbit:
    """The elements of one orbit or of an array of orbits: period (days), eccentricity, time of periastron (days from the reference epoch JD 2457389.0, Gaia's 2016.0, unless a function takes another reference_jd) and argument of periastron (radians). The node and the inclination are arguments of the functions that need them."""

    period_days: object
    ecc: object
    t_peri_days: object
    omega: object


def kepler(mean_anomaly, ecc, tolerance=1e-12, max_iterations=60):
    """The eccentric anomaly E of E - e sin E = M, for arrays of mean anomaly (radians, any range) and eccentricity in [0, 1): Danby's fourth-order iteration from the start M + 0.85 e sign(sin M) (Danby 1988, Fundamentals of Celestial Mechanics, section 6.6), after M is wrapped into [0, 2 pi).
    gaiamock.solve_kepler_eqn_on_array solves the same equation with a compiled loop at xtol 1e-10."""
    xp = array_namespace(mean_anomaly, ecc)
    mean = xp.asarray(mean_anomaly, dtype=float)
    e = xp.asarray(ecc, dtype=float)
    if xp.any((e < 0) | (e >= 1)):
        raise ValueError("the eccentricity must be in [0, 1)")
    wrapped = xp.mod(mean, 2 * xp.pi)
    anomaly = wrapped + 0.85 * e * xp.sign(xp.sin(wrapped))
    for _ in range(max_iterations):
        sine, cosine = xp.sin(anomaly), xp.cos(anomaly)
        f = anomaly - e * sine - wrapped
        if xp.max(xp.abs(f)) < tolerance:
            break
        f1, f2, f3 = 1 - e * cosine, e * sine, e * cosine
        d1 = -f / f1
        d2 = -f / (f1 + 0.5 * d1 * f2)
        d3 = -f / (f1 + 0.5 * d2 * f2 + d2 * d2 * f3 / 6)
        anomaly = anomaly + d3
    return anomaly + (mean - wrapped)


def thiele_innes(a, node, omega, inclination):
    """(A, B, F, G), the Thiele-Innes constants of a semi-major axis `a` (any unit) with the node, argument of periastron and inclination: the lines of gaiamock.predict_astrometry_binary_in_terms_of_a0."""
    xp = array_namespace(a, node, omega, inclination)
    co, so, cn, sn, ci = xp.cos(omega), xp.sin(omega), xp.cos(node), xp.sin(node), xp.cos(inclination)
    return (a * (co * cn - so * sn * ci), a * (co * sn + so * cn * ci), -a * (so * cn + co * sn * ci), -a * (so * sn - co * cn * ci))


def campbell(A, B, F, G):
    """The elements {a, node, omega, inclination} of the Thiele-Innes constants (the inverse of thiele_innes): the sum and difference of omega and node from the two combinations A + G, B - F and A - G, B + F (Heintz 1978; Halbwachs et al. 2023, A&A 674, A9, appendix), the inclination from tan(i / 2) = sqrt(a (1 - cos i) / (a (1 + cos i))).
    omega and node are determined up to adding pi to both, which leaves the constants unchanged."""
    xp = array_namespace(A, B, F, G)
    plus, minus = xp.hypot(A + G, B - F), xp.hypot(A - G, B + F)  # a (1 + cos i) and a (1 - cos i)
    total = xp.arctan2(B - F, A + G)  # omega + node
    difference = xp.arctan2(-(B + F), A - G)  # omega - node
    return {"a": 0.5 * (plus + minus), "node": xp.mod(0.5 * (total - difference), 2 * xp.pi), "omega": xp.mod(0.5 * (total + difference), 2 * xp.pi), "inclination": 2 * xp.arctan2(xp.sqrt(minus), xp.sqrt(plus))}


def true_anomaly(eccentric_anomaly, ecc):
    xp = array_namespace(eccentric_anomaly, ecc)
    return 2 * xp.arctan2(xp.sqrt(1 + ecc) * xp.sin(eccentric_anomaly / 2), xp.sqrt(1 - ecc) * xp.cos(eccentric_anomaly / 2))


def radial_velocity(t_days, period_days, ecc, t_peri_days, omega, k):
    """The radial velocity k (cos(omega + f) + e cos omega) of a single-lined orbit at times t_days (days from the reference epoch); k is the semi-amplitude, in any velocity unit."""
    xp = array_namespace(t_days, period_days, ecc, t_peri_days, omega, k)
    anomaly = kepler(2 * xp.pi / period_days * (t_days - t_peri_days), ecc)
    return k * (xp.cos(omega + true_anomaly(anomaly, ecc)) + ecc * xp.cos(omega))


def photocentre(t_yr, period_days, ecc, t_peri_days, a0_mas, inclination, node, omega):
    """(x, y), the east and north offsets of the photocentre from the barycentre at times t_yr (years from the reference epoch), for a photocentre orbit of semi-major axis a0_mas: x = B X + G Y, y = A X + F Y with X = cos E - e and Y = sqrt(1 - e^2) sin E,
    as gaiamock.predict_astrometry_binary_in_terms_of_a0."""
    xp = array_namespace(t_yr, period_days, ecc, t_peri_days, a0_mas, inclination, node, omega)
    anomaly = kepler(2 * xp.pi / period_days * (xp.asarray(t_yr, dtype=float) * DAYS_PER_YEAR - t_peri_days), ecc)
    A, B, F, G = thiele_innes(a0_mas, node, omega, inclination)
    X, Y = xp.cos(anomaly) - ecc, xp.sqrt(1 - ecc**2) * xp.sin(anomaly)
    return B * X + G * Y, A * X + F * Y


# ---- mass.py ----
G_SI, SOLAR_MASS_KG = 6.6743e-11, 1.98840987e30  # CODATA 2018 and IAU 2015 nominal solar mass parameter over G


def mass_function(period_days, k_kms, ecc):
    """f(m) = P K^3 (1 - e^2)^(3/2) / (2 pi G), in solar masses, for the period in days, the semi-amplitude K in km/s and the eccentricity (Hilditch 2001, An Introduction to Close Binary Stars, section 2.3)."""
    xp = array_namespace(period_days, k_kms, ecc)
    period_s = xp.asarray(period_days, dtype=float) * 86400.0
    k_ms = xp.asarray(k_kms, dtype=float) * 1000.0
    return period_s * k_ms**3 * (1 - xp.asarray(ecc, dtype=float) ** 2) ** 1.5 / (2 * xp.pi * G_SI) / SOLAR_MASS_KG


def min_companion_mass(f, m1, iterations=100):
    """m2 of m2^3 / (m1 + m2)^2 = f (an edge-on orbit), by bisection on arrays: the least companion mass for the mass function f and the primary's mass m1 (solar masses). A zero mass function has the zero mass."""
    xp = array_namespace(f, m1)
    f, m1 = xp.broadcast_arrays(xp.asarray(f, dtype=float), xp.asarray(m1, dtype=float))
    low = xp.zeros_like(f)
    high = xp.maximum(1.0, 2 * (f + xp.sqrt(f * m1)))
    for _ in range(60):  # the bracket doubles until the cube over the square exceeds f
        small = high**3 / (m1 + high) ** 2 < f
        if not xp.any(small):
            break
        high = xp.where(small, 2 * high, high)
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        below = mid**3 / (m1 + mid) ** 2 < f
        low, high = xp.where(below, mid, low), xp.where(below, high, mid)
    return 0.5 * (low + high) * (f > 0)


def companion_mass(f, m1, inclination):
    """m2 of m2^3 sin^3 i = f (m1 + m2)^2 at the inclination i (radians), the least mass divided out by sin^3 i."""
    xp = array_namespace(f, m1, inclination)
    return min_companion_mass(xp.asarray(f, dtype=float) / xp.sin(inclination) ** 3, m1)


def share_inclined_below(limit):
    """The share of isotropically oriented orbits whose inclination is under `limit` (radians, between 0 and pi / 2): 1 - cos(limit)."""
    xp = array_namespace(limit)
    return 1 - xp.cos(xp.clip(xp.asarray(limit, dtype=float), 0.0, xp.pi / 2))


# ---- gaiafit.py ----
# gaiamock.al_uncertainty_per_ccd_interp: the error in mas per CCD as gaiamock tabulates it (its "EDR3 adjusted" line; Figure 3 of Holl et al. 2023, A&A 674, A10, arXiv:2206.05439, plots the per-CCD uncertainties of Gaia sources beside the EDR3 median formal uncertainty of Lindegren et al. 2021 and gives no values)
G_GRID = (4, 5, 6, 7, 8.2, 8.4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
PER_CCD_ERROR = (0.4, 0.35, 0.15, 0.17, 0.23, 0.13, 0.13, 0.135, 0.125, 0.13, 0.15, 0.23, 0.36, 0.63, 1.05, 2.05, 4.1)
CCD_PER_EPOCH = 8
REJECTED_SHARE = 0.1
BRIGHT_LIMIT, BRIGHT_EXTRA_NOISE = 13.0, 0.04
MINIMUM_VISIBILITY_PERIODS, MINIMUM_EPOCHS, VISIBILITY_GAP_DAYS = 5, 6, 4.0
WINDOWS = {"DR3": (2456891.5, 2457902.0), "DR2": (2456891.5, 2457531.5)}  # JD ranges: gaiamock's DR3 window, and the same logic cut at the end of the DR2 data (2016-05-23) with the DR3 error model


def epoch_error(G):
    """The error in mas of one epoch of a star of magnitude G (a number or an array): the per-CCD error over sqrt(8)."""
    return np.interp(G, G_GRID, PER_CCD_ERROR) / np.sqrt(CCD_PER_EPOCH)


def _visibility_periods(t_days, keep):
    """Per case, the number of visibility periods of the kept epochs: 1 plus the gaps of over VISIBILITY_GAP_DAYS between consecutive kept epochs (t_days sorted ascending)."""
    cases, n = keep.shape
    index = np.where(keep, np.arange(n)[None, :], -1)
    previous = np.concatenate([np.full((cases, 1), -1), np.maximum.accumulate(index, axis=1)[:, :-1]], axis=1)
    has_previous = keep & (previous >= 0)
    gap = t_days[None, :] - t_days[np.clip(previous, 0, None)]
    return 1 + np.sum(has_previous & (gap > VISIBILITY_GAP_DAYS), axis=1)


def _excess_noise(sigma2, residual2, keep, nu, reduced, iterations=60):
    """The extra white noise e (mas, not bounded) with sum r^2 / (sigma^2 + e^2) = nu, per case, by Newton's method on x = e^2 from x = 0: the left side is convex and decreasing in x and positive at 0 for a poor fit, so the iteration rises monotonically to the root without overshooting.
    A case whose reduced chi-square is at most 1 has no excess noise (0)."""
    cases = len(nu)
    x = np.zeros(cases)
    active = np.where(reduced > 1.0)[0]
    if not len(active):
        return x
    r2, s2, k, target = residual2[active], sigma2[active], keep[active], nu[active].astype(float)
    xa = np.zeros(len(active))
    for _ in range(iterations):
        denominator = s2 + xa[:, None]
        value = np.sum(np.where(k, r2 / denominator, 0.0), axis=1) - target
        slope = -np.sum(np.where(k, r2 / denominator**2, 0.0), axis=1)
        step = value / slope
        xa = xa - step
        if np.max(np.abs(step) / np.maximum(xa, 1e-300)) < 1e-15:
            break
    x[active] = np.sqrt(xa)
    return x


def fit_five_parameter(t_yr, psi, plx_factor, observed, error, keep=None):
    """The five-parameter fit of `observed` (cases x epochs, mas) at the epochs t_yr (years, sorted ascending), scan angles psi and parallax factors plx_factor, with the epoch errors `error` (epochs, or cases x epochs) and the optional boolean mask `keep` (cases x epochs) of the epochs each case uses.
    Returns a dict of arrays over cases: parameters (ra offset, pmra, dec offset, pmdec, parallax; mas and mas/yr), parameter_errors (the formal errors inflated as the module docstring says), ruwe, excess_noise (mas), chi2, degrees_of_freedom, n_epochs, visibility_periods and ok (False where the fit
    is not attempted, with NaN in its numbers)."""
    xp = array_namespace(t_yr, psi, plx_factor, observed, error)
    t_yr, psi, plx_factor = (np.asarray(a, dtype=float) for a in (t_yr, psi, plx_factor))
    observed = np.atleast_2d(np.asarray(observed, dtype=float))
    cases, n = observed.shape
    if np.any(np.diff(t_yr) < 0):
        raise ValueError("the epochs must be sorted by time")
    keep = np.ones((cases, n), bool) if keep is None else np.asarray(keep, bool)
    sigma = np.broadcast_to(np.asarray(error, dtype=float), (cases, n))
    design = np.stack([np.sin(psi), t_yr * np.sin(psi), np.cos(psi), t_yr * np.cos(psi), plx_factor], axis=1)  # epochs x 5
    weight = np.where(keep, 1.0 / sigma**2, 0.0)
    normal = np.einsum("cn,nj,nk->cjk", weight, design, design)
    rhs = np.einsum("cn,nj,cn->cj", weight, design, observed)
    n_kept = keep.sum(axis=1)
    periods = _visibility_periods(t_yr * 365.25, keep)
    ok = (periods >= MINIMUM_VISIBILITY_PERIODS) & (n_kept >= MINIMUM_EPOCHS)
    safe = np.where(ok[:, None, None], normal, np.eye(5)[None])
    mu = np.linalg.solve(safe, rhs[:, :, None])[:, :, 0]
    residual = np.where(keep, observed - mu @ design.T, 0.0)
    nu = n_kept - 5
    chi2 = np.sum(weight * residual**2, axis=1)
    reduced = chi2 / np.where(ok, nu, 1)
    unbinned = (n_kept * CCD_PER_EPOCH - n_kept + reduced * nu) / (n_kept * CCD_PER_EPOCH - 5)
    nu_unbinned = n_kept * CCD_PER_EPOCH - 5
    inflation = np.sqrt(unbinned / (1 - 2 / (9 * nu_unbinned)) ** 3)
    covariance = np.linalg.inv(safe)
    errors = inflation[:, None] * np.sqrt(np.einsum("cjj->cj", covariance))
    excess = _excess_noise(sigma**2, residual**2, keep, nu, reduced)
    nan = np.where(ok, 1.0, np.nan)
    return {"parameters": mu * nan[:, None], "parameter_errors": errors * nan[:, None], "ruwe": np.sqrt(unbinned) * nan, "excess_noise": excess * nan, "chi2": chi2 * nan, "degrees_of_freedom": nu, "n_epochs": n_kept, "visibility_periods": periods, "ok": ok}


def observe(scan, parallax, pmra, pmdec, a0_mas, inclination, node, period_days, ecc, t_peri_days, omega, error, z_err, extra, z_extra, reference_jd=DR3_REFERENCE_JD):
    """The along-scan abscissae (cases x epochs, mas) that Gaia measures of cases of a star on a photocentre orbit: the barycentre's proper motion and parallax, the photocentre offset projected on the scan direction (y cos psi + x sin psi), the epoch noise `error` times the standard normals z_err,
    and the unmodelled extra noise `extra` (per case, mas) times z_extra (gaiamock.predict_astrometry_binary_in_terms_of_a0). Every orbital argument is an array over cases or a number; the arrays z_err and z_extra have the shape (cases, epochs)."""
    t = scan.t_yr(reference_jd)
    column = lambda a: np.asarray(a, dtype=float).reshape(-1, 1) if np.ndim(a) else np.asarray(a, dtype=float)  # noqa: E731
    x, y = photocentre(t[None, :], column(period_days), column(ecc), column(t_peri_days), column(a0_mas), column(inclination), column(node), column(omega))
    barycentre = column(pmra) * t * np.sin(scan.psi) + column(pmdec) * t * np.cos(scan.psi) + column(parallax) * scan.plx_factor[None, :]
    return barycentre + y * np.cos(scan.psi) + x * np.sin(scan.psi) + np.asarray(error, dtype=float) * z_err + column(extra) * z_extra


@dataclass
class SimFit:
    """The result of simulate_fit. `windows` maps each window's name to its dict of arrays of shape (parallaxes, orientations): ruwe, excess_noise (mas), parallax_mas (the fitted parallax), n_epochs and ok; `ruwe`, `excess_noise_mas` are those of the first window and `parallax_mas` maps every window to its fitted parallax.
    a0_mas, inclination and node are the photocentre semi-major axis (mas) and the orientation of each case, and `draws` the standard-normal and rejection arrays that made the noise, so that a run can be repeated by passing them back as `noise`."""

    windows: dict
    a0_mas: np.ndarray
    inclination: np.ndarray
    node: np.ndarray
    draws: dict = field(default_factory=dict)

    @property
    def ruwe(self):
        return next(iter(self.windows.values()))["ruwe"]

    @property
    def excess_noise_mas(self):
        return next(iter(self.windows.values()))["excess_noise"]

    @property
    def parallax_mas(self):
        return {name: values["parallax_mas"] for name, values in self.windows.items()}


def simulate_fit(scan, orb, a1_au, parallax_mas, G, flux_ratio=0.0, n_orient=1, inclination=None, node=None, windows=None, pm=(0.0, 0.0), seed=0, noise=None, device="cpu", reference_jd=DR3_REFERENCE_JD):
    """Fit the astrometry that Gaia would record of a star on an orbit, for every true parallax in `parallax_mas` and `n_orient` orientations of the orbit.

    scan: wobbler.Scan of the star. orb: wobbler.Orbit (period_days, ecc, t_peri_days in days from the reference epoch `reference_jd`, omega). a1_au: the projected semi-major axis a1 sin i of the primary's orbit about the barycentre, in AU (from the spectroscopic orbit); the semi-major axis of the primary's orbit is a1_au / sin i, and the photocentre orbit is that of a dark companion.
    parallax_mas: the true parallaxes (array); G: the magnitude of the star, a number or an array with one value per parallax (sets the epoch error and the bright-star term). flux_ratio must be 0: with a luminous companion the photocentre orbit shrinks by a factor that depends on the mass ratio, which with a1 sin i fixed also fixes the inclination, so a drawn inclination would contradict it.
    The orientations: the inclinations are drawn isotropically (cos i uniform on 0 to 1) and the nodes uniformly, or given as `inclination` / `node` arrays of shape (orientations,) or (parallaxes, orientations).
    windows: {name: (jd_low, jd_high)}, default the DR3 and DR2 windows; the noise is drawn once for all the epochs of a case, so the epochs that two windows share carry the same noise.
    pm: (pmra, pmdec) in mas/yr, the barycentre's proper motion (the fit absorbs it). seed: the numpy seed of the draws; noise: a dict {keep, z_err, z_extra, extra} of the draws of a previous run (SimFit.draws), to repeat it or to feed another implementation the same numbers.
    device: "cpu" (the NumPy path); another device raises NotImplementedError."""
    if device != "cpu":
        raise NotImplementedError("only the NumPy path (device='cpu') exists")
    if flux_ratio != 0:
        raise ValueError("flux_ratio must be 0: with a1 sin i and the mass ratio fixed the inclination of a luminous binary is determined, and only a dark companion is simulated")
    windows = WINDOWS if windows is None else windows
    parallax = np.atleast_1d(np.asarray(parallax_mas, dtype=float))
    rng = np.random.default_rng(seed)
    shape = (len(parallax), n_orient)
    magnitude = np.repeat(np.broadcast_to(np.asarray(G, dtype=float), parallax.shape), n_orient)
    cos_i = rng.uniform(0.0, 1.0, shape) if inclination is None else None
    incl = np.arccos(cos_i) if inclination is None else np.broadcast_to(np.asarray(inclination, dtype=float), shape)
    nodes = rng.uniform(0.0, 2 * np.pi, shape) if node is None else np.broadcast_to(np.asarray(node, dtype=float), shape)
    a1_mas = a1_au / np.sin(incl) * parallax[:, None]
    a0 = a1_mas
    n_all = len(scan.jd)
    cases = parallax.size * n_orient
    if noise is None:
        noise = {"keep": rng.uniform(size=(cases, n_all)) > REJECTED_SHARE, "z_err": rng.normal(size=(cases, n_all)), "z_extra": rng.normal(size=(cases, n_all))}
        noise["extra"] = np.where(magnitude < BRIGHT_LIMIT, rng.uniform(0.0, BRIGHT_EXTRA_NOISE, cases), 0.0)
    t = scan.t_yr(reference_jd)
    error = epoch_error(magnitude)[:, None]
    observed = observe(scan, np.repeat(parallax, n_orient), pm[0], pm[1], a0.reshape(-1), incl.reshape(-1), nodes.reshape(-1), orb.period_days, orb.ecc, orb.t_peri_days, orb.omega, error, noise["z_err"], noise["extra"], noise["z_extra"], reference_jd)
    out = {}
    for name, (low, high) in windows.items():
        inside = scan.window(low, high)
        fit = fit_five_parameter(t[inside], scan.psi[inside], scan.plx_factor[inside], observed[:, inside], error, keep=noise["keep"][:, inside])
        out[name] = {"ruwe": fit["ruwe"].reshape(shape), "excess_noise": fit["excess_noise"].reshape(shape), "parallax_mas": fit["parameters"][:, 4].reshape(shape), "n_epochs": fit["n_epochs"].reshape(shape), "ok": fit["ok"].reshape(shape)}
    return SimFit(windows=out, a0_mas=a0, inclination=incl, node=nodes, draws=noise)
