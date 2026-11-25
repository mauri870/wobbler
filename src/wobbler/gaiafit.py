"""Gaia's single-star five-parameter astrometric fit, forward-modelled for many orbiting stars at once: the epoch noise, the batched weighted least squares with RUWE, the astrometric excess noise, and the fitted parallax of each time window.

The model of the data follows gaiamock (El-Badry et al. 2024, arXiv:2411.00088; https://github.com/kareemelbadry/gaiamock, gaiamock/gaiamock.py):
- the per-CCD along-scan error as a function of G is al_uncertainty_per_ccd_interp; one epoch is the mean of 8 CCDs, so its error is that over sqrt(8);
- a random 10 percent of the epochs is rejected (predict_astrometry_binary_in_terms_of_a0);
- stars brighter than G = 13 carry an extra unmodelled noise drawn uniformly from 0 to 0.04 mas per star;
- the fit and RUWE are get_5par_solution_and_sigma_5d_max: the weighted least squares of the columns sin psi, t sin psi, cos psi, t cos psi and the along-scan parallax factor; the reduced chi-square of the binned epochs is corrected to the unbinned count
  (predict_reduced_chi2_unbinned_data, 8 CCDs an epoch) and its square root is the RUWE; the formal errors are inflated by sqrt(chi2_unbinned / (1 - 2 / (9 nu_unbinned))^3);
- the fit is attempted only with at least 5 visibility periods (gaps over 4 days between epochs) and 6 epochs (fit_5par_solution_only).
What the three outputs are, and are not:
- ruwe is gaiamock's square root of the unbinned reduced chi-square of the five-parameter fit. It is not Gaia's catalogued RUWE, which divides by the normalisation u0(G, colour).
- excess_noise is the root e of sum r^2 / (sigma^2 + e^2) = N - 5 over the residuals r of that first fit, not Gaia's astrometric_excess_noise, which Gaia computes from the CCD-level observations. It is 0 when the reduced chi-square is at most 1 and is not bounded above.
- The "DR2" window is gaiamock's DR3 window logic (the same error model, rejection and bright-star noise) cut at the end of DR2's data; gaiamock has no DR2 model of its own, so it is not DR2's own error model.
"""
from dataclasses import dataclass, field

import numpy as np
import torch

from ._compile import hot
from . import _tensor
from . import unbinned as _unbinned
from . import mass as _mass
from . import ruwe as _ruwe_module
from .uwe_factor import calibrated_uwe, orbit_factor as calibrated_orbit_factor, unbinned_factor as calibrated_unbinned_factor, uwe_factor as calibrated_uwe_factor
from ._tensor import as_tensor, pick_device
from .orbit import Orbit, kepler, TWO_PI
from .scan import DR3_REFERENCE_JD

# gaiamock.al_uncertainty_per_ccd_interp: the error in mas per CCD as gaiamock tabulates it (its "EDR3 adjusted" line; Figure 3 of Holl et al. 2023, A&A 674, A10, arXiv:2206.05439, plots the per-CCD uncertainties of Gaia sources beside the EDR3 median formal uncertainty of Lindegren et al. 2021 and gives no values)
G_GRID = (4, 5, 6, 7, 8.2, 8.4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
PER_CCD_ERROR = (0.4, 0.35, 0.15, 0.17, 0.23, 0.13, 0.13, 0.135, 0.125, 0.13, 0.15, 0.23, 0.36, 0.63, 1.05, 2.05, 4.1)
CCD_PER_EPOCH = 8
REJECTED_SHARE = 0.1
BRIGHT_LIMIT, BRIGHT_EXTRA_NOISE = 13.0, 0.04
MINIMUM_VISIBILITY_PERIODS, MINIMUM_EPOCHS, VISIBILITY_GAP_DAYS = 5, 6, 4.0
GPU_MIN_CASES = 700  # the run size from which device=None takes the GPU: per case, simulate_fit on the GPU is at least 20 percent faster than on 16 CPU threads from 700 cases and at every larger size, in each of three runs (GPU over CPU 0.55, 0.67 and 0.62 at 700 cases, 0.84 in one run at 500, 0.39 to 0.42 at 1,000, 0.07 at 20,000; RX 7900 XTX against Ryzen 9 9950X3D, benchmarks/crossover.py)
PARAMETER_KEYS = ("pmra_mas_yr", "pmdec_mas_yr", "parallax_error_mas", "pmra_error_mas_yr", "pmdec_error_mas_yr")  # the extra outputs of simulate_fit(parameters=True)
WINDOWS = {"DR3": (2456891.5, 2457902.0), "DR2": (2456891.5, 2457531.5)}  # JD ranges: gaiamock's DR3 window, and the same logic cut at the end of the DR2 data (2016-05-23) with the DR3 error model


def epoch_error(G, device=None):
    """The error in mas of one epoch of a star of magnitude G (a number, an array or a tensor): the per-CCD error over sqrt(8), linearly interpolated in G and constant beyond the ends of the table."""
    device = pick_device(G, device=device)
    grid, table = as_tensor(G_GRID, device), as_tensor(PER_CCD_ERROR, device)
    g = torch.clamp(as_tensor(G, device), grid[0], grid[-1])
    upper = torch.clamp(torch.searchsorted(grid, g.contiguous(), right=True), 1, len(grid) - 1)
    x0, x1, y0, y1 = grid[upper - 1], grid[upper], table[upper - 1], table[upper]
    return (y0 + (g - x0) / (x1 - x0) * (y1 - y0)) / CCD_PER_EPOCH**0.5


def _visibility_periods(t_days, keep):
    """Per case, the number of visibility periods of the kept epochs: 1 plus the gaps of over VISIBILITY_GAP_DAYS between consecutive kept epochs (t_days sorted ascending). A kept epoch starts a new period when it has a kept predecessor and none of the epochs within the gap before it is kept; the epochs within the gap are a fixed range of indices for each epoch, so a cumulative sum of the mask counts them."""
    first = torch.searchsorted(t_days, t_days - VISIBILITY_GAP_DAYS)  # the first epoch not more than the gap before each epoch
    kept = keep.to(torch.int64)
    before = torch.cumsum(kept, dim=1) - kept  # the kept epochs strictly before each epoch
    within_gap = before - before[:, first]  # the kept epochs from `first` up to each epoch, exclusive
    return 1 + torch.sum(keep & (before > 0) & (within_gap == 0), dim=1)


def _excess_noise_newton(sigma2, residual2, nu, reduced, iterations=60, tolerance=1e-13):
    """The extra white noise e (mas) with sum r^2 / (sigma^2 + e^2) = nu, per case, for epoch errors that differ between the epochs of a case, by Newton's method on x = e^2 from x = 0: the left side is convex and decreasing in x and positive at 0, so the iterates increase to the root. Each case stops when its own relative step is under `tolerance`. residual2 is zero at the epochs a case does not use, so they add nothing. A case whose reduced chi-square is at most 1 has no excess noise (0)."""
    x = torch.zeros(len(nu), dtype=sigma2.dtype, device=sigma2.device)
    active = torch.nonzero(reduced > 1.0).squeeze(1)
    if active.numel() == 0:
        return x
    r2, s2, target = residual2[active], sigma2[active], nu[active].to(sigma2.dtype)
    xa = torch.zeros(len(active), dtype=sigma2.dtype, device=sigma2.device)
    done = torch.zeros(len(active), dtype=torch.bool, device=sigma2.device)
    for _ in range(iterations):
        denominator = s2 + xa[:, None]
        value = torch.sum(r2 / denominator, dim=1) - target
        slope = -torch.sum(r2 / denominator**2, dim=1)
        step = torch.where(done, torch.zeros_like(value), value / slope)
        xa = xa - step
        done = done | (step.abs() / torch.clamp(xa, min=1e-300) < tolerance)
        if bool(done.all()):
            break
    x[active] = torch.sqrt(xa)
    return x


def _solve_spd5(a, b):
    """The solutions of the symmetric positive definite 5 x 5 systems a x = b (cases x 5 x 5 and cases x 5) by Gaussian elimination without pivoting, written out over the cases so that one fused kernel does it: a batched LU solve costs a fifth of a pass on a GPU and does not fuse."""
    m = [[a[:, i, j] for j in range(5)] for i in range(5)]
    r = [b[:, i] for i in range(5)]
    for k in range(5):
        for i in range(k + 1, 5):
            f = m[i][k] / m[k][k]
            for j in range(k + 1, 5):
                m[i][j] = m[i][j] - f * m[k][j]
            r[i] = r[i] - f * r[k]
    x = [None] * 5
    for k in reversed(range(5)):
        total = r[k]
        for j in range(k + 1, 5):
            total = total - m[k][j] * x[j]
        x[k] = total / m[k][k]
    return torch.stack(x, dim=1)


_solve_spd5_fused = hot(_solve_spd5)


def _normal_equations(weight, outer, design, observed):
    """The normal matrices (cases x 5 x 5) and right-hand sides (cases x 5) of the weighted least squares of every case, as two matrix products: sum_n w_n d_nj d_nk and sum_n w_n d_nj y_n."""
    return (weight @ outer).reshape(weight.shape[0], 5, 5), (weight * observed) @ design


def _residuals(observed, mu, design, keep, weight):
    residual = torch.where(keep, observed - mu @ design.T, torch.zeros_like(observed))
    return residual, torch.sum(weight * residual**2, dim=1), torch.sum(residual**2, dim=1)


_normal_equations_fused, _residuals_fused = hot(_normal_equations), hot(_residuals)


def fit_five_parameter(t_yr, psi, plx_factor, observed, error, keep=None, errors=True, u0=None, device=None, binned=True):
    """The five-parameter fit of `observed` (cases x epochs, mas) at the epochs t_yr (years, sorted ascending), scan angles psi and parallax factors plx_factor, with the epoch errors `error` (a number, epochs, cases x 1 or cases x epochs) and the optional boolean mask `keep` (cases x epochs) of the epochs each case uses; the arrays are NumPy arrays or tensors.
    Returns a dict of tensors over cases: parameters (ra offset, pmra, dec offset, pmdec, parallax; mas and mas/yr), parameter_errors (the formal errors inflated as the module docstring says; only with errors=True), uwe (the unit weight error, gaiamock's RUWE), ruwe (uwe divided by `u0`, one value per case, when it is given, else the same as uwe), excess_noise (mas), chi2, degrees_of_freedom, n_epochs, visibility_periods and ok (False where the fit is not attempted, with NaN in its numbers).
    binned: True (the default) when each epoch is the mean of 8 CCDs, as in gaiamock's default mode: the reduced chi-square is corrected to the unbinned count. False for epochs that are single CCD measurements (gaiamock's unbinned mode): the reduced chi-square is used as it is.
    The excess noise has a closed form when the error is the same at every epoch of a case (e^2 = sum r^2 / nu - sigma^2); otherwise Newton's method."""
    device = pick_device(t_yr, psi, plx_factor, observed, error, device=device)
    t_yr, psi, plx_factor, observed = (as_tensor(a, device) for a in (t_yr, psi, plx_factor, observed))
    observed = torch.atleast_2d(observed)
    cases, n = observed.shape
    if n > 1 and bool((t_yr[1:] < t_yr[:-1]).any()):
        raise ValueError("the epochs must be sorted by time")
    keep = torch.ones((cases, n), dtype=torch.bool, device=device) if keep is None else torch.as_tensor(keep, dtype=torch.bool, device=device)
    error = as_tensor(error, device)
    constant = error.ndim == 0 or error.shape[-1] == 1 or bool((error == error[..., :1]).all())
    sigma = torch.broadcast_to(error, (cases, n))
    design = torch.stack([torch.sin(psi), t_yr * torch.sin(psi), torch.cos(psi), t_yr * torch.cos(psi), plx_factor], dim=1)  # epochs x 5
    outer = (design[:, :, None] * design[:, None, :]).reshape(n, 25)
    weight = torch.where(keep, 1.0 / sigma**2, torch.zeros_like(sigma))
    normal, rhs = _normal_equations_fused(weight, outer, design, observed)
    n_kept = keep.sum(dim=1)
    periods = _visibility_periods(t_yr * 365.25, keep)
    ok = (periods >= MINIMUM_VISIBILITY_PERIODS) & (n_kept >= MINIMUM_EPOCHS)
    safe = torch.where(ok[:, None, None], normal, torch.eye(5, dtype=normal.dtype, device=device)[None])
    mu = _solve_spd5_fused(safe, rhs)
    residual, chi2, total = _residuals_fused(observed, mu, design, keep, weight)
    nu = n_kept - 5
    reduced = chi2 / torch.where(ok, nu, torch.ones_like(nu))
    bins = CCD_PER_EPOCH if binned else 1
    unbinned = (n_kept * bins - n_kept + reduced * nu) / (n_kept * bins - 5)
    if constant:
        sigma0 = sigma[:, 0]
        x = total / torch.where(ok, nu, torch.ones_like(nu)) - sigma0**2
        excess = torch.sqrt(torch.where(reduced > 1.0, torch.clamp(x, min=0.0), torch.zeros_like(x)))
    else:
        excess = _excess_noise_newton(sigma**2, residual**2, nu, reduced)
    nan = torch.where(ok, torch.ones_like(chi2), torch.full_like(chi2, float("nan")))
    uwe = torch.sqrt(unbinned) * nan
    ruwe = uwe if u0 is None else uwe / as_tensor(u0, device)
    out = {"parameters": mu * nan[:, None], "uwe": uwe, "ruwe": ruwe, "excess_noise": excess * nan, "chi2": chi2 * nan, "degrees_of_freedom": nu, "n_epochs": n_kept, "visibility_periods": periods, "ok": ok}
    if errors:
        nu_unbinned = (n_kept * bins - 5).to(chi2.dtype)
        inflation = torch.sqrt(unbinned / (1 - 2 / (9 * nu_unbinned)) ** 3)
        covariance = torch.linalg.inv(safe)
        out["parameter_errors"] = inflation[:, None] * torch.sqrt(torch.diagonal(covariance, dim1=1, dim2=2)) * nan[:, None]
    return out


def _abscissae(anomaly, ecc, A, B, F, G, t, psi, plx, parallax, pmra, pmdec, error, z_err, extra, z_extra):
    X, Y = torch.cos(anomaly) - ecc, torch.sqrt(1 - ecc**2) * torch.sin(anomaly)
    x, y = B * X + G * Y, A * X + F * Y
    barycentre = pmra * t * torch.sin(psi) + pmdec * t * torch.cos(psi) + parallax * plx
    return barycentre + y * torch.cos(psi) + x * torch.sin(psi) + error * z_err + extra * z_extra


_abscissae_fused = hot(_abscissae)


def observe(scan, parallax, pmra, pmdec, a0_mas, inclination, node, period_days, ecc, t_peri_days, omega, error, z_err, extra, z_extra, reference_jd=DR3_REFERENCE_JD, device=None):
    """The along-scan abscissae (cases x epochs, mas) that Gaia measures of cases of a star on a photocentre orbit: the barycentre's proper motion and parallax, the photocentre offset projected on the scan direction (y cos psi + x sin psi), the epoch noise `error` (mas, a number or one per case) times z_err,
    and the unmodelled extra noise `extra` (per case, mas) times z_extra (gaiamock.predict_astrometry_binary_in_terms_of_a0). Every orbital argument is an array over cases or a number; the arrays z_err and z_extra have the shape (cases, epochs)."""
    device = torch.device(device) if device is not None else (z_err.device if isinstance(z_err, torch.Tensor) else scan.device)
    t = scan.t_yr(reference_jd).to(device)
    psi, plx = scan.psi.to(device), scan.plx_factor.to(device)

    def column(a):
        a = as_tensor(a, device)
        return a.reshape(-1, 1) if a.ndim else a

    from .orbit import thiele_innes

    ecc_c = column(ecc)
    anomaly = kepler(2 * torch.pi / column(period_days) * (t[None, :] * 365.25 - column(t_peri_days)), ecc_c, device=device)
    A, B, F, G = thiele_innes(column(a0_mas), column(node), column(omega), column(inclination))
    return _abscissae_fused(anomaly, ecc_c, A, B, F, G, t, psi, plx, column(parallax), column(pmra), column(pmdec), column(error), as_tensor(z_err, device), column(extra), as_tensor(z_extra, device))


@dataclass
class SimFit:
    """The result of simulate_fit. `windows` maps each window's name to its dict of tensors of shape (parallaxes, orientations): uwe (the unit weight error), ruwe (uwe divided by u0, only when simulate_fit was given nu_eff), excess_noise (mas), parallax_mas (the fitted parallax), n_epochs and ok; `ruwe`, `excess_noise_mas` are those of the first window and `parallax_mas` is a dict of the fitted parallaxes by window.
    a0_mas, inclination and node are the photocentre semi-major axis (mas) and the orientation of each case, and `draws` the noise arrays that were given, or, for draws made on the device, their seed, chunk size, device and the unmodelled extra noise of each case (mas)."""

    windows: dict
    a0_mas: torch.Tensor
    inclination: torch.Tensor
    node: torch.Tensor
    draws: dict = field(default_factory=dict)

    @property
    def ruwe(self):
        first = next(iter(self.windows.values()))
        if "ruwe" not in first:
            raise AttributeError("there is no ruwe: simulate_fit was not given nu_eff, which u0 needs; the unit weight error is .uwe")
        return first["ruwe"]

    @property
    def uwe(self):
        return next(iter(self.windows.values()))["uwe"]

    @property
    def excess_noise_mas(self):
        return next(iter(self.windows.values()))["excess_noise"]

    @property
    def parallax_mas(self):
        return {name: values["parallax_mas"] for name, values in self.windows.items()}


def simulate_fit(scan, orb, a1_au, parallax_mas, G, flux_ratio=0.0, n_orient=1, inclination=None, node=None, windows=None, pm=(0.0, 0.0), seed=0, noise=None, device=None, reference_jd=DR3_REFERENCE_JD, chunk=None, nu_eff=None, formal_error_factor=1.0, uwe_factor=None, primary_mass=None, parameters=False, binned=None, position=None, sky_map=None):
    """Fit the astrometry that Gaia would record of a star on an orbit, for every true parallax in `parallax_mas` and `n_orient` orientations of the orbit, on `device`: 'cpu', 'cuda', or None (the default), which takes the GPU when torch sees one and the run has at least GPU_MIN_CASES cases, else the CPU.

    scan: wobbler.Scan of the star. orb: wobbler.Orbit (period_days, ecc, t_peri_days in days from the reference epoch `reference_jd`, omega). a1_au: the projected semi-major axis a1 sin i of the primary's orbit about the barycentre, in AU (from the spectroscopic orbit).
    parallax_mas: the true parallaxes (array); G: the magnitude of the star, a number or an array with one value per parallax (sets the epoch error and the bright-star term). flux_ratio: the companion's flux over the primary's in the G band, a number or a function of the companion's mass in solar masses (0 is a dark companion, the default). With a luminous companion the photocentre's orbit is the primary's reflex orbit times R = 1 - l (1 + M1 / M2), with l the companion's share of the light (wobbler.photocentre_ratio), and `primary_mass` (solar masses) is needed: M2 follows from a1 sin i, the period and the inclination of each case by the mass function. R is negative where the companion carries more light than its share of the mass, and a0_mas of the result is then negative (the same orbit half a turn on). The epoch error uses the G of the star, which is the total light.
    The orientations: the inclinations are drawn isotropically (cos i uniform on 0 to 1) and the nodes uniformly, or given as `inclination` / `node` arrays of shape (orientations,) or (parallaxes, orientations).
    windows: {name: (jd_low, jd_high)}, default the DR3 and DR2 windows; the noise is drawn once for all the epochs of a case, so the epochs that two windows share carry the same noise.
    pm: (pmra, pmdec) in mas/yr, the barycentre's proper motion (the fit absorbs it). seed: the seed of the draws. With noise=None the noise is drawn on `device` by a torch.Generator seeded from the seed and the index of the slice, so a run is reproducible on its device and with its chunk size, and a CPU and a GPU give different draws for the same seed; the orientations (when not given) are drawn on the host by numpy's generator. noise: a dict {keep, z_err, z_extra, extra} of draws (NumPy arrays or tensors of shape (cases, epochs) and (cases,)), to feed any backend the same cases; with it the draws are not generated.
    nu_eff: the effective wavenumber nu_eff_used_in_astrometry of the star (a number or one per parallax; per micron): with it each window has `ruwe`, the unit weight error divided by Gaia DR3's u0(G, nu_eff) (wobbler.ruwe.u0); without it there is no `ruwe`, because u0 depends on the colour, and the result carries `uwe`, the unit weight error, which is always given. formal_error_factor: the factor f by which the errors that the fit weights with are smaller than the errors that the noise is drawn with, sigma_formal = sigma_real / f (a number, one per parallax, or a function of the magnitude G that returns it); 1 (the default) weights with the errors of the noise, as gaiamock does; a larger f raises the unit weight error, by less than f. uwe_factor: how the simulated unit weight error and RUWE are brought to the scale of Gaia DR3's, which the raw simulation falls short of (up to 1.9 times near G 11.3). The default, "orbit" when binned, gives each case sqrt(f(G)^2 + B(G)^2 (w^2 - 1)) with w its raw unit weight error, f the single-star factor (wobbler.uwe_factor) and B the orbit factor (wobbler.uwe_factor.orbit_factor), both measured on DR3 stars. With `binned=False` the default is wobbler.unbinned_factor(G), the factor of single stars measured for the unbinned mode (it multiplies the raw value; it has no orbit part, and the binned mode is better for stars with a detected orbit). A number, one per parallax, or a function of the magnitude G multiplies the raw value instead (the function wobbler.uwe_factor gives f times w); 1 gives the raw simulation. Their measured ranges are in the docstring of wobbler.uwe_factor and in tables/uwe_orbit_factor_dr3.json.
    binned: None (the default) takes `scan.binned`, which Scan.from_position sets from its own `binned`; True for gaiamock's default mode, one epoch per field-of-view transit. False for its unbinned mode (wobbler.unbinned), for single stars and weak orbits; the map file healpix_16_med_ruwe.npz must be downloaded (gaiamock-download): `scan` holds one measurement per CCD (Scan.from_position(..., binned=False)), the noise of a case is drawn with the per-CCD error of G plus the term of the median RUWE at `position` (ra, dec in degrees; default `scan.position`, which Scan.from_position sets; needed) plus 0.01 mas times one standard normal draw per case, there is no bright-star extra noise, and the fit weights with the per-CCD error and does not correct the reduced chi-square for binning. sky_map: the path of healpix_16_med_ruwe.npz, else the file in $GAIAMOCK_DATA_DIR or ~/.gaiamock. A `noise` dict for the unbinned mode holds keep, z_err and jitter (cases,); z_extra and extra are not used.
    parameters: True adds to each window's result the fitted proper motion (pmra_mas_yr, pmdec_mas_yr, with the barycentre's `pm` in them) and the formal errors of the parallax, pmra and pmdec (parallax_error_mas, pmra_error_mas_yr, pmdec_error_mas_yr), inflated as the module docstring says by the raw chi-square (uwe_factor does not scale them); False, the default, returns what it always did.
    chunk: the cases fitted in one slice (the dispatches of a slice are short, and the arrays of a slice stay in cache on a CPU); the last slice is padded to the same size, so that a compiled function sees one shape. Default 100,000 on a GPU and 10,000 on a CPU (the sizes that were fastest for the compiled path in the benchmark of the package)."""
    binned = scan.binned if binned is None else binned
    position = scan.position if position is None else position
    if not binned and position is None:
        raise ValueError("binned=False needs the position (ra, dec in degrees) of the star: the map of the median RUWE is read at it")
    uwe_factor = (("orbit" if binned else calibrated_unbinned_factor) if uwe_factor is None else uwe_factor)
    luminous = callable(flux_ratio) or np.any(np.asarray(flux_ratio) != 0)
    if luminous and primary_mass is None:
        raise ValueError("a flux_ratio other than 0 needs primary_mass: it turns a1 sin i into the companion's mass at each inclination, which sets the share of the mass that the light is compared with")
    cases = np.atleast_1d(np.asarray(parallax_mas)).size * n_orient
    if device is None:
        device = torch.device("cuda" if cases >= GPU_MIN_CASES and _tensor.default_device().type == "cuda" else "cpu")
    device = torch.device(device)
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
    if luminous:
        mass_function = np.asarray(a1_au, dtype=float) ** 3 / (orb.period_days / 365.25) ** 2  # solar masses, Kepler's third law in AU, years and solar masses
        companion = _mass.companion_mass(np.broadcast_to(mass_function, shape), primary_mass, incl, device="cpu").numpy()
        ratio = np.asarray(flux_ratio(companion) if callable(flux_ratio) else np.broadcast_to(np.asarray(flux_ratio, dtype=float), shape), dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            a0 = np.where(companion > 0, a1_mas * _mass.photocentre_ratio(ratio, companion / primary_mass, device="cpu").numpy(), 0.0)
    n_all = len(scan)
    cases = parallax.size * n_orient
    drawn_here = noise is None
    scan = scan.to(device)
    t = scan.t_yr(reference_jd)
    error_host = epoch_error(magnitude, device="cpu").numpy() * (1.0 if binned else CCD_PER_EPOCH**0.5)  # the error that the fit weights with: per epoch, or per CCD
    if binned:
        noise_host = error_host
    else:
        jitter = rng.standard_normal(cases) if drawn_here else np.asarray(noise["jitter"], dtype=float)
        noise_host = _unbinned.realistic_error(error_host, position[0], position[1], jitter, _unbinned.read_sky_map(sky_map))
    parallax_cases, a0_cases, incl_cases, node_cases = np.repeat(parallax, n_orient), a0.reshape(-1), incl.reshape(-1), nodes.reshape(-1)
    chunk = min((100_000 if device.type == "cuda" else 10_000) if chunk is None else int(chunk), max(cases, 1))  # a run of fewer cases than a slice is one slice of its own size
    masks = {name: scan.window(low, high) for name, (low, high) in windows.items()}
    results = {name: {"excess_noise": torch.empty(cases, dtype=torch.float64, device=device), "parallax_mas": torch.empty(cases, dtype=torch.float64, device=device),
                      "n_epochs": torch.empty(cases, dtype=torch.int64, device=device), "ok": torch.empty(cases, dtype=torch.bool, device=device), "uwe": torch.empty(cases, dtype=torch.float64, device=device)} for name in windows}
    if nu_eff is not None:
        for values in results.values():
            values["ruwe"] = torch.empty(cases, dtype=torch.float64, device=device)
    if parameters:
        for values in results.values():
            for key in PARAMETER_KEYS:
                values[key] = torch.empty(cases, dtype=torch.float64, device=device)
    orbit_form = isinstance(uwe_factor, str) and uwe_factor == "orbit"
    uwe_cases = np.zeros(cases) if orbit_form else np.asarray(uwe_factor(magnitude) if callable(uwe_factor) else np.broadcast_to(np.asarray(uwe_factor, dtype=float), parallax.shape).repeat(n_orient) if np.ndim(uwe_factor) else np.full(cases, float(uwe_factor)), dtype=float)
    factor_cases = np.asarray(formal_error_factor(magnitude) if callable(formal_error_factor) else np.broadcast_to(np.asarray(formal_error_factor, dtype=float), parallax.shape).repeat(n_orient) if np.ndim(formal_error_factor) else np.full(cases, float(formal_error_factor)), dtype=float)
    nu_cases = None if nu_eff is None else np.broadcast_to(np.asarray(nu_eff, dtype=float), parallax.shape).repeat(n_orient)

    extras = []
    in_flight = []  # the host arrays of this slice's copies to the device, kept until the slice has finished: a copy from pageable memory must not outlive its source

    def take(array, rows):
        if isinstance(array, torch.Tensor):  # draws already on a device: no transfer, only the slice
            index = torch.arange(rows[0], rows[0] + len(rows), device=array.device).clamp_(max=len(array) - 1)
            return array[index].to(device)
        source = np.ascontiguousarray(array[rows])
        in_flight.append(source)
        return torch.as_tensor(source, device=device)

    for start in range(0, cases, chunk):
        stop = min(start + chunk, cases)
        rows = np.minimum(np.arange(start, start + chunk), cases - 1)  # the last slice is padded with its last case
        count = stop - start
        if drawn_here:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed) * 1_000_003 + start // chunk)
            draw = {"keep": torch.rand(chunk, n_all, generator=generator, dtype=torch.float64, device=device) > REJECTED_SHARE,
                    "z_err": torch.randn(chunk, n_all, generator=generator, dtype=torch.float64, device=device)}
            if binned:
                draw["z_extra"] = torch.randn(chunk, n_all, generator=generator, dtype=torch.float64, device=device)
                bright = take(magnitude, rows) < BRIGHT_LIMIT
                draw["extra"] = torch.where(bright, torch.rand(chunk, generator=generator, dtype=torch.float64, device=device) * BRIGHT_EXTRA_NOISE, torch.zeros(chunk, dtype=torch.float64, device=device))
            else:  # no unmodelled noise in gaiamock's unbinned mode
                draw["z_extra"], draw["extra"] = torch.zeros((), dtype=torch.float64, device=device), torch.zeros(chunk, dtype=torch.float64, device=device)
        else:
            draw = {key: take(noise[key], rows) for key in (("keep", "z_err", "z_extra", "extra") if binned else ("keep", "z_err"))}
            if not binned:
                draw["z_extra"], draw["extra"] = torch.zeros((), dtype=torch.float64, device=device), torch.zeros(chunk, dtype=torch.float64, device=device)
        observed = observe(scan, take(parallax_cases, rows), pm[0], pm[1], take(a0_cases, rows), take(incl_cases, rows), take(node_cases, rows), orb.period_days, orb.ecc, orb.t_peri_days, orb.omega,
                           take(noise_host, rows)[:, None], draw["z_err"], draw["extra"], draw["z_extra"], reference_jd=reference_jd, device=device)
        keep = draw["keep"]
        if drawn_here:
            extras.append(draw["extra"][:count].cpu())
        error = take(error_host, rows)[:, None] / take(factor_cases, rows)[:, None]  # the errors that the fit weights with
        u0_cases = None if nu_cases is None else _ruwe_module.u0(take(magnitude, rows), take(nu_cases, rows), device=device)
        for name, inside in masks.items():
            fit = fit_five_parameter(t[inside], scan.psi[inside], scan.plx_factor[inside], observed[:, inside], error, keep=keep[:, inside], errors=parameters, u0=u0_cases, device=device, binned=binned)
            for key, value in ((("ruwe", fit["ruwe"]),) if nu_eff is not None else ()) + (("uwe", fit["uwe"]), ("excess_noise", fit["excess_noise"]), ("parallax_mas", fit["parameters"][:, 4]), ("n_epochs", fit["n_epochs"]), ("ok", fit["ok"])):
                results[name][key][start:stop] = value[:count]
            if parameters:
                for key, value in zip(PARAMETER_KEYS, (fit["parameters"][:, 1], fit["parameters"][:, 3], *(fit["parameter_errors"][:, k] for k in (4, 1, 3)))):
                    results[name][key][start:stop] = value[:count]
        if device.type == "cuda":
            torch.cuda.synchronize(device)  # the queue holds at most one slice of dispatches
        in_flight.clear()
    if orbit_form:  # sqrt(f^2 + B^2 (w^2 - 1)) of the simulated unit weight error w of each case; the RUWE follows from the same ratio
        single, orbit = torch.as_tensor(calibrated_uwe_factor(magnitude), device=device), torch.as_tensor(calibrated_orbit_factor(magnitude), device=device)
        for values in results.values():
            raw = values["uwe"]
            calibrated = torch.sqrt(torch.clamp(single**2 + orbit**2 * (raw**2 - 1.0), min=0.0))
            if "ruwe" in values:
                values["ruwe"] = values["ruwe"] * calibrated / raw
            values["uwe"] = calibrated
    else:
        uwe_by_case = torch.as_tensor(uwe_cases, device=device)
        for values in results.values():
            for key in ("uwe", "ruwe"):
                if key in values:
                    values[key] = values[key] * uwe_by_case
    out = {name: {key: value.reshape(shape) for key, value in values.items()} for name, values in results.items()}
    return SimFit(windows=out, a0_mas=torch.as_tensor(a0, device=device), inclination=torch.as_tensor(np.array(incl), device=device), node=torch.as_tensor(np.array(nodes), device=device), draws=noise if not drawn_here else {"seed": int(seed), "chunk": chunk, "device": str(device), "extra": torch.cat(extras).numpy()})
