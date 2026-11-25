"""wobbler.gaiafit: the epoch noise of Gaia's astrometry, the batched five-parameter fit (parameters, errors, RUWE, excess noise) and the forward model that fits an orbiting star for many cases at once. Test functions: the cases are arrays of constructed epochs and an independent per-case implementation of the same equations."""
import numpy as np
import pytest

from wobbler import gaiafit, orbit, scan

pytestmark = pytest.mark.torch_cpu


def _np(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else value


class _Sim:
    """simulate_fit's result with the tensors as NumPy arrays, so that the checks below read as they did on the NumPy implementation."""

    def __init__(self, sim):
        self.windows = {name: {key: _np(v) for key, v in w.items()} for name, w in sim.windows.items()}
        self.a0_mas, self.inclination, self.node, self.draws = _np(sim.a0_mas), _np(sim.inclination), _np(sim.node), sim.draws

    uwe = property(lambda self: next(iter(self.windows.values()))["uwe"])
    excess_noise_mas = property(lambda self: next(iter(self.windows.values()))["excess_noise"])
    parallax_mas = property(lambda self: {name: w["parallax_mas"] for name, w in self.windows.items()})


def fit_five_parameter(*args, **kwargs):
    return {key: _np(v) for key, v in gaiafit.fit_five_parameter(*args, **kwargs).items()}


def simulate_fit(*args, **kwargs):
    return _Sim(gaiafit.simulate_fit(*args, **{"formal_error_factor": 1.0, "uwe_factor": 1.0, **kwargs}))  # the raw behaviour of the NumPy implementation these checks were written on


def epoch_error(g):
    return _np(gaiafit.epoch_error(g))

WINDOWS = {"DR3": (2456891.5, 2457902.0), "DR2": (2456891.5, 2457531.5)}


def make_scan(seed=0, n=200):
    rng = np.random.default_rng(seed)
    jd = np.sort(rng.uniform(2456850.0, 2457950.0, n))
    psi = rng.uniform(0, np.pi, n)
    plx = rng.uniform(-0.9, 0.9, n)
    return scan.Scan(jd=jd, psi=psi, plx_factor=plx)


def reference_fit(t, psi, plx, observed, error):
    """The five-parameter weighted least squares of one case, as a loop of plain numpy (an oracle that shares no code with the batched fit): returns ruwe, mu, sigma_mu."""
    design = np.vstack([np.sin(psi), t * np.sin(psi), np.cos(psi), t * np.cos(psi), plx]).T
    weight = np.diag(1 / error**2)
    normal = design.T @ weight @ design
    mu = np.linalg.solve(normal, design.T @ weight @ observed)
    residual = observed - design @ mu
    n = len(observed)
    nu = n - 5
    nu_unbinned = n * 8 - 5
    chi2_binned = np.sum(residual**2 / error**2) / nu
    chi2_unbinned = (n * 8 - n + chi2_binned * (n - 5)) / (n * 8 - 5)
    ruwe = np.sqrt(chi2_unbinned)
    inflation = np.sqrt(chi2_unbinned / (1 - 2 / (9 * nu_unbinned)) ** 3)
    return ruwe, mu, inflation * np.sqrt(np.diag(np.linalg.inv(normal)))


def test_the_epoch_error_is_the_per_ccd_table_over_the_square_root_of_eight():
    assert epoch_error(13.0) == pytest.approx(0.13 / np.sqrt(8))
    assert epoch_error(12.5) == pytest.approx(0.5 * (0.125 + 0.13) / np.sqrt(8))
    assert epoch_error(np.array([4.0, 20.0])).tolist() == pytest.approx([0.4 / np.sqrt(8), 4.1 / np.sqrt(8)])


def test_the_batched_fit_equals_the_case_by_case_fit_with_different_masks():
    s = make_scan(1)
    t = s.t_yr().numpy()
    rng = np.random.default_rng(5)
    cases, n = 12, len(t)
    error = np.full(n, 0.05)
    observed = rng.normal(0, 0.2, (cases, n)) + 0.3 * np.sin(s.psi.numpy()) * t
    keep = rng.uniform(size=(cases, n)) > 0.1
    batched = fit_five_parameter(t, s.psi.numpy(), s.plx_factor.numpy(), observed, error, keep=keep)
    for i in range(cases):
        k = keep[i]
        ruwe, mu, sigma = reference_fit(t[k], s.psi.numpy()[k], s.plx_factor.numpy()[k], observed[i, k], error[k])
        assert batched["ruwe"][i] == pytest.approx(ruwe, rel=1e-9), i
        assert np.allclose(batched["parameters"][i], mu, rtol=1e-8, atol=1e-10)
        assert np.allclose(batched["parameter_errors"][i], sigma, rtol=1e-8)
    assert batched["ok"].all()


def test_noise_free_data_of_a_single_star_gives_back_its_parameters_and_the_floor_of_ruwe():
    s = make_scan(2)
    t = s.t_yr().numpy()
    truth = np.array([0.4, -1.2, 0.7, 2.0, 3.3])  # ra offset, pmra, dec offset, pmdec, parallax
    design = np.vstack([np.sin(s.psi.numpy()), t * np.sin(s.psi.numpy()), np.cos(s.psi.numpy()), t * np.cos(s.psi.numpy()), s.plx_factor.numpy()]).T
    fit = fit_five_parameter(t, s.psi.numpy(), s.plx_factor.numpy(), (design @ truth)[None, :], np.full(len(t), 0.05))
    assert np.allclose(fit["parameters"][0], truth, atol=1e-9)
    n = len(t)
    assert fit["ruwe"][0] == pytest.approx(np.sqrt((n * 8 - n) / (n * 8 - 5)), rel=1e-12), "no residual: the binning term only"


def test_fewer_than_five_visibility_periods_is_not_ok_and_has_no_numbers():
    """The fit needs 5 visibility periods (gaps over 4 days between the kept epochs) and 6 epochs."""
    t_yr = np.arange(40) * 1.95 / 365.25  # one epoch every 1.95 days: no gap over 4 days between kept neighbours
    rng = np.random.default_rng(1)
    psi, plx = rng.uniform(0, np.pi, 40), rng.uniform(-0.9, 0.9, 40)
    keep = np.ones((3, 40), bool)
    keep[1, ::3] = False  # gaps of 3.9 days between kept neighbours: still not over 4
    keep[2, :] = False
    keep[2, [0, 10, 20, 30, 39, 1]] = True  # six epochs, gaps over 4 days: 5 visibility periods
    observed = rng.normal(0, 0.1, (3, 40))
    fit = fit_five_parameter(t_yr, psi, plx, observed, np.full(40, 0.05), keep=keep)
    assert fit["visibility_periods"].tolist() == [1, 1, 5]
    assert fit["ok"].tolist() == [False, False, True]
    assert np.isnan(fit["ruwe"][0]) and np.isnan(fit["parameters"][0]).all()


def test_the_excess_noise_solves_the_chi_square_equation_and_is_zero_for_a_good_fit():
    s = make_scan(4)
    t = s.t_yr().numpy()
    rng = np.random.default_rng(6)
    n = len(t)
    error = np.full(n, 0.05)
    good = rng.normal(0, 0.03, n)[None, :]
    bad = rng.normal(0, 0.4, (1, n))
    fit_good = fit_five_parameter(t, s.psi.numpy(), s.plx_factor.numpy(), good, error)
    assert fit_good["excess_noise"][0] == 0.0
    fit_bad = fit_five_parameter(t, s.psi.numpy(), s.plx_factor.numpy(), bad, error)
    e = fit_bad["excess_noise"][0]
    design = np.vstack([np.sin(s.psi.numpy()), t * np.sin(s.psi.numpy()), np.cos(s.psi.numpy()), t * np.cos(s.psi.numpy()), s.plx_factor.numpy()]).T
    residual = bad[0] - design @ fit_bad["parameters"][0]
    # the fit that defines the residuals uses the epoch errors alone, as the reference solution does
    assert np.sum(residual**2 / (error**2 + e**2)) == pytest.approx(n - 5, rel=1e-6)


def test_the_orbiting_star_with_no_orbit_has_a_ruwe_near_one_and_an_unbiased_parallax():
    s = make_scan(7)
    result = simulate_fit(s, orbit.Orbit(period_days=325.0, ecc=0.2, t_peri_days=-110.0, omega=1.1), a1_au=0.0, parallax_mas=np.array([2.0]), G=14.0, n_orient=400, windows=WINDOWS, seed=11)
    ruwe = result.windows["DR3"]["uwe"][0]
    assert 0.9 < np.median(ruwe) < 1.1
    assert abs(np.mean(result.windows["DR3"]["parallax_mas"][0]) - 2.0) < 4 * np.std(result.windows["DR3"]["parallax_mas"][0]) / np.sqrt(400)


def test_a_wide_orbit_raises_the_ruwe_and_the_same_seed_gives_the_same_numbers():
    s = make_scan(8)
    args = dict(a1_au=1.0, parallax_mas=np.array([2.0]), G=14.0, n_orient=50, windows=WINDOWS, seed=3)
    o = orbit.Orbit(period_days=325.0, ecc=0.2, t_peri_days=-110.0, omega=1.1)
    a = simulate_fit(s, o, **args)
    b = simulate_fit(s, o, **args)
    assert np.array_equal(a.windows["DR3"]["uwe"], b.windows["DR3"]["uwe"], equal_nan=True)
    c = simulate_fit(s, o, **{**args, "seed": 4})
    assert not np.array_equal(a.windows["DR3"]["uwe"], c.windows["DR3"]["uwe"], equal_nan=True)
    assert np.nanmedian(a.windows["DR3"]["uwe"]) > 3.0, "a 2 mas photocentre orbit is far above the 0.05 mas epoch noise"


def test_the_two_windows_use_the_same_noise_on_the_epochs_they_share_and_dr2_has_fewer_epochs():
    s = make_scan(9)
    o = orbit.Orbit(period_days=325.0, ecc=0.2, t_peri_days=-110.0, omega=1.1)
    result = simulate_fit(s, o, a1_au=0.5, parallax_mas=np.array([2.0]), G=14.0, n_orient=30, windows=WINDOWS, seed=5)
    assert set(result.windows) == {"DR3", "DR2"}
    assert result.windows["DR3"]["n_epochs"].min() > result.windows["DR2"]["n_epochs"].max()
    assert result.windows["DR3"]["uwe"].shape == (1, 30) == result.windows["DR2"]["parallax_mas"].shape


def test_noise_arrays_passed_in_reproduce_a_run_and_make_the_seed_irrelevant():
    s = make_scan(10)
    o = orbit.Orbit(period_days=325.0, ecc=0.2, t_peri_days=-110.0, omega=1.1)
    n = len(s.jd)
    rng = np.random.default_rng(21)
    noise = {"keep": rng.uniform(size=(12, n)) > 0.1, "z_err": rng.normal(size=(12, n)), "z_extra": rng.normal(size=(12, n)), "extra": rng.uniform(0, 0.04, 12)}
    kwargs = dict(a1_au=0.5, parallax_mas=np.array([2.0, 1.5]), G=14.0, n_orient=6, windows=WINDOWS, inclination=rng.uniform(0.3, 1.5, (2, 6)), node=rng.uniform(0, 6, (2, 6)), noise=noise)
    first = simulate_fit(s, o, seed=21, **kwargs)
    again = simulate_fit(s, o, seed=99, **kwargs)
    assert np.array_equal(first.windows["DR3"]["uwe"], again.windows["DR3"]["uwe"], equal_nan=True)
    assert np.array_equal(first.windows["DR2"]["parallax_mas"], again.windows["DR2"]["parallax_mas"], equal_nan=True)


def test_a_luminous_companion_needs_the_primary_mass_and_a_flux_ratio_of_zero_is_the_dark_companion():
    """A flux ratio above zero needs the primary's mass, which turns a1 sin i into the companion's mass at each inclination; a flux ratio of zero is the dark companion, with or without the mass."""
    s = make_scan(11)
    o = orbit.Orbit(period_days=325.0, ecc=0.2, t_peri_days=-110.0, omega=1.1)
    kwargs = dict(a1_au=1.0, parallax_mas=np.array([2.0]), G=14.0, n_orient=1, inclination=np.array([np.pi / 2]), windows=WINDOWS, seed=1)
    with pytest.raises(ValueError, match="primary_mass"):
        simulate_fit(s, o, flux_ratio=0.3, **kwargs)
    dark = simulate_fit(s, o, flux_ratio=0.0, **kwargs)
    assert dark.a0_mas[0, 0] == pytest.approx(1.0 * 2.0, rel=1e-12)
    assert np.array_equal(simulate_fit(s, o, flux_ratio=0.0, primary_mass=1.0, **kwargs).windows["DR3"]["uwe"], dark.windows["DR3"]["uwe"], equal_nan=True)


def test_the_result_carries_the_ruwe_and_excess_noise_of_the_first_window_and_the_parallax_of_each():
    s = make_scan(12)
    o = orbit.Orbit(period_days=325.0, ecc=0.2, t_peri_days=-110.0, omega=1.1)
    result = simulate_fit(s, o, a1_au=0.5, parallax_mas=np.array([2.0]), G=14.0, n_orient=5, windows=WINDOWS, seed=2)
    assert np.array_equal(result.uwe, result.windows["DR3"]["uwe"], equal_nan=True)
    assert np.array_equal(result.excess_noise_mas, result.windows["DR3"]["excess_noise"], equal_nan=True)
    assert set(result.parallax_mas) == {"DR3", "DR2"}


def test_observe_is_the_barycentre_plus_the_projected_photocentre_plus_the_noise():
    s = make_scan(13)
    rng = np.random.default_rng(2)
    cases, n = 4, len(s.jd.numpy())
    z_err, z_extra = rng.normal(size=(cases, n)), rng.normal(size=(cases, n))
    extra = np.array([0.0, 0.01, 0.02, 0.03])
    parameters = dict(parallax=np.array([1.0, 2.0, 3.0, 4.0]), pmra=np.array([0.5, -1.0, 0.0, 2.0]), pmdec=np.array([0.1, 0.2, -0.3, 0.4]), a0_mas=np.array([0.0, 1.0, 2.0, 3.0]), inclination=np.array([0.5, 1.0, 1.5, 2.0]), node=np.array([0.1, 1.0, 2.0, 3.0]),
                      period_days=np.array([300.0, 500.0, 700.0, 900.0]), ecc=np.array([0.0, 0.2, 0.4, 0.6]), t_peri_days=np.array([0.0, 10.0, 20.0, 30.0]), omega=np.array([0.3, 1.3, 2.3, 3.3]))
    observed = gaiafit.observe(s, error=0.05, z_err=z_err, extra=extra, z_extra=z_extra, **parameters)
    t = s.t_yr().numpy()
    for i in range(cases):
        x, y = (_np(v) for v in orbit.photocentre(t, *(float(parameters[k][i]) for k in ("period_days", "ecc", "t_peri_days", "a0_mas", "inclination", "node", "omega"))))
        by_hand = parameters["pmra"][i] * t * np.sin(s.psi.numpy()) + parameters["pmdec"][i] * t * np.cos(s.psi.numpy()) + parameters["parallax"][i] * s.plx_factor.numpy() + y * np.cos(s.psi.numpy()) + x * np.sin(s.psi.numpy()) + 0.05 * z_err[i] + extra[i] * z_extra[i]
        assert np.allclose(_np(observed)[i], by_hand, atol=1e-12)


def test_the_excess_noise_equals_the_bisection_of_the_same_equation():
    """The batched Newton iteration against a per-case bisection on [0, 5] mas (60 halvings), over cases from a good fit to a very poor one."""
    s = make_scan(14)
    t = s.t_yr().numpy()
    rng = np.random.default_rng(8)
    n = len(t)
    scales = np.array([0.02, 0.05, 0.1, 0.3, 1.0, 3.0])
    observed = rng.normal(size=(len(scales), n)) * scales[:, None]
    error = np.full(n, 0.05)
    keep = rng.uniform(size=observed.shape) > 0.1
    fit = fit_five_parameter(t, s.psi.numpy(), s.plx_factor.numpy(), observed, error, keep=keep)
    for i in range(len(scales)):
        k = keep[i]
        design = np.vstack([np.sin(s.psi.numpy()), t * np.sin(s.psi.numpy()), np.cos(s.psi.numpy()), t * np.cos(s.psi.numpy()), s.plx_factor.numpy()]).T[k]
        weight = 1 / error[k] ** 2
        mu = np.linalg.solve(design.T @ (weight[:, None] * design), design.T @ (weight * observed[i, k]))
        r = observed[i, k] - design @ mu
        nu = k.sum() - 5
        low, high = 0.0, 5.0
        if np.sum(r**2 * weight) / nu <= 1:
            expected = 0.0
        else:
            for _ in range(60):
                mid = (low + high) / 2
                low, high = (mid, high) if np.sum(r**2 / (error[k] ** 2 + mid**2)) > nu else (low, mid)
            expected = (low + high) / 2
        assert fit["excess_noise"][i] == pytest.approx(expected, rel=1e-10, abs=1e-12), (i, scales[i])


def test_the_excess_noise_is_not_clipped_at_5_mas():
    """Residuals of 20 mas against errors of 0.05 mas have an excess noise of about 20 mas; the root of sum r^2 / (sigma^2 + e^2) = N - 5 by bisection on [0, 100] mas is the reference."""
    s = make_scan(15)
    t = s.t_yr().numpy()
    n = len(t)
    observed = np.random.default_rng(9).normal(size=(1, n)) * 20.0
    error = np.full(n, 0.05)
    fit = fit_five_parameter(t, s.psi.numpy(), s.plx_factor.numpy(), observed, error)
    design = np.vstack([np.sin(s.psi.numpy()), t * np.sin(s.psi.numpy()), np.cos(s.psi.numpy()), t * np.cos(s.psi.numpy()), s.plx_factor.numpy()]).T
    weight = 1 / error**2
    mu = np.linalg.solve(design.T @ (weight[:, None] * design), design.T @ (weight * observed[0]))
    r = observed[0] - design @ mu
    low, high = 0.0, 100.0
    for _ in range(80):
        mid = (low + high) / 2
        low, high = (mid, high) if np.sum(r**2 / (error**2 + mid**2)) > n - 5 else (low, mid)
    assert fit["excess_noise"][0] == pytest.approx((low + high) / 2, rel=1e-9)
    assert fit["excess_noise"][0] > 5.0


def test_the_epoch_error_and_the_bright_star_noise_follow_an_array_of_magnitudes():
    magnitudes = np.array([12.0, 14.0, 12.9, 13.0])
    assert np.allclose(epoch_error(magnitudes), [epoch_error(float(g)) for g in magnitudes])
    s = make_scan(16)
    o = orbit.Orbit(period_days=325.0, ecc=0.2, t_peri_days=-110.0, omega=1.1)
    kwargs = dict(a1_au=0.5, parallax_mas=np.array([2.0, 2.0, 2.0, 2.0]), n_orient=5, inclination=np.pi / 2, node=0.3, windows=WINDOWS, seed=4)
    together = simulate_fit(s, o, G=magnitudes, **kwargs)
    extra = together.draws["extra"].reshape(4, 5)
    assert np.all(extra[[0, 2]] > 0) and np.all(extra[[1, 3]] == 0), "the extra noise is drawn for the stars brighter than G = 13 only"
    n = len(s.jd)
    rng = np.random.default_rng(3)
    noise = {"keep": rng.uniform(size=(20, n)) > 0.1, "z_err": rng.normal(size=(20, n)), "z_extra": rng.normal(size=(20, n)), "extra": extra.reshape(-1)}
    together = simulate_fit(s, o, G=magnitudes, noise=noise, **kwargs)
    for i, g in enumerate(magnitudes):
        rows = slice(5 * i, 5 * i + 5)
        alone = simulate_fit(s, o, G=float(g), noise={key: value[rows] for key, value in noise.items()}, **{**kwargs, "parallax_mas": np.array([2.0])})
        assert np.allclose(alone.windows["DR3"]["uwe"][0], together.windows["DR3"]["uwe"][i], rtol=1e-12, atol=0, equal_nan=True), g  # the matrix products are not bitwise independent of the batch size


def test_the_run_size_from_which_the_default_device_is_the_gpu_is_the_measured_one():
    # Measured with 16 CPU threads on simulate_fit with the noise drawn on the device: the compiled CPU path takes 4.59 ms for 700 cases and the GPU 3.65 ms, and the GPU is faster from there up; at 500 cases the two are equal (3.60 and 3.64 ms).
    assert gaiafit.GPU_MIN_CASES == 700


def luminous_kwargs(**extra):
    s = make_scan(11)
    o = orbit.Orbit(period_days=325.0, ecc=0.2, t_peri_days=-110.0, omega=1.1)
    kwargs = dict(a1_au=1.0, parallax_mas=np.array([2.0]), G=14.0, n_orient=3, inclination=np.array([[np.pi / 2, 1.2, 0.9]]), node=np.array([[0.3, 1.0, 2.0]]), windows=WINDOWS, seed=4)
    return s, o, {**kwargs, **extra}


def test_the_photocentre_orbit_of_a_luminous_companion_is_the_reflex_orbit_times_one_minus_the_light_fraction_over_the_mass_fraction():
    from scipy.optimize import brentq

    s, o, kwargs = luminous_kwargs()
    result = simulate_fit(s, o, flux_ratio=0.5, primary_mass=1.0, **kwargs)
    f = 1.0**3 / (325.0 / 365.25) ** 2  # the mass function in solar masses, from a1 sin i = 1 AU
    for k, incl in enumerate(kwargs["inclination"][0]):
        m2 = brentq(lambda m: m**3 * np.sin(incl) ** 3 - f * (1 + m) ** 2, 1e-6, 1e3)
        light = 0.5 / 1.5
        expected = (1.0 / np.sin(incl)) * 2.0 * (1 - light * (1 + 1.0 / m2))
        assert result.a0_mas[0, k] == pytest.approx(expected, rel=1e-9), (incl, m2)


def test_when_the_light_fraction_equals_the_mass_fraction_the_photocentre_does_not_move_and_the_fit_is_the_single_star_one():
    s, o, kwargs = luminous_kwargs()
    balanced = simulate_fit(s, o, flux_ratio=lambda m2: m2 / 1.0, primary_mass=1.0, **kwargs)  # F2 / F1 = M2 / M1 gives light fraction = mass fraction
    still = simulate_fit(s, o, **{**kwargs, "a1_au": 0.0})
    assert np.abs(balanced.a0_mas).max() < 1e-9
    assert np.allclose(balanced.windows["DR3"]["uwe"], still.windows["DR3"]["uwe"], rtol=1e-9, atol=0, equal_nan=True)


def test_a_photocentre_that_follows_the_companion_is_the_same_orbit_half_a_turn_on_and_the_same_fit():
    s, o, kwargs = luminous_kwargs()
    flipped = simulate_fit(s, o, flux_ratio=10.0, primary_mass=1.0, **kwargs)
    assert (flipped.a0_mas < 0).all(), "a companion much brighter than its mass share puts the photocentre on its own side"
    # the dark companion whose a0 is the same size and whose node is half a turn on: a1 sin i = |a0| sin i / parallax per case
    a1_equal = np.abs(flipped.a0_mas) / 2.0 * np.sin(kwargs["inclination"])
    equal = simulate_fit(s, o, **{**kwargs, "a1_au": a1_equal, "node": kwargs["node"] + np.pi})
    assert np.allclose(equal.a0_mas, -flipped.a0_mas, rtol=1e-12)
    assert np.allclose(flipped.windows["DR3"]["uwe"], equal.windows["DR3"]["uwe"], rtol=1e-9, equal_nan=True)
