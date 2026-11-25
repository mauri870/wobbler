"""The torch package against the frozen NumPy implementation (tests/oracle_numpy.py), run on fixed seeds by oracle_cases.py, in float64, on the CPU and, where there is one, on the GPU; compiled against eager; and the properties of the changes that the port made (the closed-form excess noise, three Danby updates, the padded slices of simulate_fit). The tolerances, fixed before the comparison: orbit functions 1e-10 relative with 1e-12 absolute, the fit 1e-9 relative, the excess noise 1e-9, counts and masks exactly.
Function tests: the checks are on arrays recorded from another implementation and on properties of random inputs, not on a light curve."""
import numpy as np
import pytest
import torch

import oracle_cases
import oracle_numpy
import wobbler as wb
from wobbler import gaiafit

pytestmark = pytest.mark.torch_cpu
F = oracle_cases.build()  # the oracle's inputs and outputs (oracle_cases.py); its digest is checked by test_oracle_digest.py
DEVICES = ["cpu", pytest.param("cuda", marks=[pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU visible")])]
ORBIT = dict(period_days=325.006, ecc=0.2103, t_peri_days=-110.37, omega=1.1048)


def host(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def close(a, b, rtol, atol=1e-12):
    a, b = host(a), host(b)
    ok = np.isfinite(b)
    assert np.array_equal(np.isfinite(a), ok), "the NaN pattern differs"
    assert np.allclose(a[ok], b[ok], rtol=rtol, atol=atol), float(np.max(np.abs(a[ok] - b[ok])))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("compiled", [False, True])
def test_the_orbit_functions_agree_with_the_numpy_outputs(device, compiled):
    wb.set_compile(compiled)
    close(wb.kepler(F["kepler_mean"], F["kepler_ecc"], device=device), F["kepler_E"], 1e-10)
    low = F["kepler_ecc"] <= 0.8
    close(wb.kepler(F["kepler_mean"][low], F["kepler_ecc"][low], device=device), F["kepler_E"][low], 1e-10)  # the batch that takes the three updates
    constants = wb.thiele_innes(F["ti_a"], F["ti_node"], F["ti_omega"], F["ti_incl"], device=device)
    close(torch.stack(constants), F["ti_ABFG"], 1e-10)
    camp = wb.campbell(*constants)
    close(torch.stack([camp["a"], camp["node"], camp["omega"], camp["inclination"]]), F["camp"], 1e-10)
    close(wb.radial_velocity(F["rv_t"], F["rv_period"][:, None], F["rv_ecc"][:, None], F["rv_tp"][:, None], F["rv_omega"][:, None], F["rv_k"][:, None], device=device), F["rv"], 1e-10, 1e-10)
    x, y = wb.photocentre(F["pc_t"], F["rv_period"][:, None], F["rv_ecc"][:, None], F["rv_tp"][:, None], F["pc_a0"][:, None], F["ti_incl"][:, None], F["ti_node"][:, None], F["ti_omega"][:, None], device=device)
    close(torch.stack([x, y]), F["pc_xy"], 1e-10, 1e-10)
    close(wb.true_anomaly(F["true_anomaly_in"], F["true_anomaly_ecc"], device=device), F["true_anomaly"], 1e-10)


@pytest.mark.parametrize("device", DEVICES)
def test_the_mass_functions_agree_with_the_numpy_outputs(device):
    close(wb.mass_function(F["mass_period"], F["mass_k"], F["mass_ecc"], device=device), F["mass_function"], 1e-10)
    close(wb.min_companion_mass(F["mass_f"], F["mass_m1"], device=device), F["min_companion_mass"], 1e-10)
    close(wb.companion_mass(F["mass_f"], F["mass_m1"], F["mass_incl"], device=device), F["companion_mass"], 1e-10)
    close(wb.share_inclined_below(np.linspace(-0.2, 1.8, 30), device=device), F["share_inclined"], 1e-12)


def fit_inputs(device):
    scan = wb.Scan(F["fit_jd"], F["fit_psi"], F["fit_plx"], device=device)
    parameters = {k: F["fit_" + k] for k in ("parallax", "pmra", "pmdec", "a0_mas", "inclination", "node", "period_days", "ecc", "t_peri_days", "omega")}
    return scan, parameters


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("compiled", [False, True])
def test_observe_and_the_fit_agree_with_the_numpy_outputs_with_constant_and_varying_errors(device, compiled):
    wb.set_compile(compiled)
    scan, parameters = fit_inputs(device)
    observed = wb.observe(scan, error=float(F["fit_error"]), z_err=F["fit_z_err"], extra=F["fit_extra"], z_extra=F["fit_z_extra"], **parameters)
    close(observed, F["fit_observed"], 1e-10, 1e-10)
    for label in ("const", "vary"):
        fit = wb.fit_five_parameter(scan.t_yr(), scan.psi, scan.plx_factor, F["fit_observed"], F["fit_%s_error" % label], keep=F["fit_keep"], device=device)
        for key in ("parameters", "parameter_errors", "ruwe", "chi2"):
            close(fit[key], F["fit_%s_%s" % (label, key)], 1e-9)
        close(fit["excess_noise"], F["fit_%s_excess_noise" % label], 1e-9)
        for key in ("degrees_of_freedom", "n_epochs", "visibility_periods", "ok"):
            assert np.array_equal(host(fit[key]), F["fit_%s_%s" % (label, key)]), key


def oracle_draws():
    """The draws and orientations of the NumPy implementation for the seeded case of the recorded outputs (seed 3, 20 parallaxes by 10 orientations), by running the oracle itself."""
    scan = oracle_numpy.Scan(jd=F["fit_jd"], psi=F["fit_psi"], plx_factor=F["fit_plx"])
    sim = oracle_numpy.simulate_fit(scan, oracle_numpy.Orbit(**ORBIT), a1_au=1.2, parallax_mas=F["sim_parallax"], G=12.57, n_orient=10, seed=3)
    return sim.draws, sim.inclination, sim.node


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("compiled", [False, True])
def test_simulate_fit_on_the_draws_of_the_numpy_implementation_agrees_with_its_recorded_outputs(device, compiled):
    wb.set_compile(compiled)
    scan, _ = fit_inputs(device)
    draws, inclination, node = oracle_draws()
    assert np.array_equal(inclination, F["sim_incl"]) and np.array_equal(node, F["sim_node"]), "the oracle's draws are the recorded ones"
    sim = wb.simulate_fit(scan, wb.Orbit(**ORBIT), a1_au=1.2, parallax_mas=F["sim_parallax"], G=12.57, n_orient=10, inclination=inclination, node=node, noise=draws, device=device, formal_error_factor=1.0, uwe_factor=1.0)
    for name in ("DR3", "DR2"):
        for key in ("uwe", "excess_noise", "parallax_mas"):
            close(sim.windows[name][key], F["sim_%s_%s" % (name, "ruwe" if key == "uwe" else key)], 1e-9)
        for key in ("n_epochs", "ok"):
            assert np.array_equal(host(sim.windows[name][key]), F["sim_%s_%s" % (name, key)]), key


@pytest.mark.parametrize("device", DEVICES)
def test_a_run_with_the_noise_drawn_on_the_device_repeats_from_its_seed_and_changes_with_it(device):
    scan, _ = fit_inputs(device)
    kwargs = dict(a1_au=1.2, parallax_mas=np.linspace(0.5, 4, 8), G=12.57, n_orient=9, device=device, formal_error_factor=1.0, uwe_factor=1.0)
    a, b, c = (wb.simulate_fit(scan, wb.Orbit(**ORBIT), seed=seed, **kwargs) for seed in (5, 5, 6))
    assert np.array_equal(host(a.uwe), host(b.uwe), equal_nan=True) and not np.array_equal(host(a.uwe), host(c.uwe), equal_nan=True)
    assert a.draws["seed"] == 5 and a.draws["device"].startswith(device)


@pytest.mark.parametrize("device", DEVICES)
def test_the_compiled_and_the_eager_outputs_agree_to_1e_12_on_the_golden(device):
    scan, _ = fit_inputs(device)
    draws, inclination, node = oracle_draws()
    results = []
    for compiled in (False, True):
        wb.set_compile(compiled)
        results.append(wb.simulate_fit(scan, wb.Orbit(**ORBIT), a1_au=1.2, parallax_mas=F["sim_parallax"], G=12.57, n_orient=10, inclination=inclination, node=node, noise=draws, device=device, formal_error_factor=1.0, uwe_factor=1.0))
    for name in ("DR3", "DR2"):
        for key in ("uwe", "excess_noise", "parallax_mas"):
            close(results[1].windows[name][key], results[0].windows[name][key], 1e-12, 1e-12)


@pytest.mark.parametrize("device", DEVICES)
def test_three_danby_updates_reach_rounding_below_an_eccentricity_of_0_8_and_the_checked_loop_takes_over_above(device):
    rng = np.random.default_rng(11)
    mean = rng.uniform(-20, 20, 200_000)
    for ecc_max in (0.8, 0.95):
        ecc = rng.uniform(0, ecc_max, mean.size)
        anomaly = host(wb.kepler(mean, ecc, device=device))
        residual = anomaly - ecc * np.sin(anomaly) - mean
        assert np.max(np.abs(residual)) < 1e-12, (ecc_max, np.max(np.abs(residual)))
    with pytest.raises(ValueError, match="eccentricity"):
        wb.kepler(1.0, 1.0, device=device)


def test_the_newton_path_for_varying_errors_agrees_with_the_closed_form_for_a_constant_error():
    rng = np.random.default_rng(5)
    cases, n = 300, 60
    sigma2 = torch.full((cases, n), 0.05**2, dtype=torch.float64)
    residual = torch.as_tensor(rng.normal(0, 0.3, (cases, n)))
    r2 = residual**2
    nu = torch.full((cases,), n - 5, dtype=torch.int64)
    reduced = torch.sum(r2 / sigma2, dim=1) / nu
    newton = gaiafit._excess_noise_newton(sigma2, r2, nu, reduced)
    closed = torch.sqrt(torch.clamp(torch.sum(r2, dim=1) / nu - 0.05**2, min=0.0)) * (reduced > 1)
    assert np.allclose(host(newton), host(closed), rtol=1e-11, atol=1e-14)


@pytest.mark.parametrize("device", DEVICES)
def test_the_slices_of_simulate_fit_do_not_change_the_numbers_and_the_last_one_is_padded(device):
    scan, _ = fit_inputs(device)
    cases, n = 77, len(F["fit_jd"])
    rng = np.random.default_rng(9)
    noise = {"keep": rng.uniform(size=(cases, n)) > 0.1, "z_err": rng.normal(size=(cases, n)), "z_extra": rng.normal(size=(cases, n)), "extra": rng.uniform(0, 0.04, cases)}
    kwargs = dict(a1_au=1.2, parallax_mas=np.linspace(0.5, 4, 7), G=12.57, n_orient=11, inclination=rng.uniform(0.2, 1.4, (7, 11)), node=rng.uniform(0, 6, (7, 11)), noise=noise, device=device, formal_error_factor=1.0, uwe_factor=1.0)
    whole = wb.simulate_fit(scan, wb.Orbit(**ORBIT), chunk=10_000, **kwargs)
    for chunk in (1, 5, 32, 77):  # 77 cases: a last slice of every size
        sliced = wb.simulate_fit(scan, wb.Orbit(**ORBIT), chunk=chunk, **kwargs)
        for name in ("DR3", "DR2"):
            for key in ("uwe", "excess_noise", "parallax_mas"):
                close(sliced.windows[name][key], whole.windows[name][key], 1e-13, 1e-13)
            assert np.array_equal(host(sliced.windows[name]["n_epochs"]), host(whole.windows[name]["n_epochs"]))


@pytest.mark.parametrize("device", DEVICES)
def test_a_single_star_with_no_orbit_has_a_ruwe_near_one(device):
    """The property of the five-parameter fit that does not depend on any implementation: pure noise at the epoch error gives a reduced chi-square of one."""
    rng = np.random.default_rng(3)
    n = 191
    scan = wb.Scan(np.sort(rng.uniform(*wb.WINDOWS["DR3"], n)), rng.uniform(0, np.pi, n), rng.uniform(-0.9, 0.9, n), device=device)
    sim = wb.simulate_fit(scan, wb.Orbit(**ORBIT), a1_au=0.0, parallax_mas=[2.0], G=15.0, n_orient=4000, windows={"DR3": wb.WINDOWS["DR3"]}, seed=1, device=device, formal_error_factor=1.0, uwe_factor=1.0)
    ruwe = host(sim.windows["DR3"]["uwe"]).ravel()
    ruwe = ruwe[np.isfinite(ruwe)]
    assert len(ruwe) > 3900
    assert abs(np.mean(ruwe) - 1.0) < 0.005, np.mean(ruwe)
    assert np.mean((ruwe > 0.95) & (ruwe < 1.05)) >= 0.999, np.mean((ruwe > 0.95) & (ruwe < 1.05))


def test_a_scan_of_no_epoch_and_of_one_epoch_is_a_scan():
    empty = wb.Scan([], [], [], device="cpu")
    assert len(empty) == 0 and empty.window(0, 1e9).numel() == 0 and empty.t_yr().numel() == 0
    one = wb.Scan([2457389.0], [0.5], [0.1], device="cpu")
    assert len(one) == 1 and host(one.t_yr()).tolist() == [0.0] and host(one.window(2457000.0, 2458000.0)).tolist() == [True]


@pytest.mark.parametrize("device", DEVICES)
def test_the_visibility_periods_equal_the_numpy_implementations_on_random_masks_and_gaps(device):
    rng = np.random.default_rng(17)
    for n, spacing in ((60, 1.0), (60, 3.9), (200, 0.3), (28, 12.0)):
        t_days = np.sort(np.cumsum(rng.uniform(0.2, 2 * spacing, n)))
        keep = rng.uniform(size=(500, n)) > rng.uniform(0.0, 0.7)
        expected = oracle_numpy.fit_five_parameter.__globals__["_visibility_periods"](t_days, keep)
        got = gaiafit._visibility_periods(torch.as_tensor(t_days, device=device), torch.as_tensor(keep, device=device))
        assert np.array_equal(host(got), expected), (n, spacing)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("compiled", [False, True])
def test_the_unrolled_solver_agrees_with_the_batched_lu_solve_on_normal_matrices(device, compiled):
    wb.set_compile(compiled)
    rng = np.random.default_rng(23)
    n = 120
    design = rng.normal(size=(n, 5)) * np.array([1.0, 0.5, 1.0, 0.5, 0.8])
    weight = rng.uniform(0.5, 4.0, (300, n)) * (rng.uniform(size=(300, n)) > 0.1)
    normal = torch.as_tensor(np.einsum("cn,nj,nk->cjk", weight, design, design), device=device)
    rhs = torch.as_tensor(rng.normal(size=(300, 5)), device=device)
    got = gaiafit._solve_spd5_fused(normal, rhs)
    assert np.allclose(host(got), host(torch.linalg.solve(normal, rhs[:, :, None])[:, :, 0]), rtol=1e-9, atol=1e-12)
