"""wobbler.epoch_astrometry: the grid search for an astrometric orbit in along-scan abscissae, for many target vectors at once. The reference is a plain NumPy loop over the grid and the targets (the algorithm of the original per-star search: for each period, eccentricity and phase the four Thiele-Innes columns, projected off the base model, solved by the normal equations), which shares no code with the batched version."""
import math

import numpy as np
import pytest

from wobbler import epoch_astrometry

pytestmark = pytest.mark.torch_cpu


def make_data(seed=0, points=40, span=900.0):
    rng = np.random.default_rng(seed)
    jd = np.sort(rng.uniform(2457000.0, 2457000.0 + span, points))
    theta = rng.uniform(0, 360, points)
    factor = rng.uniform(-0.9, 0.9, points)
    sigma = rng.uniform(0.05, 0.2, points)
    tau = (jd - 2457500.0) / 365.25
    s, c = np.sin(np.radians(theta)), np.cos(np.radians(theta))
    base = np.column_stack([s, c, factor, s * tau, c * tau])
    return jd, theta, sigma, base, rng


def reference(jd, theta, sigma, base, targets, bands, step):
    """(chi2, period, eccentricity, periastron) per target, by loops."""
    q, _ = np.linalg.qr(base / sigma[:, None])
    cosine, sine = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    out = []
    for w in targets.T:
        t = w / sigma
        t = t - q @ (q.T @ t)
        best = (math.inf, None, None, None)
        for low, high, eccentricities, phases in bands:
            for frequency in np.arange(low, high + step, step):
                period = 1.0 / frequency
                for ecc in eccentricities:
                    for k in range(phases):
                        tp = jd.min() + period * k / phases
                        mean = 2 * math.pi * (jd - tp) / period
                        E = mean + ecc * np.sin(mean)
                        for _ in range(80):
                            E = E - (E - ecc * np.sin(E) - mean) / (1 - ecc * np.cos(E))
                        X, Y = np.cos(E) - ecc, math.sqrt(1 - ecc**2) * np.sin(E)
                        columns = np.column_stack([cosine * X, sine * X, cosine * Y, sine * Y]) / sigma[:, None]
                        columns = columns - q @ (q.T @ columns)
                        solution = np.linalg.solve(columns.T @ columns + 1e-12 * np.eye(4), columns.T @ t)
                        chi2 = float(t @ t - solution @ (columns.T @ t))
                        if chi2 < best[0]:
                            best = (chi2, period, float(ecc), float(tp))
        out.append(best)
    return out


SMALL = dict(period_range=(300.0, 900.0), floor=100.0, eccentricities=(0.0, 0.5), phases=4, short_eccentricities=(0.0, 0.3), short_phases=2, step_share=2.0)


def small_bands(span, jd):
    step = 1.0 / (SMALL["step_share"] * float(np.ptp(jd)))
    bands = [(1.0 / 900.0, 1.0 / 300.0, SMALL["eccentricities"], SMALL["phases"]), (1.0 / 300.0 + step, 1.0 / 100.0, SMALL["short_eccentricities"], SMALL["short_phases"])]
    return bands, step


def test_the_grid_search_equals_the_loop_reference_for_several_targets_with_both_bands():
    jd, theta, sigma, base, rng = make_data(1)
    targets = rng.normal(size=(len(jd), 3)) * sigma[:, None] * 2.0
    bands, step = small_bands(900.0, jd)
    expected = reference(jd, theta, sigma, base, targets, bands, step)
    result = epoch_astrometry.grid_search(jd, theta, sigma, base, targets, device="cpu", **SMALL)
    for k, (chi2, period, ecc, tp) in enumerate(expected):
        assert float(result["chi2"][k]) == pytest.approx(chi2, rel=1e-8, abs=1e-8), k
        assert (float(result["period"][k]), float(result["eccentricity"][k])) == pytest.approx((period, ecc), rel=1e-9), k
        assert float(result["periastron"][k]) == pytest.approx(tp, rel=1e-12), k


def test_the_improvement_of_a_target_does_not_depend_on_the_other_targets():
    jd, theta, sigma, base, rng = make_data(2)
    targets = rng.normal(size=(len(jd), 4)) * sigma[:, None]
    together = epoch_astrometry.grid_search(jd, theta, sigma, base, targets, device="cpu", **SMALL)
    alone = epoch_astrometry.grid_search(jd, theta, sigma, base, targets[:, 2:3], device="cpu", **SMALL)
    assert float(together["chi2"][2]) == pytest.approx(float(alone["chi2"][0]), rel=1e-12)
    assert float(together["period"][2]) == float(alone["period"][0])


def test_chi2_without_an_orbit_is_the_residual_of_the_base_model_and_an_orbit_never_makes_it_larger():
    jd, theta, sigma, base, rng = make_data(3)
    targets = rng.normal(size=(len(jd), 2)) * sigma[:, None]
    result = epoch_astrometry.grid_search(jd, theta, sigma, base, targets, device="cpu", **SMALL)
    weighted = base / sigma[:, None]
    residual = targets / sigma[:, None] - weighted @ np.linalg.lstsq(weighted, targets / sigma[:, None], rcond=None)[0]
    np.testing.assert_allclose(result["chi2_base"].numpy(), (residual**2).sum(0), rtol=1e-9)
    assert np.all(result["chi2"].numpy() <= result["chi2_base"].numpy() + 1e-9)


def test_an_injected_orbit_is_found_at_its_period_and_is_a_large_improvement():
    jd, theta, sigma, base, rng = make_data(4, points=60)
    sigma = np.full(len(jd), 0.05)
    period, ecc, tp = 500.0, 0.0, jd.min() + 120.0
    mean = 2 * math.pi * (jd - tp) / period
    X, Y = np.cos(mean), np.sin(mean)
    w = 3.0 * (np.cos(np.radians(theta)) * X) + 2.0 * (np.sin(np.radians(theta)) * Y) + rng.normal(size=len(jd)) * sigma
    result = epoch_astrometry.grid_search(jd, theta, sigma, base, w[:, None], device="cpu", **{**SMALL, "step_share": 6.0, "eccentricities": (0.0,), "phases": 8})
    assert abs(float(result["period"][0]) - period) <= period**2 / (6.0 * float(np.ptp(jd))), "within one frequency step of the grid (about 46 d at 500 d)"
    assert float(result["chi2_base"][0] - result["chi2"][0]) > 500.0


def test_a_chunk_of_one_frequency_gives_the_same_result_as_one_chunk_of_all():
    jd, theta, sigma, base, rng = make_data(5)
    targets = rng.normal(size=(len(jd), 2)) * sigma[:, None]
    whole = epoch_astrometry.grid_search(jd, theta, sigma, base, targets, device="cpu", **SMALL)
    pieces = epoch_astrometry.grid_search(jd, theta, sigma, base, targets, device="cpu", chunk=1, **SMALL)
    assert np.array_equal(whole["chi2"].numpy(), pieces["chi2"].numpy()) or np.allclose(whole["chi2"].numpy(), pieces["chi2"].numpy(), rtol=1e-12)
    assert np.array_equal(whole["period"].numpy(), pieces["period"].numpy())
