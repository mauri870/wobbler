"""A companion that carries light: simulate_fit with a flux ratio and the primary's mass (the photocentre orbit shrinks with the light the companion carries)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from wobbler import Orbit, Scan, mass, simulate_fit  # noqa: E402

pytestmark = pytest.mark.torch_cpu


def scan():
    rng = np.random.default_rng(11)  # seed of the synthetic scan
    return Scan(jd=np.sort(rng.uniform(2456891.5, 2457902.0, 60)), psi=rng.uniform(0, np.pi, 60), plx_factor=rng.uniform(-1, 1, 60), device="cpu")


ORBIT = Orbit(period_days=200.0, ecc=0.3, t_peri_days=0.0, omega=0.5)
COMMON = dict(a1_au=1.0, parallax_mas=np.array([2.0]), G=13.0, n_orient=40, seed=4, nu_eff=1.5, device="cpu")


def test_the_photocentre_axis_is_the_primarys_axis_times_the_photocentre_ratio_at_each_inclination():
    fit = simulate_fit(scan(), ORBIT, flux_ratio=0.5, primary_mass=1.0, **COMMON)
    incl = fit.inclination.numpy().ravel()
    a1 = 1.0 / np.sin(incl) * 2.0
    f = 1.0**3 / (200.0 / 365.25) ** 2
    companion = mass.companion_mass(np.full(incl.shape, f), 1.0, incl, device="cpu").numpy()
    expected = a1 * (1 - 0.5 / 1.5 * (1 + 1.0 / companion))
    assert np.allclose(fit.a0_mas.numpy().ravel(), expected, rtol=1e-9), (fit.a0_mas.numpy().ravel()[:3], expected[:3])


def test_a_flux_ratio_of_zero_gives_the_dark_companion_and_needs_no_primary_mass():
    dark = simulate_fit(scan(), ORBIT, **COMMON)
    zero = simulate_fit(scan(), ORBIT, flux_ratio=0.0, **COMMON)
    assert np.array_equal(dark.ruwe.numpy(), zero.ruwe.numpy())


def test_a_flux_ratio_without_the_primary_mass_is_refused_with_the_reason():
    with pytest.raises(ValueError, match="primary_mass"):
        simulate_fit(scan(), ORBIT, flux_ratio=0.3, **COMMON)


def test_a_flux_ratio_that_is_a_function_of_the_companion_mass_is_evaluated_at_each_case():
    seen = []

    def ratio(companion_mass):
        seen.append(np.asarray(companion_mass).shape)
        return 0.2 * np.ones_like(companion_mass)

    simulate_fit(scan(), ORBIT, flux_ratio=ratio, primary_mass=1.0, **COMMON)
    assert seen == [(1, 40)], seen


def test_a_companion_that_carries_the_light_of_its_mass_leaves_no_photocentre_wobble():
    f = 1.0 ** 3 / (200.0 / 365.25) ** 2
    incl = np.array([1.0])
    companion = mass.companion_mass(np.array([f]), 1.0, incl, device="cpu").numpy()
    flux = companion / 1.0  # the light share equals the mass share
    assert abs(float(mass.photocentre_ratio(flux, companion / 1.0, device="cpu"))) < 1e-12
