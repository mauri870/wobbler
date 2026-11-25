"""wobbler.mass: the mass function, the least companion mass and the share of isotropic orientations below an inclination. Closed forms and round trips."""
import numpy as np
import pytest

from wobbler import mass

pytestmark = pytest.mark.torch_cpu


def test_the_mass_function_of_a_known_orbit():
    """Period 1217.19 d, e 0.08, K 16.75 km/s: 0.587 solar masses (the published orbit of a long-period single-lined binary)."""
    assert mass.mass_function(1217.19, 16.75, 0.08) == pytest.approx(0.587, abs=0.002)


def test_the_least_companion_mass_solves_the_mass_function_at_edge_on():
    f = np.array([0.001, 0.5, 0.876, 3.0])
    m1 = 0.95
    m2 = mass.min_companion_mass(f, m1)
    assert np.allclose(m2**3 / (m1 + m2) ** 2, f, rtol=1e-9)


def test_the_companion_mass_at_an_inclination_is_larger_than_the_least_mass():
    f, m1 = 0.876, 0.95
    least = float(mass.min_companion_mass(f, m1))
    m2 = mass.companion_mass(f, m1, np.array([np.pi / 2, np.pi / 3, 0.3]))
    assert m2[0] == pytest.approx(least, rel=1e-9) and m2[1] > least and m2[2] > m2[1]
    assert np.allclose(m2**3 * np.sin(np.array([np.pi / 2, np.pi / 3, 0.3])) ** 3 / (m1 + m2) ** 2, f, rtol=1e-8)


def test_the_share_of_isotropic_orientations_below_an_inclination():
    assert mass.share_inclined_below(0.0) == 0.0
    assert mass.share_inclined_below(np.pi / 2) == pytest.approx(1.0)
    assert mass.share_inclined_below(np.pi / 3) == pytest.approx(0.5)


def test_a_zero_mass_function_has_no_companion_mass():
    assert mass.min_companion_mass(0.0, 1.0) == 0.0


def test_the_photocentre_ratio_is_one_for_a_dark_companion_and_zero_when_light_follows_mass_and_its_inverse_returns_the_flux_ratio():
    from wobbler import flux_ratio_from_ratio, photocentre_ratio

    assert float(photocentre_ratio(0.0, 1.0)) == pytest.approx(1.0)
    assert float(photocentre_ratio(1.0, 1.0)) == pytest.approx(0.0, abs=1e-15)  # equal masses, equal light
    assert float(photocentre_ratio(0.5, 1.0)) == pytest.approx(1.0 / 3.0)  # light fraction 1/3, mass fraction 1/2: 1 - (1/3)(1 + 1)
    assert float(photocentre_ratio(10.0, 1.0)) < 0
    for flux_ratio, mass_ratio in ((0.0, 0.3), (0.2, 0.5), (0.5, 1.0), (3.0, 2.0)):
        ratio = photocentre_ratio(flux_ratio, mass_ratio)
        assert float(flux_ratio_from_ratio(ratio, mass_ratio)) == pytest.approx(flux_ratio, abs=1e-12)


def test_the_reflex_semi_major_axis_of_a_one_year_orbit_at_the_earths_speed_is_one_au_and_falls_with_the_eccentricity():
    from wobbler import reflex_semi_major_axis

    assert float(reflex_semi_major_axis(29.7847, 365.25636, 0.0, device="cpu")) == pytest.approx(1.0, rel=2e-3)
    assert float(reflex_semi_major_axis(20.0, 500.0, 0.5, device="cpu")) == pytest.approx(float(reflex_semi_major_axis(20.0, 500.0, 0.0, device="cpu")) * np.sqrt(0.75), rel=1e-12)
    out = reflex_semi_major_axis(np.array([10.0, 20.0, 40.0]), 100.0, 0.1, device="cpu")
    assert tuple(out.shape) == (3,) and float(out[2]) == pytest.approx(4 * float(out[0]), rel=1e-12), "linear in the semi-amplitude"
